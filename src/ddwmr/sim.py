"""Vectorized batch closed-loop simulator for the anchor DDWMR plant.

One engine covers single-target runs (WP0/WP1/WP3/WP4) and waypoint sequences
with gain switching (WP2/WP5). Fixed-step RK4 at dt=1 ms; controller feedback
is continuous (static maps + integrator/observer states inside the ODE), which
mirrors the anchor's Simulink implementation; exogenous signals (sensor noise,
slip, disturbance) are zero-order-held over each step.

ODE state layout per rollout (15 columns):
  0:x 1:y 2:theta | 3:wr 4:wl 5:ir 6:il | 7:eps_r 8:eps_l |
  9-12: observer xhat(wr,wl,ir,il) | 13: I_d (PID) 14: I_eth (PID)
"""
from dataclasses import dataclass, field
import numpy as np

from .params import (NOMINAL, PlantParams, motor_coeffs, A4_NOM, B4_NOM, K2,
                     KE, DT, ARRIVAL_TOL, V_MAX, PID_TABLE3)

# ------------------------------------------------------------------ controllers


@dataclass
class LyapGains:
    """Per-rollout NPC gains (Eq. 25-26). Arrays broadcast to batch."""
    Kp: np.ndarray
    Kth: np.ndarray


@dataclass
class PIDGains:
    """Dual-PID (Table 3): v = PID1(d), w = PID2(e_theta)."""
    KP1: float = PID_TABLE3["KP1"]; KI1: float = PID_TABLE3["KI1"]; KD1: float = PID_TABLE3["KD1"]
    KP2: float = PID_TABLE3["KP2"]; KI2: float = PID_TABLE3["KI2"]; KD2: float = PID_TABLE3["KD2"]


@dataclass
class Disturbances:
    """Exogenous per-rollout disturbance spec (all optional; WP3)."""
    pose_noise_xy: float = 0.0        # std [m], resampled each step (zero-mean)
    pose_noise_th: float = 0.0        # std [rad]
    bias_xy: np.ndarray | None = None   # (B,2) constant additive sensor bias [m]
    bias_th: np.ndarray | None = None   # (B,) constant heading bias [rad]
    slip_common: np.ndarray | None = None   # (B,) s; both wheels scaled by (1-s)
    slip_diff: np.ndarray | None = None     # (B,) s; right *(1-s), left *(1+s)
    slip_duty: float = 1.0            # fraction of time slip is active (intermittent)
    slip_period: float = 1.0          # s, on/off cycle for intermittent slip
    dist_torque: np.ndarray | None = None   # (B,) step disturbance torque [N m]
    dist_t0: np.ndarray | None = None       # (B,) onset time [s]
    dist_dur: float = 0.5             # s


@dataclass
class PerturbedPlant:
    """Per-rollout physical coefficients (controller stays nominal)."""
    a11: np.ndarray; a12: np.ndarray; a13: np.ndarray; a14: np.ndarray
    km_la: np.ndarray; ra_la: np.ndarray; inv_la: np.ndarray
    r: np.ndarray; d: np.ndarray
    inv_eff: np.ndarray  # M^-1 row sums for disturbance torque injection (p-q)/det

    @classmethod
    def from_samples(cls, m, Ig, d, beta, Iw, r, Km, La, Ra):
        pp = 0.25 * m * r**2 + (r**2 / d**2) * Ig + Iw
        qq = 0.25 * m * r**2 - (r**2 / d**2) * Ig
        det = pp**2 - qq**2
        return cls(a11=-beta * pp / det, a12=beta * qq / det,
                   a13=Km * pp / det, a14=-Km * qq / det,
                   km_la=Km / La, ra_la=Ra / La, inv_la=1.0 / La,
                   r=r, d=d, inv_eff=(pp - qq) / det)

    @classmethod
    def nominal(cls):
        n = NOMINAL
        one = np.float64(1.0)
        return cls.from_samples(n.m * one, n.Ig * one, n.d * one, n.beta * one,
                                n.Iw * one, n.r * one, n.Km * one, n.La * one, n.Ra * one)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _npc_law(d, eth, Kp, Kth):
    """Eqs. 25-26."""
    v_ref = Kp * d * np.cos(eth)
    w_ref = Kp * np.cos(eth) * np.sin(eth) + Kth * eth
    return v_ref, w_ref


def _pid_law(d, eth, Id, Ieth, v_act, w_act, g: PIDGains):
    """Dual decoupled PID (anchor §7.1). Derivative terms use the measurable
    rates ddot = -v cos(e_th) (Eq. 32) and edot_th = v sin(e_th)/d - w (Eq. 33)."""
    ddot = -v_act * np.cos(eth)
    ethdot = v_act * np.sin(eth) / np.maximum(d, 1e-6) - w_act
    v_ref = g.KP1 * d + g.KI1 * Id + g.KD1 * ddot
    w_ref = g.KP2 * eth + g.KI2 * Ieth + g.KD2 * ethdot
    return v_ref, w_ref


# ------------------------------------------------------------------ derivative

def _deriv(S, targets, ctrl_kind, gains, plant: PerturbedPlant, meas_off,
           slip_r, slip_l, dist_acc):
    """Time derivative of the full closed loop for active rollouts.

    S: (B,15); targets: (B,2); meas_off: (B,3) sensor offset (noise+bias), ZOH;
    slip_r/slip_l: (B,) multiplicative wheel-speed factors for kinematics;
    dist_acc: (B,) additive wheel angular acceleration from disturbance torque.
    Returns dS (B,15) and aux dict (voltages, refs, actuals) for metrics.
    """
    x, y, th = S[:, 0], S[:, 1], S[:, 2]
    wr, wl, ir, il = S[:, 3], S[:, 4], S[:, 5], S[:, 6]
    eps = S[:, 7:9]
    xhat = S[:, 9:13]
    Id, Ieth = S[:, 13], S[:, 14]

    # --- measured pose (sensor model) ---
    xm = x + meas_off[:, 0]; ym = y + meas_off[:, 1]; thm = th + meas_off[:, 2]

    ex = targets[:, 0] - xm
    ey = targets[:, 1] - ym
    d = np.sqrt(ex * ex + ey * ey)
    eth = _wrap(np.arctan2(ey, ex) - thm)

    # actual body velocities (encoder-measured, used by PID derivative terms)
    n = NOMINAL
    v_act = (n.r / 2.0) * (wr + wl)
    w_act = (n.r / n.d) * (wr - wl)

    if ctrl_kind == "lyap":
        v_ref, w_ref = _npc_law(d, eth, gains.Kp, gains.Kth)
        dId = np.zeros_like(d); dIeth = np.zeros_like(d)
    else:
        v_ref, w_ref = _pid_law(d, eth, Id, Ieth, v_act, w_act, gains)
        dId = d; dIeth = eth

    # geometry matrix (controller-side, NOMINAL r and d)
    wr_ref = (v_ref + 0.5 * n.d * w_ref) / n.r
    wl_ref = (v_ref - 0.5 * n.d * w_ref) / n.r

    # --- inner loop: u = -K2 xhat + eps  (Eqs. 46-49, integral gain = +I) ---
    V = eps - xhat @ K2.T
    V = np.clip(V, -V_MAX, V_MAX)
    Vr, Vl = V[:, 0], V[:, 1]

    # --- true plant (possibly perturbed) ---
    p = plant
    dwr = p.a11 * wr + p.a12 * wl + p.a13 * ir + p.a14 * il + dist_acc
    dwl = p.a12 * wr + p.a11 * wl + p.a14 * ir + p.a13 * il + dist_acc
    dir_ = -p.km_la * wr - p.ra_la * ir + p.inv_la * Vr
    dil = -p.km_la * wl - p.ra_la * il + p.inv_la * Vl

    # kinematics with true r,d and wheel slip
    wr_eff = wr * slip_r
    wl_eff = wl * slip_l
    v_true = (p.r / 2.0) * (wr_eff + wl_eff)
    dx = v_true * np.cos(th)
    dy = v_true * np.sin(th)
    dth = (p.r / p.d) * (wr_eff - wl_eff)

    # integrators (Eqs. 44-45), measured wheel speeds
    deps_r = wr_ref - wr
    deps_l = wl_ref - wl

    # observer (Eqs. 50-51), NOMINAL model
    y_meas = np.stack([wr, wl], axis=1)
    innov = y_meas - xhat[:, :2]
    dxhat = xhat @ A4_NOM.T + V @ B4_NOM.T + innov @ KE.T

    dS = np.empty_like(S)
    dS[:, 0], dS[:, 1], dS[:, 2] = dx, dy, dth
    dS[:, 3], dS[:, 4], dS[:, 5], dS[:, 6] = dwr, dwl, dir_, dil
    dS[:, 7], dS[:, 8] = deps_r, deps_l
    dS[:, 9:13] = dxhat
    dS[:, 13], dS[:, 14] = dId, dIeth
    aux = dict(V=V, v_ref=v_ref, w_ref=w_ref, v_act=v_act, w_act=w_act,
               d=d, eth=eth, ir=ir, il=il)
    return dS, aux


# ------------------------------------------------------------------ engine

@dataclass
class RunResult:
    arrived: np.ndarray        # bool (B,) — final waypoint reached
    t_end: np.ndarray          # (B,) arrival time of final waypoint (or T_max)
    theta_end: np.ndarray      # (B,) heading at final arrival
    e_l_mean: np.ndarray       # (B,) mean |perp dev| over whole run [m]
    e_l_rmse: np.ndarray       # (B,) cross-track RMSE [m]
    energy: np.ndarray         # (B,) ∫(Vr ir + Vl il) dt [J]
    peak_V: np.ndarray         # (B,) max |V| [V]
    sup_dv: np.ndarray         # (B,) sup |v - v_ref|
    rms_dv: np.ndarray
    sup_dw: np.ndarray         # (B,) sup |w - w_ref|
    rms_dw: np.ndarray
    diverged: np.ndarray       # bool (B,) — non-finite state or runaway
    seg_records: list | None = None   # per-segment dicts (waypoint mode)
    series: dict | None = None        # recorded time series (subset)


def run_closedloop(targets, ctrl_kind, gains, *,
                   waypoints=None,            # (B,K,2) overrides targets if given
                   n_waypoints=None,          # (B,) valid count per row
                   gains_per_segment=None,    # (B,K,2) Lyap gains per segment
                   gain_policy=None,          # callable(pose (n,3), tgt (n,2)) -> (n,2)
                   switch_policy="halt1cm",   # halt1cm | tol5cm | timer
                   switch_timer=4.0,          # s, for timer policy
                   start_pose=None,           # (B,3)
                   T_max=25.0, dt=DT,
                   plant: PerturbedPlant | None = None,
                   dist: Disturbances | None = None,
                   rng=None,
                   record_series=0,           # record series for first N rows
                   series_stride=10,          # steps between samples (10 -> 100 Hz)
                   compact_every=2000):
    """Simulate a batch to completion. Single-target if waypoints is None."""
    plant = plant or PerturbedPlant.nominal()
    dist = dist or Disturbances()
    rng = rng or np.random.default_rng(0)

    seq_mode = waypoints is not None
    if seq_mode:
        B, Kmax, _ = waypoints.shape
        n_wp = n_waypoints if n_waypoints is not None else np.full(B, Kmax)
        wp_idx = np.zeros(B, dtype=int)
        cur_tgt = waypoints[np.arange(B), 0].copy()
    else:
        cur_tgt = np.array(targets, dtype=float).reshape(-1, 2).copy()
        B = cur_tgt.shape[0]

    S = np.zeros((B, 15))
    if start_pose is not None:
        S[:, :3] = start_pose
    seg_start = S[:, :2].copy()
    seg_t0 = np.zeros(B)
    tol = ARRIVAL_TOL if switch_policy != "tol5cm" else 0.05

    # per-segment gains
    def seg_gains(rows, k_idx):
        if gains_per_segment is not None:
            g = gains_per_segment[rows, k_idx]
            return g[:, 0], g[:, 1]
        return None

    if ctrl_kind == "lyap":
        Kp = np.broadcast_to(np.asarray(gains.Kp, float), (B,)).copy()
        Kth = np.broadcast_to(np.asarray(gains.Kth, float), (B,)).copy()
        if seq_mode and gains_per_segment is not None:
            Kp[:], Kth[:] = gains_per_segment[:, 0, 0], gains_per_segment[:, 0, 1]
        if gain_policy is not None:
            g0 = gain_policy(S[:, :3], cur_tgt)
            Kp[:], Kth[:] = g0[:, 0], g0[:, 1]
        glive = LyapGains(Kp=Kp, Kth=Kth)
    else:
        glive = gains

    n_steps = int(round(T_max / dt))
    # global accumulators (indexed by original row id)
    acc = dict(
        perp_sum=np.zeros(B), perp_sq=np.zeros(B), n_pts=np.zeros(B),
        energy=np.zeros(B), peak_V=np.zeros(B),
        sup_dv=np.zeros(B), sq_dv=np.zeros(B),
        sup_dw=np.zeros(B), sq_dw=np.zeros(B),
        t_end=np.full(B, T_max), th_end=np.zeros(B),
        arrived=np.zeros(B, bool), diverged=np.zeros(B, bool),
    )
    seg_records = [] if seq_mode else None

    # series recording
    nrec = min(record_series, B)
    series = None
    if nrec > 0:
        nsamp = n_steps // series_stride + 1
        series = dict(t=np.zeros(nsamp),
                      V_lyap=np.full((nsamp, nrec), np.nan),
                      d=np.full((nsamp, nrec), np.nan),
                      eth=np.full((nsamp, nrec), np.nan),
                      dv=np.full((nsamp, nrec), np.nan),
                      dw=np.full((nsamp, nrec), np.nan),
                      x=np.full((nsamp, nrec), np.nan),
                      y=np.full((nsamp, nrec), np.nan))
        si = 0

    active = np.arange(B)          # original row ids of live rollouts
    h = dt
    t = 0.0
    timer_elapsed = np.zeros(B)

    for step in range(n_steps):
        if active.size == 0:
            break
        rowsel = active
        Ssub = S[rowsel]
        tg = cur_tgt[rowsel]
        if ctrl_kind == "lyap":
            gsub = LyapGains(Kp=glive.Kp[rowsel], Kth=glive.Kth[rowsel])
        else:
            gsub = glive

        # ---- exogenous signals, ZOH over the step ----
        nB = rowsel.size
        moff = np.zeros((nB, 3))
        if dist.pose_noise_xy > 0:
            moff[:, :2] = rng.normal(0, dist.pose_noise_xy, (nB, 2))
        if dist.pose_noise_th > 0:
            moff[:, 2] = rng.normal(0, dist.pose_noise_th, nB)
        if dist.bias_xy is not None:
            moff[:, :2] += dist.bias_xy[rowsel]
        if dist.bias_th is not None:
            moff[:, 2] += dist.bias_th[rowsel]

        slip_on = 1.0
        if dist.slip_duty < 1.0:
            slip_on = float((t % dist.slip_period) < dist.slip_duty * dist.slip_period)
        sr = np.ones(nB); sl = np.ones(nB)
        if dist.slip_common is not None:
            sc = dist.slip_common[rowsel] * slip_on
            sr *= (1 - sc); sl *= (1 - sc)
        if dist.slip_diff is not None:
            sd = dist.slip_diff[rowsel] * slip_on
            sr *= (1 - sd); sl *= (1 + sd)

        dacc = np.zeros(nB)
        if dist.dist_torque is not None:
            on = ((t >= dist.dist_t0[rowsel]) &
                  (t < dist.dist_t0[rowsel] + dist.dist_dur))
            dacc = dist.dist_torque[rowsel] * plant_field(plant, "inv_eff", rowsel) * on

        psub = subset_plant(plant, rowsel)

        # ---- RK4 ----
        k1, aux = _deriv(Ssub, tg, ctrl_kind, gsub, psub, moff, sr, sl, dacc)
        k2_, _ = _deriv(Ssub + 0.5 * h * k1, tg, ctrl_kind, gsub, psub, moff, sr, sl, dacc)
        k3, _ = _deriv(Ssub + 0.5 * h * k2_, tg, ctrl_kind, gsub, psub, moff, sr, sl, dacc)
        k4, _ = _deriv(Ssub + h * k3, tg, ctrl_kind, gsub, psub, moff, sr, sl, dacc)
        Snew = Ssub + (h / 6.0) * (k1 + 2 * k2_ + 2 * k3 + k4)
        S[rowsel] = Snew
        t += h
        timer_elapsed[rowsel] += h

        # ---- metric accumulation (start-of-step aux) ----
        st = seg_start[rowsel]
        seg_vec = tg - st
        seg_len = np.maximum(np.linalg.norm(seg_vec, axis=1), 1e-9)
        rel = Ssub[:, :2] - st
        perp = np.abs(rel[:, 0] * seg_vec[:, 1] - rel[:, 1] * seg_vec[:, 0]) / seg_len
        acc["perp_sum"][rowsel] += perp
        acc["perp_sq"][rowsel] += perp * perp
        acc["n_pts"][rowsel] += 1
        pwr = aux["V"][:, 0] * aux["ir"] + aux["V"][:, 1] * aux["il"]
        acc["energy"][rowsel] += np.abs(pwr) * h
        acc["peak_V"][rowsel] = np.maximum(acc["peak_V"][rowsel],
                                           np.abs(aux["V"]).max(axis=1))
        dv = np.abs(aux["v_act"] - aux["v_ref"])
        dw = np.abs(aux["w_act"] - aux["w_ref"])
        acc["sup_dv"][rowsel] = np.maximum(acc["sup_dv"][rowsel], dv)
        acc["sq_dv"][rowsel] += dv * dv
        acc["sup_dw"][rowsel] = np.maximum(acc["sup_dw"][rowsel], dw)
        acc["sq_dw"][rowsel] += dw * dw

        # ---- series ----
        if nrec > 0 and step % series_stride == 0:
            recmask = rowsel < nrec
            rr = rowsel[recmask]
            series["t"][si] = t
            Vl_ = 0.5 * (aux["d"] ** 2 + aux["eth"] ** 2)
            series["V_lyap"][si, rr] = Vl_[recmask]
            series["d"][si, rr] = aux["d"][recmask]
            series["eth"][si, rr] = aux["eth"][recmask]
            series["dv"][si, rr] = (aux["v_act"] - aux["v_ref"])[recmask]
            series["dw"][si, rr] = (aux["w_act"] - aux["w_ref"])[recmask]
            series["x"][si, rr] = Ssub[recmask, 0]
            series["y"][si, rr] = Ssub[recmask, 1]
            si += 1

        # ---- divergence check ----
        bad = ~np.isfinite(Snew).all(axis=1) | (np.abs(Snew[:, :2]).max(axis=1) > 50)
        if bad.any():
            ids = rowsel[bad]
            acc["diverged"][ids] = True
            acc["t_end"][ids] = t
            active = active[~np.isin(active, ids)]
            if active.size == 0:
                break
            continue_mask = ~bad
        else:
            continue_mask = None

        # ---- arrival / switching ----
        exn = cur_tgt[rowsel, 0] - Snew[:, 0]
        eyn = cur_tgt[rowsel, 1] - Snew[:, 1]
        dn = np.sqrt(exn * exn + eyn * eyn)
        if switch_policy == "timer" and seq_mode:
            hit = timer_elapsed[rowsel] >= switch_timer
        else:
            hit = dn < tol
        if continue_mask is not None:
            hit &= continue_mask
        if hit.any():
            ids = rowsel[hit]
            if seq_mode:
                # V = 0.5 (d^2 + e_theta^2) just before the switch
                exb = cur_tgt[ids, 0] - S[ids, 0]
                eyb = cur_tgt[ids, 1] - S[ids, 1]
                db = np.sqrt(exb * exb + eyb * eyb)
                ethb = _wrap(np.arctan2(eyb, exb) - S[ids, 2])
                V_before = 0.5 * (db * db + ethb * ethb)
                recs = {}
                for i, rid in enumerate(ids):
                    k = wp_idx[rid]
                    recs[rid] = dict(
                        row=int(rid), seg=int(k),
                        Lk=float(np.linalg.norm(cur_tgt[rid] - seg_start[rid])),
                        Kp=float(glive.Kp[rid]) if ctrl_kind == "lyap" else np.nan,
                        Kth=float(glive.Kth[rid]) if ctrl_kind == "lyap" else np.nan,
                        t_arrive=float(t - seg_t0[rid]),
                        d_at_switch=float(db[i]),
                        V_before=float(V_before[i]), V_after=np.nan,
                        eps=tol, t_switch=float(t))
                wp_idx[ids] += 1
                done = wp_idx[ids] >= n_wp[ids]
                fin = ids[done]
                if fin.size:
                    acc["arrived"][fin] = True
                    acc["t_end"][fin] = t
                    acc["th_end"][fin] = S[fin, 2]
                    active = active[~np.isin(active, fin)]
                cont = ids[~done]
                if cont.size:
                    seg_start[cont] = S[cont, :2]
                    seg_t0[cont] = t
                    timer_elapsed[cont] = 0.0
                    cur_tgt[cont] = waypoints[cont, wp_idx[cont]]
                    if ctrl_kind == "lyap" and gains_per_segment is not None:
                        g = gains_per_segment[cont, np.minimum(wp_idx[cont],
                                                gains_per_segment.shape[1] - 1)]
                        glive.Kp[cont] = g[:, 0]
                        glive.Kth[cont] = g[:, 1]
                    if ctrl_kind == "lyap" and gain_policy is not None:
                        gnew = gain_policy(S[cont, :3], cur_tgt[cont])
                        glive.Kp[cont] = gnew[:, 0]
                        glive.Kth[cont] = gnew[:, 1]
                    # V just after retarget (same pose, new target, new gains)
                    exa = cur_tgt[cont, 0] - S[cont, 0]
                    eya = cur_tgt[cont, 1] - S[cont, 1]
                    da = np.sqrt(exa * exa + eya * eya)
                    etha = _wrap(np.arctan2(eya, exa) - S[cont, 2])
                    Va = 0.5 * (da * da + etha * etha)
                    for i, rid in enumerate(cont):
                        recs[rid]["V_after"] = float(Va[i])
                seg_records.extend(recs.values())
            else:
                acc["arrived"][ids] = True
                acc["t_end"][ids] = t
                acc["th_end"][ids] = S[ids, 2]
                active = active[~np.isin(active, ids)]

    notarr = ~acc["arrived"]
    acc["th_end"][notarr] = np.where(np.isfinite(S[notarr, 2]), S[notarr, 2], 0.0)
    npts = np.maximum(acc["n_pts"], 1)
    if series is not None:
        for k in series:
            series[k] = series[k][:si] if k != "t" else series[k][:si]
    return RunResult(
        arrived=acc["arrived"], t_end=acc["t_end"], theta_end=acc["th_end"],
        e_l_mean=acc["perp_sum"] / npts,
        e_l_rmse=np.sqrt(acc["perp_sq"] / npts),
        energy=acc["energy"], peak_V=acc["peak_V"],
        sup_dv=acc["sup_dv"], rms_dv=np.sqrt(acc["sq_dv"] / npts),
        sup_dw=acc["sup_dw"], rms_dw=np.sqrt(acc["sq_dw"] / npts),
        diverged=acc["diverged"], seg_records=seg_records, series=series)


def subset_plant(p: PerturbedPlant, rows):
    def pick(v):
        v = np.asarray(v)
        return v if v.ndim == 0 else v[rows]
    return PerturbedPlant(a11=pick(p.a11), a12=pick(p.a12), a13=pick(p.a13),
                          a14=pick(p.a14), km_la=pick(p.km_la), ra_la=pick(p.ra_la),
                          inv_la=pick(p.inv_la), r=pick(p.r), d=pick(p.d),
                          inv_eff=pick(p.inv_eff))


def plant_field(p: PerturbedPlant, name, rows):
    v = np.asarray(getattr(p, name))
    return v if v.ndim == 0 else v[rows]
