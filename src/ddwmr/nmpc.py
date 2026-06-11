"""NMPC baseline (WP4) — CasADi + IPOPT multiple shooting.

Blueprint-sanctioned fallback for acados (build not available in this
environment); cf. Frey et al. 2025 for the acados recipe on a voltage-driven
differential-drive robot. Voltage-level inputs, horizon 1 s (N=50, Ts=20 ms),
receding horizon with warm start.

Internal prediction model: 5 states [x y th wr wl] with armature currents
eliminated by quasi-steady-state (La/Ra = 0.67 ms << Ts = 20 ms; documented
time-scale separation). The EVALUATED plant is always the full 7-state model.

Stage cost mirrors the three anchor objectives (documented mapping):
  line adherence : w_perp * perp^2
  elapsed time   : w_v * (v - v_des(d))^2 , v_des = min(1/K_t, 2 d) m/s
  arrival heading: terminal  W_th * wrap(th - phi)^2 (+ weak stage heading)
  + progress d^2 and input regularization; |V| <= 12.
"""
import time
import numpy as np
import casadi as ca

from .params import NOMINAL, motor_coeffs, V_MAX, K_T

_n = NOMINAL
_a11, _a12, _a13, _a14 = motor_coeffs(_n)
TS = 0.02
N_HOR = 50


def _f5(s, u):
    """Quasi-steady-state 5-state dynamics (casadi SX)."""
    th, wr, wl = s[2], s[3], s[4]
    ir = (u[0] - _n.Km * wr) / _n.Ra
    il = (u[1] - _n.Km * wl) / _n.Ra
    dwr = _a11 * wr + _a12 * wl + _a13 * ir + _a14 * il
    dwl = _a12 * wr + _a11 * wl + _a14 * ir + _a13 * il
    v = (_n.r / 2.0) * (wr + wl)
    return ca.vertcat(v * ca.cos(th), v * ca.sin(th),
                      (_n.r / _n.d) * (wr - wl), dwr, dwl)


class NMPCController:
    def __init__(self, target, start=(0.0, 0.0)):
        self.tgt = np.asarray(target, float)
        sx, sy = start
        seg = self.tgt - np.array([sx, sy])
        L = np.linalg.norm(seg)
        self.phi = np.arctan2(seg[1], seg[0])
        ex, ey = seg / max(L, 1e-9)
        self.solve_times = []

        opti = ca.Opti()
        X = opti.variable(5, N_HOR + 1)
        U = opti.variable(2, N_HOR)
        x0p = opti.parameter(5)
        opti.subject_to(X[:, 0] == x0p)
        J = 0
        v_max_des = 1.0 / K_T
        for k in range(N_HOR):
            s = X[:, k]; u = U[:, k]
            # RK4 over Ts on the QSS model
            k1 = _f5(s, u)
            k2 = _f5(s + 0.5 * TS * k1, u)
            k3 = _f5(s + 0.5 * TS * k2, u)
            k4 = _f5(s + TS * k3, u)
            opti.subject_to(X[:, k + 1] == s + (TS / 6) * (k1 + 2 * k2 + 2 * k3 + k4))
            opti.subject_to(opti.bounded(-V_MAX, u, V_MAX))
            dx = self.tgt[0] - s[0]; dy = self.tgt[1] - s[1]
            d2 = dx * dx + dy * dy
            # perpendicular deviation from the start->target line
            relx = s[0] - sx; rely = s[1] - sy
            perp = relx * ey - rely * ex
            v = (_n.r / 2.0) * (s[3] + s[4])
            v_des = ca.fmin(v_max_des, 2.0 * ca.sqrt(d2 + 1e-9))
            ang = s[2] - self.phi
            J += (60.0 * perp ** 2 + 1.0 * d2 + 3.0 * (v - v_des) ** 2
                  + 0.05 * ca.atan2(ca.sin(ang), ca.cos(ang)) ** 2
                  + 1e-4 * (u[0] ** 2 + u[1] ** 2))
        dxT = self.tgt[0] - X[0, -1]; dyT = self.tgt[1] - X[1, -1]
        angT = X[2, -1] - self.phi
        J += 200.0 * (dxT ** 2 + dyT ** 2) \
            + 8.0 * ca.atan2(ca.sin(angT), ca.cos(angT)) ** 2 \
            + 0.5 * (X[3, -1] ** 2 + X[4, -1] ** 2)
        opti.minimize(J)
        opti.solver("ipopt", dict(print_time=False),
                    dict(print_level=0, max_iter=200, warm_start_init_point="yes",
                         mu_init=1e-2, tol=1e-6))
        self.opti, self.X, self.U, self.x0p = opti, X, U, x0p
        self.prev = None

    def step(self, state7):
        """state7 = [x y th wr wl ir il] -> (Vr, Vl); logs solve time."""
        s5 = np.concatenate([state7[:5]])
        self.opti.set_value(self.x0p, s5)
        if self.prev is not None:
            Xp, Up = self.prev
            self.opti.set_initial(self.X, np.hstack([Xp[:, 1:], Xp[:, -1:]]))
            self.opti.set_initial(self.U, np.hstack([Up[:, 1:], Up[:, -1:]]))
        t0 = time.perf_counter()
        try:
            sol = self.opti.solve()
            Xs, Us = sol.value(self.X), sol.value(self.U)
        except RuntimeError:
            Xs = self.opti.debug.value(self.X)
            Us = self.opti.debug.value(self.U)
        self.solve_times.append(time.perf_counter() - t0)
        self.prev = (Xs, Us)
        return float(Us[0, 0]), float(Us[1, 0])


def rollout_nmpc(target, T_max=20.0, start=(0.0, 0.0), theta0=0.0,
                 plant_step=None, tol=0.01, s0=None, noise=None, noise_seed=0):
    """Closed-loop NMPC on the FULL 7-state plant (RK4 at 1 ms, ZOH 20 ms).

    plant_step: optional override for perturbed plants:
        f(state7, Vr, Vl, n_sub) -> new state7
    s0: optional full initial 7-state (overrides start/theta0 for chaining);
    noise: optional dict(pose_noise_xy=, pose_noise_th=) applied to the
    MEASURED state fed to the controller (ZOH per control step).
    Returns dict with trajectory metrics compatible with metrics.anchor_metric.
    """
    from .params import ARRIVAL_TOL
    if s0 is not None:
        start = (float(s0[0]), float(s0[1]))
        theta0 = float(s0[2])
    ctl = NMPCController(target, start)
    nrng = np.random.default_rng(noise_seed)
    s = np.zeros(7) if s0 is None else np.array(s0, float)
    s[0], s[1], s[2] = start[0], start[1], theta0
    a11, a12, a13, a14 = _a11, _a12, _a13, _a14
    km_la, ra_la, inv_la = _n.Km / _n.La, _n.Ra / _n.La, 1.0 / _n.La

    def f7(st, Vr, Vl):
        x, y, th, wr, wl, ir, il = st
        v = (_n.r / 2) * (wr + wl)
        return np.array([
            v * np.cos(th), v * np.sin(th), (_n.r / _n.d) * (wr - wl),
            a11 * wr + a12 * wl + a13 * ir + a14 * il,
            a12 * wr + a11 * wl + a14 * ir + a13 * il,
            -km_la * wr - ra_la * ir + inv_la * Vr,
            -km_la * wl - ra_la * il + inv_la * Vl])

    def default_step(st, Vr, Vl, n_sub):
        h = 0.001
        for _ in range(n_sub):
            k1 = f7(st, Vr, Vl); k2 = f7(st + 0.5 * h * k1, Vr, Vl)
            k3 = f7(st + 0.5 * h * k2, Vr, Vl); k4 = f7(st + h * k3, Vr, Vl)
            st = st + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return st

    stepper = plant_step or default_step
    tgt = np.asarray(target, float)
    seg = tgt - np.asarray(start, float)
    L = np.linalg.norm(seg)
    perp_sum = perp_sq = 0.0
    npts = 0
    energy = 0.0
    peakV = 0.0
    t = 0.0
    arrived = False
    n_ctrl = int(round(T_max / TS))
    for k in range(n_ctrl):
        s_meas = s.copy()
        if noise is not None:
            s_meas[0] += nrng.normal(0, noise.get("pose_noise_xy", 0.0))
            s_meas[1] += nrng.normal(0, noise.get("pose_noise_xy", 0.0))
            s_meas[2] += nrng.normal(0, noise.get("pose_noise_th", 0.0))
        Vr, Vl = ctl.step(s_meas)
        peakV = max(peakV, abs(Vr), abs(Vl))
        energy += abs(Vr * s[5] + Vl * s[6]) * TS
        s = stepper(s, Vr, Vl, int(TS / 0.001))
        t += TS
        perp = abs((s[0] - start[0]) * seg[1] - (s[1] - start[1]) * seg[0]) / max(L, 1e-9)
        perp_sum += perp; perp_sq += perp * perp; npts += 1
        d = np.hypot(tgt[0] - s[0], tgt[1] - s[1])
        if d < tol:
            arrived = True
            break
    st = np.array(ctl.solve_times)
    return dict(arrived=arrived, t_end=t, theta_end=s[2],
                e_l_mean=perp_sum / max(npts, 1),
                e_l_rmse=np.sqrt(perp_sq / max(npts, 1)),
                energy=energy, peak_V=peakV, s_final=s.copy(),
                solve_ms_mean=1e3 * st.mean(), solve_ms_p95=1e3 * np.percentile(st, 95),
                n_solves=len(st))
