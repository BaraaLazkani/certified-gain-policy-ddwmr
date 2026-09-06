"""Differentiable (PyTorch) closed-loop rollout for BPTT gain-policy training.

Integrator design (documented deviation from the spec's literal "torch RK4"):
the 10-state inner block z = [wr wl ir il eps_r eps_l xhat(4)] is LINEAR
time-invariant given the wheel-speed references u = (wr_ref, wl_ref) — the
LQR feedback V = eps - K2 xhat is linear. We therefore discretize it EXACTLY
at dt=1 ms via matrix exponential with ZOH on u (references vary at outer-loop
bandwidth ~ few rad/s, so ZOH at 1 kHz is far inside its validity), and
integrate only the 3 pose states with a trapezoidal/midpoint update. This is
*more* accurate than RK4 on the stiff electrical states (|lambda| ~ 1500/s)
and ~25x cheaper in autograd ops. Cross-validated against the numpy RK4 eval
sim in tests/test_diffsim.py; the exact non-smooth metric from sim.py is the
ONLY metric used for reported evaluations.

Arrival-ball treatment (all confined to d < 1 cm, mirroring the eval halt):
both reference velocities are gated by g(d) = sigmoid((d - tol/2)/(tol/6)),
so the inner LQR parks the wheels; arrival time uses the soft surrogate
t_soft = sum dt * sigmoid((d - tol)/kappa); line-deviation average is
weighted by the same soft not-yet-arrived indicator.

Loss = anchor metric Eq. 36 with FIXED per-target normalization constants C_l
from the brute-force run (devil's-advocate condition iii).
"""
import numpy as np
import torch
from scipy.linalg import expm

from .params import (NOMINAL, A4_NOM, B4_NOM, K2, KE, DT, ARRIVAL_TOL,
                     K_T, METRIC_SCALE, TANH_TIMESCALE)

_n = NOMINAL
KAPPA = ARRIVAL_TOL / 3.0  # wider: forward pass is exact (straight-through)
NZ = 10  # inner LTI states


def _inner_AB(a11, a12, a13, a14, km_la, ra_la, inv_la):
    """Continuous (Az, Bz, Bd_dist) of the inner closed loop.

    z = [wr wl ir il eps_r eps_l h1 h2 h3 h4], u = [wr_ref, wl_ref, dist_acc].
    Plant rows use the (possibly perturbed) coefficients; the CONTROLLER
    blocks (K2, Ke, observer A4/B4, 1/La in observer) are always NOMINAL.
    """
    Az = np.zeros((NZ, NZ))
    Az[0, :4] = [a11, a12, a13, a14]
    Az[1, :4] = [a12, a11, a14, a13]
    Az[2, 0] = -km_la; Az[2, 2] = -ra_la
    Az[3, 1] = -km_la; Az[3, 3] = -ra_la
    # V = eps - K2 h  enters electrical rows with TRUE 1/La
    Az[2, 4] = inv_la; Az[2, 6:10] = -inv_la * K2[0]
    Az[3, 5] = inv_la; Az[3, 6:10] = -inv_la * K2[1]
    # integrators
    Az[4, 0] = -1.0
    Az[5, 1] = -1.0
    # observer (NOMINAL model): h' = A4 h + B4 V + Ke(y - C h)
    Az[6:10, 6:10] = A4_NOM - KE @ np.array([[1., 0, 0, 0], [0, 1., 0, 0]])
    Az[6:10, 0] = KE[:, 0]
    Az[6:10, 1] = KE[:, 1]
    Az[6:10, 4] = B4_NOM[:, 0] * 1.0   # B4 @ V, V = eps - K2 h (nominal 1/La inside B4)
    Az[6:10, 5] = B4_NOM[:, 1] * 1.0
    Az[6:10, 6:10] += -B4_NOM @ K2
    Bz = np.zeros((NZ, 3))
    Bz[4, 0] = 1.0   # eps_r' += wr_ref
    Bz[5, 1] = 1.0
    Bz[0, 2] = 1.0   # disturbance wheel acceleration (common to both wheels)
    Bz[1, 2] = 1.0
    return Az, Bz


def discretize(a11=None, a12=None, a13=None, a14=None, km_la=None,
               ra_la=None, inv_la=None, dt=DT):
    """Exact ZOH discretization. Scalars -> (Ad, Bd) torch float32."""
    from .params import motor_coeffs
    if a11 is None:
        a11, a12, a13, a14 = motor_coeffs(_n)
        km_la, ra_la, inv_la = _n.Km / _n.La, _n.Ra / _n.La, 1.0 / _n.La
    Az, Bz = _inner_AB(a11, a12, a13, a14, km_la, ra_la, inv_la)
    M = np.zeros((NZ + 3, NZ + 3))
    M[:NZ, :NZ] = Az
    M[:NZ, NZ:] = Bz
    Md = expm(M * dt)
    Ad = torch.tensor(Md[:NZ, :NZ], dtype=torch.float32)
    Bd = torch.tensor(Md[:NZ, NZ:], dtype=torch.float32)
    return Ad, Bd


def discretize_batch(samples, dt=DT):
    """Per-row perturbed plants (WP3 DR). samples: dict of (B,) numpy arrays
    with keys m, Ig, d, beta, Iw, r, Km, La, Ra. Returns (B,NZ,NZ), (B,NZ,3),
    plus per-row kinematic (r, d_track, inv_eff) tensors."""
    m, Ig, dtr, beta, Iw, r, Km, La, Ra = (np.asarray(samples[k], float) for k in
                                           ("m", "Ig", "d", "beta", "Iw", "r", "Km", "La", "Ra"))
    pp = 0.25 * m * r**2 + (r**2 / dtr**2) * Ig + Iw
    qq = 0.25 * m * r**2 - (r**2 / dtr**2) * Ig
    det = pp**2 - qq**2
    B = len(m)
    Ms = np.zeros((B, NZ + 3, NZ + 3))
    for i in range(B):
        Az, Bz = _inner_AB(-beta[i] * pp[i] / det[i], beta[i] * qq[i] / det[i],
                           Km[i] * pp[i] / det[i], -Km[i] * qq[i] / det[i],
                           Km[i] / La[i], Ra[i] / La[i], 1.0 / La[i])
        Ms[i, :NZ, :NZ] = Az
        Ms[i, :NZ, NZ:] = Bz
    Md = torch.matrix_exp(torch.tensor(Ms * dt, dtype=torch.float64)).to(torch.float32)
    inv_eff = (pp - qq) / det   # multiply raw torque by this before passing as u3
    return (Md[:, :NZ, :NZ], Md[:, :NZ, NZ:],
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(dtr, dtype=torch.float32),
            torch.tensor(inv_eff, dtype=torch.float32))


def _wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


def rollout_metric(gains, targets, C_l, *, T=12.0, dt=DT, AdBd=None,
                   r_true=None, d_true=None, kt=K_T, start=None, theta0=None,
                   noise=None, slip=None, dist_acc_seq=None,
                   return_parts=False):
    """Differentiable anchor metric Eq. 36 for a batch of single-target runs.

    gains (B,2) [grad path], targets (B,2), C_l (B,) fixed constants.
    AdBd: (Ad, Bd) shared or ((B,NZ,NZ),(B,NZ,3)) per-row for DR.
    noise: optional dict(xy=std_m, th=std_rad, gen=torch.Generator) — resampled
    each step, ZOH (matches eval sim).
    slip: optional (B,2) per-row multiplicative wheel factors (right,left),
    or callable(step)->.(B,2).
    dist_acc_seq: optional callable(step)->(B,) disturbance input (3rd u col).
    """
    Bn = targets.shape[0]
    Kp, Kth = gains[:, 0], gains[:, 1]
    if AdBd is None:
        AdBd = discretize()
    Ad, Bd = AdBd
    batched = Ad.dim() == 3
    r_t = r_true if r_true is not None else torch.tensor(_n.r)
    d_t = d_true if d_true is not None else torch.tensor(_n.d)

    z = torch.zeros(Bn, NZ)
    x = torch.zeros(Bn) if start is None else start[:, 0].clone()
    y = torch.zeros(Bn) if start is None else start[:, 1].clone()
    th = torch.zeros(Bn) if theta0 is None else theta0.clone()

    n_steps = int(round(T / dt))
    d_gs = torch.linalg.norm(targets, dim=1) if start is None else \
        torch.linalg.norm(targets - start, dim=1)
    sx = torch.zeros(Bn) if start is None else start[:, 0]
    sy = torch.zeros(Bn) if start is None else start[:, 1]
    phi = torch.atan2(targets[:, 1] - sy, targets[:, 0] - sx)
    inv_dgs = 1.0 / torch.clamp(d_gs, min=1e-9)

    perp_acc = torch.zeros(Bn)
    w_acc = torch.zeros(Bn)
    t_soft = torch.zeros(Bn)
    inv_r = 1.0 / _n.r
    half_track = 0.5 * _n.d

    v_prev = (r_t / 2.0) * (z[:, 0] + z[:, 1])
    w_prev = (r_t / d_t) * (z[:, 0] - z[:, 1])
    dmin = torch.full((Bn,), 1e9)

    for k in range(n_steps):
        ex = targets[:, 0] - x
        ey = targets[:, 1] - y
        d = torch.sqrt(ex * ex + ey * ey + 1e-12)
        dmin = torch.minimum(dmin, d)
        # latching halt, straight-through: forward pass is the eval sim's hard
        # halt-at-first-crossing; gradient flows through the soft sigmoid
        alive_soft = torch.sigmoid((dmin - ARRIVAL_TOL) / KAPPA)
        alive = alive_soft + ((dmin > ARRIVAL_TOL).float() - alive_soft).detach()
        xm, ym, thm = x, y, th
        if noise is not None:
            xm = x + torch.randn(Bn, generator=noise.get("gen")) * noise.get("xy", 0.0)
            ym = y + torch.randn(Bn, generator=noise.get("gen")) * noise.get("xy", 0.0)
            thm = th + torch.randn(Bn, generator=noise.get("gen")) * noise.get("th", 0.0)
            exm = targets[:, 0] - xm; eym = targets[:, 1] - ym
            dm = torch.sqrt(exm * exm + eym * eym + 1e-12)
            eth = _wrap(torch.atan2(eym, exm) - thm)
        else:
            dm = d
            eth = _wrap(torch.atan2(ey, ex) - th)
        v_ref = Kp * dm * torch.cos(eth) * alive
        w_ref = (Kp * torch.cos(eth) * torch.sin(eth) + Kth * eth) * alive
        u1 = (v_ref + half_track * w_ref) * inv_r
        u2 = (v_ref - half_track * w_ref) * inv_r
        u3 = dist_acc_seq(k) if dist_acc_seq is not None else torch.zeros(Bn)
        u = torch.stack([u1, u2, u3], dim=1)

        if batched:
            z = torch.bmm(Ad, z.unsqueeze(2)).squeeze(2) + \
                torch.bmm(Bd, u.unsqueeze(2)).squeeze(2)
        else:
            z = z @ Ad.T + u @ Bd.T

        wr, wl = z[:, 0], z[:, 1]
        if slip is not None:
            s = slip(k) if callable(slip) else slip
            wr = wr * s[:, 0]
            wl = wl * s[:, 1]
        v_now = (r_t / 2.0) * (wr + wl)
        w_now = (r_t / d_t) * (wr - wl)
        th_new = th + 0.5 * dt * (w_prev + w_now)
        th_mid = 0.5 * (th + th_new)
        v_mid = 0.5 * (v_prev + v_now)
        x = x + dt * v_mid * torch.cos(th_mid)
        y = y + dt * v_mid * torch.sin(th_mid)
        th = th_new
        v_prev, w_prev = v_now, w_now

        t_soft = t_soft + dt * alive
        perp = torch.abs((x - sx) * (targets[:, 1] - sy)
                         - (y - sy) * (targets[:, 0] - sx)) * inv_dgs
        perp_acc = perp_acc + alive * perp
        w_acc = w_acc + alive

    e_l = perp_acc / torch.clamp(w_acc, min=1.0)
    dth = torch.abs(_wrap(th - phi))
    t_norm = torch.abs(t_soft - kt * d_gs)
    e_line = METRIC_SCALE * e_l / C_l
    e_th = METRIC_SCALE * dth / np.pi
    e_time = METRIC_SCALE * torch.tanh(TANH_TIMESCALE * t_norm)
    e_total = 1.5 * e_line + 2.0 * e_th + 2.0 * e_time
    if return_parts:
        return e_total, dict(e_line=e_line, e_th=e_th, e_time=e_time,
                             t_soft=t_soft, e_l=e_l, theta_end=th)
    return e_total
