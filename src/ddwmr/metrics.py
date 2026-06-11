"""Anchor performance metric (Eqs. 36-39) + helpers.

E_i = 1000 * (1.5 * e_l_i / C_l  +  2 * |dtheta_i| / pi  +  2 * tanh(0.1 * t_norm_i))
t_norm_i = | t_end_i - K_t * d_gs |          (Eq. 37)
dtheta_i = wrap(theta_end_i - phi)            (Eq. 39)

C_l is the line-deviation normalization constant. The anchor uses
max_j e_{l,j} over the candidate set; per devil's-advocate condition (iii) we
use FIXED constants taken from the brute-force reference run so the metric is
identical across methods and differentiable-trainable. Component values
(e_line, e_th, e_time) follow the anchor's reporting convention (each scaled
to max 1000, weights applied only in e_total; upper bound 5500).
"""
import numpy as np
from .params import K_T, METRIC_SCALE, TANH_TIMESCALE


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def anchor_metric(e_l, theta_end, t_end, start, target, C_l, *, kt=K_T):
    """Vectorized Eq. 36. All inputs broadcastable (B,). Returns dict."""
    start = np.asarray(start, float).reshape(-1, 2)
    target = np.asarray(target, float).reshape(-1, 2)
    d_gs = np.linalg.norm(target - start, axis=1)
    phi = np.arctan2(target[:, 1] - start[:, 1], target[:, 0] - start[:, 0])
    dth = np.abs(wrap(theta_end - phi))
    t_norm = np.abs(t_end - kt * d_gs)
    e_line = METRIC_SCALE * np.asarray(e_l) / C_l
    e_th = METRIC_SCALE * dth / np.pi
    e_time = METRIC_SCALE * np.tanh(TANH_TIMESCALE * t_norm)
    e_total = 1.5 * e_line + 2.0 * e_th + 2.0 * e_time
    return dict(e_line=e_line, e_th=e_th, e_time=e_time, e_total=e_total,
                t_norm=t_norm, dtheta=dth)
