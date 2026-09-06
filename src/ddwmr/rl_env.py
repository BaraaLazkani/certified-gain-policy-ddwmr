"""Gymnasium env for the SAC/PPO baseline (WP4).

Per the experiment spec: observation (e_x, e_y, e_theta, w_r, w_l), action (V_r, V_l)
in [-12,12] V, reward = negative incremental anchor metric:
  running:  -(1.5 * perp/C_l + lambda_t) * dt * SCALE   (line + elapsed-time)
  terminal (arrival): -(2 * |dtheta|/pi + 2 * tanh(0.1*t_norm)) * SCALE
  terminal (timeout): -(2 * 1.0 + 2 * tanh(0.1*t_norm)) * SCALE  (worst heading)
with C_l = 1.0 m fixed, SCALE = 1000 matching Eq. 36's components; the
running line-term integrates to 1.5*e_line and the elapsed-time term is paid
at termination, so episode return == -E(Eq. 36, C_l=1m) up to the running
lambda_t shaping constant (documented; identical across seeds/controllers).

Plant: raw 7-state model [x y th wr wl ir il] with RK4 at 1 ms, control ZOH
at 10 ms (the RL agent replaces NPC + LQR + observer entirely).
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .params import NOMINAL, motor_coeffs, ARRIVAL_TOL, V_MAX, K_T

_n = NOMINAL
_a11, _a12, _a13, _a14 = motor_coeffs(_n)
_KM_LA, _RA_LA, _INV_LA = _n.Km / _n.La, _n.Ra / _n.La, 1.0 / _n.La
CTRL_DT = 0.01
SIM_DT = 0.001
SCALE = 1000.0
LAMBDA_T = 0.05      # running time-pressure shaping (per second, x SCALE/1000)


def _deriv(s, Vr, Vl):
    x, y, th, wr, wl, ir, il = s
    dwr = _a11 * wr + _a12 * wl + _a13 * ir + _a14 * il
    dwl = _a12 * wr + _a11 * wl + _a14 * ir + _a13 * il
    dir_ = -_KM_LA * wr - _RA_LA * ir + _INV_LA * Vr
    dil = -_KM_LA * wl - _RA_LA * il + _INV_LA * Vl
    v = (_n.r / 2.0) * (wr + wl)
    return np.array([v * np.cos(th), v * np.sin(th),
                     (_n.r / _n.d) * (wr - wl), dwr, dwl, dir_, dil])


class DDWMRPointEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed=None, t_max=15.0):
        super().__init__()
        self.observation_space = spaces.Box(-np.inf, np.inf, (5,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.rng = np.random.default_rng(seed)
        self.t_max = t_max
        self.substeps = int(round(CTRL_DT / SIM_DT))

    def _obs(self):
        ex = self.tgt[0] - self.s[0]
        ey = self.tgt[1] - self.s[1]
        eth = np.arctan2(ey, ex) - self.s[2]
        eth = (eth + np.pi) % (2 * np.pi) - np.pi
        return np.array([ex, ey, eth, self.s[3], self.s[4]], np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        ang = self.rng.uniform(-0.7 * np.pi, 0.7 * np.pi)
        rad = self.rng.uniform(0.1, 1.0)
        self.tgt = np.array([rad * np.cos(ang), rad * np.sin(ang)])
        self.d_gs = rad
        self.phi = ang
        self.s = np.zeros(7)
        self.t = 0.0
        return self._obs(), {}

    def step(self, action):
        Vr, Vl = np.clip(action, -1, 1) * V_MAX
        perp_acc = 0.0
        for _ in range(self.substeps):
            k1 = _deriv(self.s, Vr, Vl)
            k2 = _deriv(self.s + 0.5 * SIM_DT * k1, Vr, Vl)
            k3 = _deriv(self.s + 0.5 * SIM_DT * k2, Vr, Vl)
            k4 = _deriv(self.s + SIM_DT * k3, Vr, Vl)
            self.s = self.s + (SIM_DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            self.t += SIM_DT
            perp_acc += abs(self.s[0] * self.tgt[1] - self.s[1] * self.tgt[0]) \
                / max(self.d_gs, 1e-9) * SIM_DT
        d = float(np.hypot(self.tgt[0] - self.s[0], self.tgt[1] - self.s[1]))
        # running cost: line adherence (time-integral mean approximated by
        # integral/expected-duration) + time pressure
        reward = -SCALE * (1.5 * perp_acc / max(K_T * self.d_gs, 1e-6)
                           + LAMBDA_T * CTRL_DT)
        terminated = d < ARRIVAL_TOL
        truncated = self.t >= self.t_max
        if terminated or truncated:
            t_norm = abs(self.t - K_T * self.d_gs)
            if terminated:
                dth = abs((self.s[2] - self.phi + np.pi) % (2 * np.pi) - np.pi)
            else:
                dth = np.pi
            reward += -SCALE * (2.0 * dth / np.pi + 2.0 * np.tanh(0.1 * t_norm))
        return self._obs(), float(reward), bool(terminated), bool(truncated), {}
