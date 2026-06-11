"""Anchor-paper plant parameters and derived matrices.

All equation/table numbers refer to Deeb, Alsaleh & Hatem (in press, IJCAS),
"AI-Driven Control Strategy for DDWMR".
Verified: LQR(Q,R of Eqs. 46-47) on the Eq. 42 augmented system reproduces the
published gains Eqs. 48-49 to 4 decimals; Eqs. 12-15 match a first-principles
M^-1 derivation (scripts/check_lqr.py).
"""
from dataclasses import dataclass, field, replace
import numpy as np
from scipy.linalg import solve_continuous_are


@dataclass(frozen=True)
class PlantParams:
    """Table 1 physical parameters (perturbable for WP3)."""
    m: float = 1.2        # kg
    Ig: float = 0.003     # kg m^2
    d: float = 0.215      # m, distance between wheels
    beta: float = 0.0013  # N m s
    Iw: float = 0.0009    # kg m^2
    r: float = 0.035      # m
    Km: float = 0.3012    # N m / A
    La: float = 0.003     # H
    Ra: float = 4.5       # Ohm


NOMINAL = PlantParams()

# ---- gain box (SPEC_AMENDMENTS #1: Kp_min = 0.1, NOT 0) ----
KP_MIN, KP_MAX = 0.1, 1.0
KTH_MIN, KTH_MAX = 0.5, 3.0
# anchor's original box, used ONLY for the Fig-8 surface shape comparison
KP_MIN_ANCHOR_FIG8 = 0.0

# ---- task constants ----
DT = 1e-3                # s, fixed-step RK4 (blueprint §1)
ARRIVAL_TOL = 0.01       # m (anchor §7.2)
K_T = 3.5                # s/m desired traversal time gain (inferred: 6 m linear
                         # path x 3.5 = 21.0 s matches Figs. 29-32 exactly)
V_MAX = 12.0             # V saturation (anchor Fig. 13 shows sat block, limit
                         # unstated; 12 V assumed, peaks in Fig. 6 are ~5.3 V)
METRIC_WEIGHTS = (1.5, 2.0, 2.0)   # Eq. 36
METRIC_SCALE = 1000.0
TANH_TIMESCALE = 0.1

# Table 3 baseline gains
LYAP1 = (0.8, 1.7)
LYAP2 = (0.5, 2.5)
PID_TABLE3 = dict(KP1=0.5, KI1=0.01, KD1=0.001, KP2=1.0, KI2=0.2, KD2=0.0)


def motor_coeffs(p: PlantParams):
    """a11..a14 of Eqs. 12-15 via first-principles M^-1 (verified identical)."""
    pp = 0.25 * p.m * p.r**2 + (p.r**2 / p.d**2) * p.Ig + p.Iw
    qq = 0.25 * p.m * p.r**2 - (p.r**2 / p.d**2) * p.Ig
    det = pp**2 - qq**2
    a11 = -p.beta * pp / det
    a12 = p.beta * qq / det
    a13 = p.Km * pp / det
    a14 = -p.Km * qq / det
    return a11, a12, a13, a14


def build_matrices(p: PlantParams):
    a11, a12, a13, a14 = motor_coeffs(p)
    A4 = np.array([
        [a11, a12, a13, a14],
        [a12, a11, a14, a13],
        [-p.Km / p.La, 0.0, -p.Ra / p.La, 0.0],
        [0.0, -p.Km / p.La, 0.0, -p.Ra / p.La],
    ])
    B4 = np.array([[0.0, 0.0], [0.0, 0.0], [1.0 / p.La, 0.0], [0.0, 1.0 / p.La]])
    return A4, B4


def lqr_gains():
    """K (2x6) for u = -K [x4; eps] on the NOMINAL augmented system (Eq. 42)."""
    A4, B4 = build_matrices(NOMINAL)
    A6 = np.zeros((6, 6)); A6[:4, :4] = A4; A6[4, 0] = -1.0; A6[5, 1] = -1.0
    B6 = np.zeros((6, 2)); B6[:4, :] = B4
    Q = np.diag([0.1, 0.1, 1.0, 1.0, 1000.0, 1000.0])
    R = np.diag([1000.0, 1000.0])
    P = solve_continuous_are(A6, B6, Q, R)
    return np.linalg.solve(R, B6.T @ P)


# Controller-side constants (always NOMINAL — model mismatch is the WP3 point)
A4_NOM, B4_NOM = build_matrices(NOMINAL)
K_LQR = lqr_gains()                       # (2,6); integral block is -I
K2 = K_LQR[:, :4]                         # state-feedback block (Eq. 49)
KE = np.array([[0.0389, 0.0051],          # observer gain, Eq. 52 verbatim
               [0.0051, 0.0389],
               [-0.0021, -0.0005],
               [-0.0005, -0.0021]])
C_OUT = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])

_K2_PAPER = np.array([[0.0577, 0.0114, 0.0087, -0.0001],
                      [0.0114, 0.0577, -0.0001, 0.0087]])
assert np.abs(K2 - _K2_PAPER).max() < 5e-4, "LQR gains diverge from Eq. 49"
assert np.abs(K_LQR[:, 4:] + np.eye(2)).max() < 1e-6, "integral block != -I (Eq. 48)"
