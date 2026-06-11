"""WP0 pre-check: verify transcribed dynamics by reproducing the anchor's
published LQR gains (Eqs. 48-49) from their Q,R (Eqs. 46-47), and checking
Eqs. 12-15 coefficients against a first-principles M^-1 derivation."""
import numpy as np
from scipy.linalg import solve_continuous_are

# Table 1
m, Ig, d, beta, Iw, r, Km, La, Ra = 1.2, 0.003, 0.215, 0.0013, 0.0009, 0.035, 0.3012, 0.003, 4.5

# First-principles: M [wr_dot; wl_dot] = [Km*ir - beta*wr; Km*il - beta*wl]
p = 0.25 * m * r**2 + (r**2 / d**2) * Ig + Iw
q = 0.25 * m * r**2 - (r**2 / d**2) * Ig
det = p**2 - q**2
a11_fp = -beta * p / det
a12_fp = beta * q / det
a13_fp = Km * p / det
a14_fp = -Km * q / det

# Paper Eqs. 12-15 verbatim
den = (2 * Iw * d**2 + 4 * Ig * r**2) * (m * r**2 + 2 * Iw)
a11 = -beta * (m * d**2 * r**2 + 4 * Iw * d**2 + 4 * Ig * r**2) / den
a12 = -beta * r**2 * (-m * d**2 + 4 * Ig) / den
a13 = Km * (m * d**2 * r**2 + 4 * Iw * d**2 + 4 * Ig * r**2) / den
a14 = Km * r**2 * (-m * d**2 + 4 * Ig) / den

print("Eq12-15 vs first-principles M^-1:")
for name, a, b in [("a11", a11, a11_fp), ("a12", a12, a12_fp), ("a13", a13, a13_fp), ("a14", a14, a14_fp)]:
    print(f"  {name}: paper={a:.6f}  derived={b:.6f}  match={np.isclose(a, b)}")

# 4-state motor-robot model, Eq. 10
A4 = np.array([
    [a11, a12, a13, a14],
    [a12, a11, a14, a13],
    [-Km / La, 0, -Ra / La, 0],
    [0, -Km / La, 0, -Ra / La],
])
B4 = np.array([[0, 0], [0, 0], [1 / La, 0], [0, 1 / La]])

# Augmented 6-state, Eq. 42: x_aug = [wr, wl, ir, il, eps_r, eps_l]
A6 = np.zeros((6, 6)); A6[:4, :4] = A4; A6[4, 0] = -1.0; A6[5, 1] = -1.0
B6 = np.zeros((6, 2)); B6[:4, :] = B4

Q = np.diag([0.1, 0.1, 1.0, 1.0, 1000.0, 1000.0])
R = np.diag([1000.0, 1000.0])
P = solve_continuous_are(A6, B6, Q, R)
K = np.linalg.solve(R, B6.T @ P)  # u = -K x_aug
print("\nLQR K (state block, expect Eq.49 K2):")
print(np.round(K[:, :4], 4))
print("LQR K (integral block, expect Eq.48 K1 = -I):")
print(np.round(K[:, 4:], 4))

K2_paper = np.array([[0.0577, 0.0114, 0.0087, -0.0001], [0.0114, 0.0577, -0.0001, 0.0087]])
print("\nmax |K2 - paper| =", np.abs(K[:, :4] - K2_paper).max())

# Observer check: A4 - Ke C stable, Ke from Eq. 52
Ke = np.array([[0.0389, 0.0051], [0.0051, 0.0389], [-0.0021, -0.0005], [-0.0005, -0.0021]])
C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
print("\nplant eig(A4):", np.sort(np.linalg.eigvals(A4).real))
print("observer eig(A4 - Ke C):", np.sort(np.linalg.eigvals(A4 - Ke @ C).real))
print("closed inner-loop eig(A6 - B6 K):", np.sort(np.linalg.eigvals(A6 - B6 @ K).real))
