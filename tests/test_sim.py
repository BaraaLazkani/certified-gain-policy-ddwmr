"""Sanity + integrator cross-validation for the closed-loop simulator."""
import sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from scipy.integrate import solve_ivp

from ddwmr.sim import (run_closedloop, LyapGains, PIDGains, PerturbedPlant,
                       _deriv, Disturbances)
from ddwmr.metrics import anchor_metric
from ddwmr.params import LYAP1, LYAP2

TGT = np.array([[0.6, 0.4]])

def single(ctrl, gains, T=25.0):
    return run_closedloop(TGT, ctrl, gains, T_max=T)

def test_lyap_arrives():
    for (kp, kth), name in [(LYAP1, "Lyap1"), (LYAP2, "Lyap2")]:
        r = single("lyap", LyapGains(Kp=np.array([kp]), Kth=np.array([kth])))
        assert r.arrived[0] and not r.diverged[0], name
        print(f"{name}: t_end={r.t_end[0]:.3f}s  e_l={r.e_l_mean[0]*1000:.2f}mm "
              f"th_end={np.degrees(r.theta_end[0]):.2f}deg  peakV={r.peak_V[0]:.2f}V "
              f"E={r.energy[0]:.3f}J supdv={r.sup_dv[0]:.4f}")

def test_pid_arrives():
    r = single("pid", PIDGains())
    assert r.arrived[0] and not r.diverged[0]
    print(f"PID  : t_end={r.t_end[0]:.3f}s  e_l={r.e_l_mean[0]*1000:.2f}mm "
          f"th_end={np.degrees(r.theta_end[0]):.2f}deg  peakV={r.peak_V[0]:.2f}V")

def test_rk45_crosscheck():
    """Fixed-step RK4 dt=1ms vs scipy RK45 (rtol=1e-8) on the same closed loop."""
    g = LyapGains(Kp=np.array([0.8]), Kth=np.array([1.7]))
    p = PerturbedPlant.nominal()
    z = np.zeros(3); o = np.ones(1)
    def f(t, s):
        dS, _ = _deriv(s.reshape(1, -1), TGT, "lyap", g, p,
                       np.zeros((1, 3)), o, o, np.zeros(1))
        return dS.ravel()
    sol = solve_ivp(f, (0, 6.0), np.zeros(15), method="RK45",
                    rtol=1e-8, atol=1e-10, dense_output=True)
    # fixed-step result at t=6
    r = run_closedloop(TGT, "lyap", g, T_max=6.0)
    # re-run keeping state: use a manual integration to t=6 via run with huge tol
    S_ref = sol.y[:, -1]
    # manual RK4
    S = np.zeros((1, 15)); h = 1e-3
    for _ in range(6000):
        k1, _ = _deriv(S, TGT, "lyap", g, p, np.zeros((1, 3)), o, o, z[:1])
        k2, _ = _deriv(S + 0.5*h*k1, TGT, "lyap", g, p, np.zeros((1, 3)), o, o, z[:1])
        k3, _ = _deriv(S + 0.5*h*k2, TGT, "lyap", g, p, np.zeros((1, 3)), o, o, z[:1])
        k4, _ = _deriv(S + h*k3, TGT, "lyap", g, p, np.zeros((1, 3)), o, o, z[:1])
        S = S + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    err = np.abs(S.ravel()[:9] - S_ref[:9]).max()
    print(f"RK4(1ms) vs RK45(rtol1e-8) max state diff @t=6s: {err:.2e}")
    assert err < 1e-5

def test_speed():
    B = 2500
    rng = np.random.default_rng(0)
    g = LyapGains(Kp=rng.uniform(0.1, 1.0, B), Kth=rng.uniform(0.5, 3.0, B))
    tg = np.tile([[0.6, 0.4]], (B, 1))
    t0 = time.perf_counter()
    r = run_closedloop(tg, "lyap", g, T_max=30.0)
    el = time.perf_counter() - t0
    print(f"batch {B} x 30s: {el:.1f}s wall  arrived={r.arrived.mean()*100:.1f}% "
          f"diverged={r.diverged.sum()}")

if __name__ == "__main__":
    test_lyap_arrives()
    test_pid_arrives()
    test_rk45_crosscheck()
    test_speed()
    print("ALL OK")
