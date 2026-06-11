"""WP4: fair PID baseline — anchor's dual-PID structure tuned by GP-EI BO
with the SAME total rollout budget as our method (25 rollouts/target x 200
targets; the PID is context-free, so each BO evaluation spends one batched
200-target sim and the budget buys 25 BO iterations).

Output: results/wp4_pid_tuned.csv (best params + per-iteration trace),
        models/pid_tuned.json
"""
import sys, csv, json, time, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ddwmr.sim import run_closedloop, PIDGains
from ddwmr.metrics import anchor_metric

RES = ROOT / "passport" / "results"
MOD = ROOT / "passport" / "models"
SEED = 555
N_ITER, N_INIT = 25, 8
LO = np.array([0.1, 0.0, 0.0, 0.2, 0.0, 0.0])
HI = np.array([3.0, 0.5, 0.05, 6.0, 1.0, 0.05])
NAMES = ["KP1", "KI1", "KD1", "KP2", "KI2", "KD2"]


def objective(p, targets, C_l):
    g = PIDGains(**dict(zip(NAMES, p)))
    r = run_closedloop(targets, "pid", g, T_max=40.0)
    E = np.zeros(len(targets))
    for i in range(len(targets)):
        m = anchor_metric(r.e_l_mean[i:i+1], r.theta_end[i:i+1], r.t_end[i:i+1],
                          np.zeros((1, 2)), targets[i:i+1], C_l=C_l[i])
        E[i] = m["e_total"][0]
    bad = (~r.arrived) | r.diverged
    E[bad] = np.maximum(E[bad], 5000.0)
    return float(E.mean()), float(np.median(E)), float(bad.mean())


def main():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
    from scipy.stats import norm
    bf = np.load(RES / "wp0_bruteforce.npz")
    n = int(bf["n_workspace"])
    targets, C_l = bf["targets"][:n], bf["C_l_bf"][:n]
    rng = np.random.default_rng(SEED)
    Xo, Yo, rows = [], [], []
    t0 = time.perf_counter()
    for it in range(N_ITER):
        if it < N_INIT:
            p = rng.uniform(LO, HI)
            if it == 0:
                p = np.array([0.5, 0.01, 0.001, 1.0, 0.2, 0.0])  # Table 3 seed
        else:
            Xn = (np.array(Xo) - LO) / (HI - LO)
            yv = np.array(Yo); ym, ys = yv.mean(), max(yv.std(), 1e-9)
            gp = GaussianProcessRegressor(
                kernel=ConstantKernel(1.0, (0.05, 20.0))
                * Matern(length_scale=[0.3] * 6, length_scale_bounds=(0.05, 2.0), nu=2.5)
                + WhiteKernel(1e-4, noise_level_bounds=(1e-8, 1e-1)),
                n_restarts_optimizer=0, alpha=1e-8)
            gp.fit(Xn, (yv - ym) / ys)
            pool = rng.uniform(0, 1, (3000, 6))
            mu, sd = gp.predict(pool, return_std=True)
            mu = mu * ys + ym; sd = sd * ys
            best = yv.min()
            z = (best - mu) / np.maximum(sd, 1e-9)
            ei = (best - mu) * norm.cdf(z) + sd * norm.pdf(z)
            p = LO + pool[ei.argmax()] * (HI - LO)
        meanE, medE, failr = objective(p, targets, C_l)
        Xo.append(p); Yo.append(meanE)
        rows.append([it, *[f"{v:.5f}" for v in p], f"{meanE:.2f}",
                     f"{medE:.2f}", f"{failr:.3f}"])
        print(f"iter {it}: meanE={meanE:.1f} medE={medE:.1f} fail={failr:.2%}", flush=True)
    best_i = int(np.argmin(Yo))
    best = dict(zip(NAMES, map(float, Xo[best_i])))
    with open(MOD / "pid_tuned.json", "w") as f:
        json.dump(dict(params=best, mean_E=Yo[best_i], seed=SEED,
                       budget_rollouts=N_ITER * len(targets),
                       wall_s=time.perf_counter() - t0), f, indent=1)
    with open(RES / "wp4_pid_tuned.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", *NAMES, "mean_E", "median_E", "fail_rate"])
        w.writerows(rows)
    print("BEST PID:", best, "meanE", Yo[best_i])


if __name__ == "__main__":
    main()
