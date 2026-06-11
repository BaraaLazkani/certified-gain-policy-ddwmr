"""WP1: gain-policy training and H1 sample-efficiency comparison.

Methods (all on the SAME 200 workspace targets, corrected gain box, exact
anchor metric with per-target C_l from the brute-force run):
  bf   : brute-force best-so-far curves (random order, from saved surfaces)
  bptt : MLP policy pi(x,y)->(Kp,Kth) trained end-to-end through the
         differentiable rollout (sigmoid head => gains strictly in box =>
         Eq. 35 frozen-gain stability by construction)
  bo   : per-target GP-EI Bayesian optimization (5 init + 20 EI iters),
         batched across targets; labels distilled into the same MLP

x-axis unit: trajectory rollouts PER TARGET (BF: 2500; BO: 25; BPTT: 1/iter).
Checkpoint evaluations use the exact non-smooth sim and are NOT counted as
training rollouts (logged separately).

Outputs: results/wp1_sample_efficiency.csv, results/wp1_final.csv,
         models/policy_bptt_seed*.pt, models/pnn_bo_distilled_seed*.pt,
         logs/wp1_*.csv
"""
import sys, time, csv, pathlib, json, multiprocessing as mp
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RES = ROOT / "passport" / "results"
MOD = ROOT / "passport" / "models"
LOG = ROOT / "passport" / "logs"

SEEDS = list(range(10))
BPTT_ITERS = 150
BPTT_CKPT = [1, 2, 5, 10, 25, 50, 75, 100, 125, 150]
BO_INIT, BO_ITERS = 5, 20


def load_bf():
    bf = np.load(RES / "wp0_bruteforce.npz")
    n = int(bf["n_workspace"])
    return (bf["targets"][:n], bf["C_l_bf"][:n], bf["E"][:n],
            bf["gains"], bf["labels"][:n])


def eval_exact(gains_arr, targets, C_l):
    """Exact-sim metric for per-target gains. Returns E (nT,)."""
    from ddwmr.sim import run_closedloop, LyapGains
    from ddwmr.metrics import anchor_metric
    r = run_closedloop(targets, "lyap",
                       LyapGains(Kp=gains_arr[:, 0].copy(), Kth=gains_arr[:, 1].copy()),
                       T_max=40.0)
    E = np.zeros(len(targets))
    for i in range(len(targets)):
        m = anchor_metric(r.e_l_mean[i:i+1], r.theta_end[i:i+1], r.t_end[i:i+1],
                          np.zeros((1, 2)), targets[i:i+1], C_l=C_l[i])
        E[i] = m["e_total"][0]
    return E


def summarize(E, Estar):
    ratio = E / Estar
    return (float(np.median(ratio)), float(np.percentile(ratio, 25)),
            float(np.percentile(ratio, 75)), float((ratio <= 1.02).mean()))


# ------------------------------------------------------------------ BF curves
def bf_curves(seed, E_surf, Estar):
    rng = np.random.default_rng(1000 + seed)
    nT, nG = E_surf.shape
    order = np.stack([rng.permutation(nG) for _ in range(nT)])
    rows = []
    best = np.full(nT, np.inf)
    ckpts = sorted(set(np.unique(np.round(np.geomspace(1, nG, 40)).astype(int))))
    nxt = 0
    for j in range(nG):
        best = np.minimum(best, E_surf[np.arange(nT), order[:, j]])
        if nxt < len(ckpts) and j + 1 == ckpts[nxt]:
            med, p25, p75, frac = summarize(best, Estar)
            rows.append(["bf", seed, j + 1, med, p25, p75, frac])
            nxt += 1
    return rows


# ------------------------------------------------------------------ BPTT
def bptt_worker(seed):
    import torch
    torch.set_num_threads(4)
    from ddwmr.diffsim import rollout_metric, discretize
    from ddwmr.pnn import GainMLP
    targets, C_l, E_surf, _, _ = load_bf()
    Estar = E_surf.min(axis=1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = GainMLP(head="sigmoid")
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    X = torch.tensor(targets, dtype=torch.float32)
    Cl_t = torch.tensor(C_l, dtype=torch.float32)
    AdBd = discretize()
    rows = []
    t0 = time.perf_counter()
    n_eval_rollouts = 0
    for it in range(1, BPTT_ITERS + 1):
        opt.zero_grad()
        g = net(X)
        E = rollout_metric(g, X, Cl_t, T=14.0, AdBd=AdBd)
        E.mean().backward()
        opt.step()
        if it in BPTT_CKPT:
            import torch as _t
            with _t.no_grad():
                gg = net(X).numpy().astype(float)
            Eex = eval_exact(gg, targets, C_l)
            n_eval_rollouts += len(targets)
            med, p25, p75, frac = summarize(Eex, Estar)
            rows.append(["bptt", seed, it, med, p25, p75, frac])
    wall = time.perf_counter() - t0
    torch.save(net.state_dict(), MOD / f"policy_bptt_seed{seed}.pt")
    return rows, dict(method="bptt", seed=seed, wall_s=wall,
                      train_rollouts_per_target=BPTT_ITERS,
                      eval_rollouts_total=n_eval_rollouts)


# ------------------------------------------------------------------ BO
def _ei(mu, sd, best):
    from scipy.stats import norm
    z = (best - mu) / np.maximum(sd, 1e-9)
    return (best - mu) * norm.cdf(z) + sd * norm.pdf(z)


def bo_worker(seed):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
    from ddwmr.params import KP_MIN, KP_MAX, KTH_MIN, KTH_MAX
    targets, C_l, E_surf, _, _ = load_bf()
    Estar = E_surf.min(axis=1)
    nT = len(targets)
    rng = np.random.default_rng(2000 + seed)
    lo = np.array([KP_MIN, KTH_MIN]); hi = np.array([KP_MAX, KTH_MAX])
    Xobs = [[] for _ in range(nT)]; Yobs = [[] for _ in range(nT)]
    rows = []
    t0 = time.perf_counter()
    for it in range(BO_INIT + BO_ITERS):
        cand = np.zeros((nT, 2))
        if it < BO_INIT:
            cand = rng.uniform(lo, hi, (nT, 2))
        else:
            pool = rng.uniform(lo, hi, (800, 2))
            poolN = (pool - lo) / (hi - lo)
            for i in range(nT):
                Xn = (np.array(Xobs[i]) - lo) / (hi - lo)
                yv = np.array(Yobs[i])
                ym, ys = yv.mean(), max(yv.std(), 1e-6)
                gp = GaussianProcessRegressor(
                    kernel=ConstantKernel(1.0, (0.05, 20.0))
                    * Matern(length_scale=[0.3, 0.3],
                             length_scale_bounds=(0.05, 2.0), nu=2.5)
                    + WhiteKernel(1e-4, noise_level_bounds=(1e-8, 1e-1)),
                    normalize_y=False, n_restarts_optimizer=0, alpha=1e-8)
                gp.fit(Xn, (yv - ym) / ys)
                mu, sd = gp.predict(poolN, return_std=True)
                ei = _ei(mu * ys + ym, sd * ys, yv.min())
                cand[i] = pool[ei.argmax()]
        E = eval_exact(cand, targets, C_l)
        for i in range(nT):
            Xobs[i].append(cand[i]); Yobs[i].append(E[i])
        best = np.array([min(y) for y in Yobs])
        med, p25, p75, frac = summarize(best, Estar)
        rows.append(["bo", seed, it + 1, med, p25, p75, frac])
    wall = time.perf_counter() - t0

    # distill labels -> MLP (anchor recipe)
    import torch
    torch.set_num_threads(4)
    from ddwmr.pnn import train_supervised, predict_gains
    labels = np.array([Xobs[i][int(np.argmin(Yobs[i]))] for i in range(nT)])
    net, _ = train_supervised(targets, labels, seed=seed)
    torch.save(net.state_dict(), MOD / f"pnn_bo_distilled_seed{seed}.pt")
    gg = predict_gains(net, targets)
    Edist = eval_exact(gg, targets, C_l)
    medd, p25d, p75d, fracd = summarize(Edist, Estar)
    rows.append(["bo_distilled", seed, BO_INIT + BO_ITERS, medd, p25d, p75d, fracd])
    return rows, dict(method="bo", seed=seed, wall_s=wall,
                      train_rollouts_per_target=BO_INIT + BO_ITERS,
                      eval_rollouts_total=0)


def main():
    targets, C_l, E_surf, _, _ = load_bf()
    Estar = E_surf.min(axis=1)
    all_rows = []
    meta = []

    # BF curves (cheap, from saved surfaces)
    for s in SEEDS:
        all_rows += bf_curves(s, E_surf, Estar)
    meta.append(dict(method="bf", seed=-1, wall_s=907.1,
                     train_rollouts_per_target=E_surf.shape[1],
                     eval_rollouts_total=0,
                     note="wall time from wp0_bruteforce_log (515000 rollouts)"))

    with mp.Pool(5) as pool:
        for rows, m in pool.imap_unordered(bptt_worker, SEEDS):
            all_rows += rows; meta.append(m)
            print("bptt seed done:", m)
    with mp.Pool(5) as pool:
        for rows, m in pool.imap_unordered(bo_worker, SEEDS):
            all_rows += rows; meta.append(m)
            print("bo seed done:", m)

    with open(RES / "wp1_sample_efficiency.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "seed", "rollouts_per_target", "median_E_ratio",
                    "p25_E_ratio", "p75_E_ratio", "frac_within_2pct"])
        w.writerows(all_rows)
    with open(LOG / "wp1_budgets.json", "w") as f:
        json.dump(meta, f, indent=1)

    # H1 summary
    final = {}
    for m in ["bptt", "bo", "bo_distilled"]:
        rows = [r for r in all_rows if r[0] == m]
        last_per_seed = {}
        for r in rows:
            last_per_seed[r[1]] = r
        med = np.median([r[3] for r in last_per_seed.values()])
        frac = np.median([r[6] for r in last_per_seed.values()])
        final[m] = dict(median_E_ratio=float(med), frac_within_2pct=float(frac))
    with open(RES / "wp1_final.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "median_E_ratio_final", "median_frac_within_2pct",
                    "rollouts_per_target", "speedup_vs_bf"])
        w.writerow(["bf", 1.0, 1.0, E_surf.shape[1], 1.0])
        w.writerow(["bptt", final["bptt"]["median_E_ratio"],
                    final["bptt"]["frac_within_2pct"], BPTT_ITERS,
                    E_surf.shape[1] / BPTT_ITERS])
        w.writerow(["bo", final["bo"]["median_E_ratio"],
                    final["bo"]["frac_within_2pct"], BO_INIT + BO_ITERS,
                    E_surf.shape[1] / (BO_INIT + BO_ITERS)])
        w.writerow(["bo_distilled", final["bo_distilled"]["median_E_ratio"],
                    final["bo_distilled"]["frac_within_2pct"], BO_INIT + BO_ITERS,
                    E_surf.shape[1] / (BO_INIT + BO_ITERS)])
    print(json.dumps(final, indent=1))


if __name__ == "__main__":
    main()
