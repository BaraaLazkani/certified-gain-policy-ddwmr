"""WP7-1 + WP7-2: BO label capture, honest cost timing, and the 2x2
head x label-source ablation.

Why this exists
---------------
An earlier version of this campaign labelled a result row "Ours (BO policy)" that is in
fact loaded from policy_bptt_seed*.pt (wp5_eval.py:78).  The BO policy is
never evaluated downstream.  Worse, the BO-distilled nets were trained with
head="linear" (wp1_train.py:175 -> pnn.train_supervised default), so their
output is unbounded and Assumption 1 does NOT hold for them: they cannot be
the certified policy without retraining with a sigmoid head.

The BO labels themselves were never saved, so they are regenerated here.

The ablation separates two explanations for the held-out win that the earlier write-up
currently attributes entirely to Bayesian optimization:

                     brute-force labels        BO labels
    linear head      anchor's PNN              current pnn_bo_distilled
    sigmoid head     box alone                 the true certified BO policy

If sigmoid+BF matches sigmoid+BO on held-out targets, the win is the output
box, not the trainer, and contribution 2 must be rewritten.

Cost timing (WP7-1) is recorded at a single stated parallelism with the core
count logged, because the earlier comparison came from runs at different
core counts (grid Pool(8) vs BO/BPTT Pool(5)) and is therefore not well posed.
"""
import sys, os, csv, json, time, pathlib, platform, multiprocessing as mp
import numpy as np

CODE = pathlib.Path(__file__).resolve().parents[1]   # repository root
sys.path.insert(0, str(CODE / "src"))
sys.path.insert(0, str(CODE / "scripts"))

OUT = CODE / "wp7"
RES, MOD, LOGS = OUT / "results", OUT / "models", OUT / "logs"
for d in (RES, MOD, LOGS): d.mkdir(parents=True, exist_ok=True)

import wp1_train as W1                                    # reuse verified helpers
# wp1_train resolves its paths relative to its own root; repoint them at the
# original campaign's passport, which we read but never write.
PASSPORT = CODE / "passport"
W1.RES, W1.MOD, W1.LOG = (PASSPORT / "results", PASSPORT / "models",
                          PASSPORT / "logs")
from wp1_train import eval_exact, summarize, _ei, BO_INIT, BO_ITERS
load_bf = W1.load_bf                                      # binds the patched W1.RES

SEEDS = list(range(int(os.environ.get("WP7_SEEDS", 10))))
NPROC = int(os.environ.get("WP7_NPROC", 5))               # ONE stated parallelism


# ---------------------------------------------------------------- WP7-1: BO
def bo_with_labels(seed):
    """Identical to wp1_train.bo_worker, but returns the labels it found."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
    from ddwmr.params import KP_MIN, KP_MAX, KTH_MIN, KTH_MAX
    targets, C_l, E_surf, _, _ = load_bf()
    nT = len(targets)
    rng = np.random.default_rng(2000 + seed)
    lo = np.array([KP_MIN, KTH_MIN]); hi = np.array([KP_MAX, KTH_MAX])
    Xobs = [[] for _ in range(nT)]; Yobs = [[] for _ in range(nT)]
    t0 = time.perf_counter()
    for it in range(BO_INIT + BO_ITERS):
        if it < BO_INIT:
            cand = rng.uniform(lo, hi, (nT, 2))
        else:
            cand = np.zeros((nT, 2))
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
                cand[i] = pool[_ei(mu * ys + ym, sd * ys, yv.min()).argmax()]
        E = eval_exact(cand, targets, C_l)
        for i in range(nT):
            Xobs[i].append(cand[i]); Yobs[i].append(E[i])
    wall = time.perf_counter() - t0
    labels = np.array([Xobs[i][int(np.argmin(Yobs[i]))] for i in range(nT)])
    np.save(RES / f"wp7_bo_labels_seed{seed}.npy", labels)
    return seed, wall


def wp7_1():
    print(f"[WP7-1] BO re-run, {len(SEEDS)} seeds at Pool({NPROC})", flush=True)
    t0 = time.perf_counter()
    with mp.Pool(NPROC) as pool:
        res = list(pool.imap_unordered(bo_with_labels, SEEDS))
    total = time.perf_counter() - t0
    per = {s: w for s, w in res}
    meta = dict(
        nproc=NPROC, seeds=len(SEEDS), cpu=platform.processor() or platform.machine(),
        cores_available=os.cpu_count(), python=platform.python_version(),
        bo_wall_s_per_seed=per,
        bo_wall_s_median=float(np.median(list(per.values()))),
        bo_batch_wall_s=total,
        bo_core_seconds=float(np.sum(list(per.values()))),
        note=("Wall time per seed measured inside the worker; core-seconds is the "
              "sum over seeds, i.e. the parallelism-independent cost. The earlier "
              "comparison used Pool(8) for the grid and Pool(5) here, so wall times "
              "were not comparable; core-seconds is."),
    )
    (LOGS / "wp7_cost_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[WP7-1] done: median {meta['bo_wall_s_median']:.0f}s/seed, "
          f"{meta['bo_core_seconds']:.0f} core-s total", flush=True)
    return meta


# ------------------------------------------------------- WP7-2: 2x2 ablation
def heldout_targets(n=30, seed=77):
    """Same construction as wp5_eval: outside the training workspace."""
    rng = np.random.default_rng(seed)
    rad = rng.uniform(1.0, 1.5, n)
    ang = rng.uniform(-0.9 * np.pi, 0.9 * np.pi, n)
    return np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)


def wp7_2():
    import torch
    from ddwmr.pnn import train_supervised, predict_gains
    from ddwmr.metrics import anchor_metric
    from ddwmr.sim import run_closedloop, LyapGains
    torch.set_num_threads(4)

    targets, C_l, E_surf, gains_grid, bf_labels = load_bf()
    Estar = E_surf.min(axis=1)
    ho = heldout_targets()
    rows = []

    label_sets = {"bf": bf_labels}
    bo_files = sorted(RES.glob("wp7_bo_labels_seed*.npy"))
    if not bo_files:
        raise SystemExit("WP7-2 requires WP7-1 labels; run WP7-1 first")

    for head in ("linear", "sigmoid"):
        for lsrc in ("bf", "bo"):
            for seed in SEEDS:
                lab = (bf_labels if lsrc == "bf"
                       else np.load(RES / f"wp7_bo_labels_seed{seed}.npy"))
                net, _ = train_supervised(targets, lab, seed=seed, head=head)
                torch.save(net.state_dict(), MOD / f"wp7_{head}_{lsrc}_seed{seed}.pt")

                # (a) training workspace
                g_in = predict_gains(net, targets)
                E_in = eval_exact(g_in, targets, C_l)
                med_ratio = float(np.median(E_in / Estar))

                # (b) held-out, C_l = 1 m so cells are comparable
                g_ho = predict_gains(net, ho)
                r = run_closedloop(ho, "lyap",
                                   LyapGains(Kp=g_ho[:, 0].copy(), Kth=g_ho[:, 1].copy()),
                                   T_max=40.0)
                m = anchor_metric(r.e_l_mean, r.theta_end, r.t_end,
                                  np.zeros((len(ho), 2)), ho, C_l=1.0)
                rows.append([head, lsrc, seed,
                             f"{np.median(E_in):.2f}", f"{med_ratio:.4f}",
                             f"{np.median(m['e_total']):.2f}",
                             f"{r.arrived.mean():.4f}", int(r.diverged.sum()),
                             f"{np.median(r.e_l_rmse):.5f}",
                             f"{g_ho[:,0].min():.3f}", f"{g_ho[:,0].max():.3f}",
                             f"{g_ho[:,1].min():.3f}", f"{g_ho[:,1].max():.3f}"])
                print(f"  {head:<7} {lsrc:<3} seed{seed}: "
                      f"in-dist ratio {med_ratio:.3f}  held-out E "
                      f"{np.median(m['e_total']):8.1f}  arrival "
                      f"{r.arrived.mean()*100:5.1f}%", flush=True)

    with open(RES / "wp7_ablation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["head", "labels", "seed", "E_indist_median",
                    "E_over_Egrid_median", "E_heldout_median_Cl1m",
                    "heldout_arrival", "heldout_diverged", "heldout_xt_rmse",
                    "Kp_min", "Kp_max", "Kth_min", "Kth_max"])
        w.writerows(rows)
    print(f"[WP7-2] wrote {len(rows)} rows -> wp7_ablation.csv", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "1"): wp7_1()
    if which in ("all", "2"): wp7_2()
    print("WP7-1/WP7-2 DONE")
