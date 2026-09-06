"""WP0/WP1 brute-force reference run.

Grid: 50x50 gains over the CORRECTED box Kp in [0.1,1.0], Kth in [0.5,3.0]
(Kp_min > 0) x 200 workspace targets (anchor Eqs. 40-41) plus the
4 validation targets (Tables 4-7) and the Fig-8 target (0.45,0.77).
For the Fig-8 target only, an ADDITIONAL surface over the anchor's original
box Kp in [0,1] is computed for the shape comparison (verification artifact).

Outputs: passport/results/wp0_bruteforce.npz  (full E surfaces + raw metrics)
         passport/results/wp1_bruteforce_labels.csv (argmin gains per target)
         passport/logs/wp0_bruteforce_log.txt (rollout count + wall time = H1 yardstick)
"""
import sys, time, os, pathlib, multiprocessing as mp
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ddwmr.sim import run_closedloop, LyapGains
from ddwmr.metrics import anchor_metric
from ddwmr.params import KP_MIN, KP_MAX, KTH_MIN, KTH_MAX, K_T

N_KP, N_KTH = 50, 50
T_MAX = 40.0
SEED = 20260612

KP_GRID = np.linspace(KP_MIN, KP_MAX, N_KP)
KTH_GRID = np.linspace(KTH_MIN, KTH_MAX, N_KTH)
KPM, KTHM = np.meshgrid(KP_GRID, KTH_GRID, indexing="ij")
GAINS = np.stack([KPM.ravel(), KTHM.ravel()], axis=1)          # (2500,2)

# anchor original box for Fig-8 shape check only
KP_GRID_FIG8 = np.linspace(0.0, 1.0, N_KP)
KPM8, KTHM8 = np.meshgrid(KP_GRID_FIG8, KTH_GRID, indexing="ij")
GAINS_FIG8 = np.stack([KPM8.ravel(), KTHM8.ravel()], axis=1)

def workspace_targets():
    radii = np.linspace(0.1, 1.0, 10)
    angs = np.linspace(-0.7 * np.pi, 0.7 * np.pi, 20)
    pts = [(d * np.cos(a), d * np.sin(a)) for d in radii for a in angs]
    return np.array(pts)

VAL_TARGETS = np.array([[-0.2, -0.5], [-0.3, 0.5], [0.5, -0.5], [0.6, 0.4]])
FIG8_TARGET = np.array([[0.45, 0.77]])


def run_target(args):
    idx, tgt, gains = args
    B = gains.shape[0]
    tg = np.tile(tgt, (B, 1))
    g = LyapGains(Kp=gains[:, 0].copy(), Kth=gains[:, 1].copy())
    t0 = time.perf_counter()
    r = run_closedloop(tg, "lyap", g, T_max=T_MAX)
    wall = time.perf_counter() - t0
    return idx, dict(e_l=r.e_l_mean, t_end=r.t_end, th_end=r.theta_end,
                     arrived=r.arrived, diverged=r.diverged, wall=wall)


def main():
    res_dir = ROOT / "passport" / "results"
    log_dir = ROOT / "passport" / "logs"
    res_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    ws = workspace_targets()
    all_targets = np.vstack([ws, VAL_TARGETS, FIG8_TARGET])   # 205 targets
    jobs = [(i, all_targets[i], GAINS) for i in range(len(all_targets))]
    jobs.append((len(all_targets), FIG8_TARGET[0], GAINS_FIG8))  # anchor-box surface

    t0 = time.perf_counter()
    nproc = min(8, os.cpu_count())
    with mp.Pool(nproc) as pool:
        results = dict(pool.imap_unordered(run_target, jobs, chunksize=1))
    wall_total = time.perf_counter() - t0

    nT = len(all_targets)
    B = GAINS.shape[0]
    e_l = np.zeros((nT, B)); t_end = np.zeros((nT, B)); th_end = np.zeros((nT, B))
    arrived = np.zeros((nT, B), bool); diverged = np.zeros((nT, B), bool)
    for i in range(nT):
        d = results[i]
        e_l[i], t_end[i], th_end[i] = d["e_l"], d["t_end"], d["th_end"]
        arrived[i], diverged[i] = d["arrived"], d["diverged"]
    fig8 = results[nT]

    # anchor metric per target with per-target brute-force C_l = max e_l (the
    # anchor's apparent e_total convention) AND with C_l = 1 m (component conv.)
    C_l_bf = e_l.max(axis=1)                                  # (nT,)
    E = np.zeros((nT, B)); E_cl1 = np.zeros((nT, B))
    for i in range(nT):
        st = np.zeros((B, 2)); tg = np.tile(all_targets[i], (B, 1))
        m = anchor_metric(e_l[i], th_end[i], t_end[i], st, tg, C_l=C_l_bf[i])
        E[i] = m["e_total"]
        m1 = anchor_metric(e_l[i], th_end[i], t_end[i], st, tg, C_l=1.0)
        E_cl1[i] = m1["e_total"]

    best = E.argmin(axis=1)
    labels = GAINS[best]                                       # (nT,2)

    np.savez_compressed(
        res_dir / "wp0_bruteforce.npz",
        targets=all_targets, gains=GAINS, kp_grid=KP_GRID, kth_grid=KTH_GRID,
        e_l=e_l, t_end=t_end, th_end=th_end, arrived=arrived, diverged=diverged,
        E=E, E_cl1=E_cl1, C_l_bf=C_l_bf, labels=labels,
        n_workspace=len(ws), val_targets=VAL_TARGETS, fig8_target=FIG8_TARGET,
        fig8_anchorbox_e_l=fig8["e_l"], fig8_anchorbox_t_end=fig8["t_end"],
        fig8_anchorbox_th_end=fig8["th_end"], fig8_gains=GAINS_FIG8,
        kp_grid_fig8=KP_GRID_FIG8)

    import csv
    with open(res_dir / "wp1_bruteforce_labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target_idx", "x_ref", "y_ref", "Kp_star", "Kth_star",
                    "E_star", "C_l_bf", "is_workspace"])
        for i in range(nT):
            w.writerow([i, f"{all_targets[i,0]:.6f}", f"{all_targets[i,1]:.6f}",
                        f"{labels[i,0]:.6f}", f"{labels[i,1]:.6f}",
                        f"{E[i, best[i]]:.4f}", f"{C_l_bf[i]:.6f}",
                        int(i < len(ws))])

    n_rollouts = B * (nT + 1)
    with open(log_dir / "wp0_bruteforce_log.txt", "w") as f:
        f.write(f"seed={SEED}\nn_targets={nT} (+1 fig8 anchor-box surface)\n"
                f"grid={N_KP}x{N_KTH} gain box Kp[{KP_MIN},{KP_MAX}] "
                f"Kth[{KTH_MIN},{KTH_MAX}] (corrected gain box)\n"
                f"total_rollouts={n_rollouts}\nwall_time_s={wall_total:.1f}\n"
                f"nproc={nproc}\nT_max={T_MAX}\ndt=0.001\nK_t={K_T}\n"
                f"divergences={int(diverged.sum())}\n"
                f"never_arrived={int((~arrived).sum())} of {nT*B}\n")
    print(f"DONE rollouts={n_rollouts} wall={wall_total:.0f}s "
          f"div={diverged.sum()} noarr={(~arrived).sum()}")


if __name__ == "__main__":
    main()
