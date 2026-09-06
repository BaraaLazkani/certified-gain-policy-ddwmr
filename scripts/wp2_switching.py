"""WP2: switching-stability empirical campaign (tests H2 + theory P1-P5 data).

1000 random waypoint sequences (5-20 waypoints, spacing U(0.1,1) m, turn
angle U(-0.7pi, 0.7pi)) x three switching policies:
  (a) halt1cm : arrival tolerance 1 cm, switch on arrival (anchor semantics)
  (b) tol5cm  : switch at 5 cm (violates the dwell/arrival-set assumption)
  (c) timer   : forced switch every T_sw = 2.0 s regardless of distance
plus a fixed-gain control condition (Lyap1, halt1cm).

Gains per segment from the WP0 PNN queried in the robot BODY frame at each
switch. Theory-verification records:
  - per-segment: L_k, gains, measured arrival time, eps, bound
        T_k = ln(4)/Kth + (2/Kp) ln((L_k+eps)/eps)
  - full V(t), delta_v(t), delta_w(t) series for runs 0-49 of each policy
  - sup/RMS delta_v, delta_w for ALL runs
"""
import sys, csv, json, time, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from ddwmr.sim import run_closedloop, LyapGains
from ddwmr.pnn import GainMLP, predict_gains
from ddwmr.params import LYAP1, ARRIVAL_TOL

SEED = 31415
N_RUNS = 1000
T_SW = 2.0
RES = ROOT / "passport" / "results"
LOG = ROOT / "passport" / "logs"

rng = np.random.default_rng(SEED)


def gen_sequences(n_runs):
    n_wp = rng.integers(5, 21, n_runs)
    Kmax = int(n_wp.max())
    wps = np.zeros((n_runs, Kmax, 2))
    for i in range(n_runs):
        pos = np.zeros(2); head = 0.0
        for k in range(n_wp[i]):
            head = head + rng.uniform(-0.7 * np.pi, 0.7 * np.pi) if k else \
                rng.uniform(-0.7 * np.pi, 0.7 * np.pi)
            L = rng.uniform(0.1, 1.0)
            pos = pos + L * np.array([np.cos(head), np.sin(head)])
            wps[i, k] = pos
        wps[i, n_wp[i]:] = pos
    return wps, n_wp


net = GainMLP(head="linear")
net.load_state_dict(torch.load(ROOT / "passport" / "models" / "pnn_wp0.pt",
                               weights_only=True))
net.eval()


def pnn_policy(poses, tgts):
    """Body-frame relative target -> gains (clipped to corrected box)."""
    dx = tgts[:, 0] - poses[:, 0]
    dy = tgts[:, 1] - poses[:, 1]
    c, s = np.cos(-poses[:, 2]), np.sin(-poses[:, 2])
    rel = np.stack([c * dx - s * dy, s * dx + c * dy], axis=1)
    return predict_gains(net, rel)


def t_bound(Lk, Kp, Kth, eps):
    return np.log(4.0) / Kth + (2.0 / Kp) * np.log((Lk + eps) / eps)


def main():
    wps, n_wp = gen_sequences(N_RUNS)
    run_rows, seg_rows = [], []
    series_store = {}
    budgets = {}
    conditions = [
        ("halt1cm_pnn", "halt1cm", pnn_policy, None, 150.0),
        ("tol5cm_pnn", "tol5cm", pnn_policy, None, 150.0),
        ("timer_pnn", "timer", pnn_policy, None, 90.0),
        ("halt1cm_fixedL1", "halt1cm", None,
         LyapGains(Kp=np.full(N_RUNS, LYAP1[0]), Kth=np.full(N_RUNS, LYAP1[1])), 150.0),
    ]
    for name, pol, gp, fixed_gains, tmax in conditions:
        t0 = time.perf_counter()
        g = fixed_gains if fixed_gains is not None else \
            LyapGains(Kp=np.full(N_RUNS, 0.5), Kth=np.full(N_RUNS, 1.5))
        r = run_closedloop(None, "lyap", g, waypoints=wps, n_waypoints=n_wp,
                           gain_policy=gp, switch_policy=pol, switch_timer=T_SW,
                           T_max=tmax, record_series=50, series_stride=10)
        wall = time.perf_counter() - t0
        budgets[name] = dict(wall_s=wall, n_runs=N_RUNS)
        eps = ARRIVAL_TOL if pol != "tol5cm" else 0.05

        segs_by_row = {}
        for s_ in r.seg_records:
            segs_by_row.setdefault(s_["row"], []).append(s_)
        for i in range(N_RUNS):
            segs = sorted(segs_by_row.get(i, []), key=lambda s_: s_["seg"])
            dwells = [s_["t_arrive"] for s_ in segs]
            run_rows.append([name, i, int(n_wp[i]), bool(r.arrived[i]),
                             bool(r.diverged[i]), float(r.t_end[i]),
                             len(segs),
                             float(min(dwells)) if dwells else np.nan,
                             float(r.sup_dv[i]), float(r.rms_dv[i]),
                             float(r.sup_dw[i]), float(r.rms_dw[i]),
                             float(r.e_l_rmse[i]), float(r.peak_V[i])])
            for s_ in segs:
                Tb = t_bound(s_["Lk"], s_["Kp"], s_["Kth"], eps) \
                    if np.isfinite(s_["Kp"]) else np.nan
                seg_rows.append([name, i, s_["seg"], f"{s_['Lk']:.6f}",
                                 f"{s_['Kp']:.6f}", f"{s_['Kth']:.6f}",
                                 f"{s_['t_arrive']:.4f}", eps,
                                 f"{Tb:.4f}" if np.isfinite(Tb) else "",
                                 int(s_["t_arrive"] <= Tb) if np.isfinite(Tb) else "",
                                 f"{s_['d_at_switch']:.6f}",
                                 f"{s_['V_before']:.8f}",
                                 f"{s_['V_after']:.8f}" if np.isfinite(s_["V_after"]) else "",
                                 f"{s_['t_switch']:.4f}"])
        series_store[name] = r.series
        print(f"{name}: wall={wall:.0f}s completed={r.arrived.mean()*100:.1f}% "
              f"diverged={int(r.diverged.sum())}", flush=True)

    with open(RES / "wp2_switching.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "run", "n_waypoints", "completed", "diverged",
                    "t_end_s", "n_segments_done", "min_dwell_s",
                    "sup_dv", "rms_dv", "sup_dw", "rms_dw",
                    "cross_track_rmse_m", "peak_V"])
        w.writerows(run_rows)
    with open(RES / "wp2_segments.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "run", "seg", "L_k_m", "Kp", "Kth",
                    "t_arrive_s", "eps_m", "T_bound_s", "within_bound",
                    "d_at_switch_m", "V_before", "V_after", "t_switch_s"])
        w.writerows(seg_rows)
    np.savez_compressed(
        RES / "wp2_vseries.npz",
        **{f"{name}_{key}": series_store[name][key]
           for name in series_store for key in ("t", "V_lyap", "dv", "dw", "d", "eth")},
        note=np.array(["runs 0-49 of each condition, 100 Hz, NaN after halt"]))
    with open(LOG / "wp2_budgets.json", "w") as f:
        json.dump(dict(seed=SEED, n_runs=N_RUNS, T_sw=T_SW, budgets=budgets), f, indent=1)
    print("WP2 DONE")


if __name__ == "__main__":
    main()
