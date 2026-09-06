"""WP7-4 - loop transport delay sweep, redone (corrects defect D4).

Why this study was redone
---------------------------------------------
Three defects were found in the earlier delay study in this repository:

D4a  SCOPE.  The earlier sweep delayed *every* feedback signal reaching the
     controller - pose AND wheel speeds - through a single knob.  That is not a
     physical configuration: the wheel encoders are wired to the motor-controller
     board and their transport lag is microseconds, while only the pose estimate
     travels over the link that carries tens of milliseconds.  The collapse the
     paper reports therefore may be an artefact of delaying a path that is never
     delayed in hardware.  Here the three scopes - "pose", "wheels", "all" - are
     swept SEPARATELY and reported separately; they are never merged.

D4b  ORACLE TRIGGER.  The arrival test (sim.py, `exn = cur_tgt - Snew`) read the
     TRUE plant state.  The controller acted on a delayed pose while the
     supervisor that declared arrival - the rule the entire dwell-time guarantee
     rests on - read the plant directly.  `Disturbances.arrival_uses_measured`
     (added for this study, default False so every existing result is unchanged)
     makes the switching rule read the same delayed pose the controller reads.
     Both trigger modes are run, so the size of the oracle's contribution is
     measured rather than assumed.

D4c  LABEL.  The BPTT policy was reported as "Ours (BO policy)".  It is loaded
     from policy_bptt_seed9.pt and is labelled "ours_bptt" here.

Grid: tau in {0,10,20,50,100,125,150,175,200} ms  x  {pose,wheels,all}
      x  {oracle,measured}  x  6 controllers  x  N single-target runs.
Targets are drawn as in the earlier sweep (rad U(0.1,1.0),
ang U(-0.7pi,0.7pi), seed 60_2026) so the two studies are comparable.

Reads  ../../experiments/passport/models/  (read-only).
Writes ../wp7/results/  and  ../wp7/logs/ .
"""
import sys, os, csv, json, time, pathlib, platform, multiprocessing as mp

import numpy as np

CODE = pathlib.Path(__file__).resolve().parents[1]   # repository root
sys.path.insert(0, str(CODE / "src"))

from ddwmr.sim import (run_closedloop, LyapGains, PIDGains,   # noqa: E402
                       Disturbances)
from ddwmr.metrics import anchor_metric                       # noqa: E402
from ddwmr.params import LYAP1, LYAP2, K_T                    # noqa: E402

PASSPORT = CODE / "passport"
OUT = CODE / "wp7"
RES, LOGS = OUT / "results", OUT / "logs"
for d in (RES, LOGS):
    d.mkdir(parents=True, exist_ok=True)

SEED = 60_2026            # shared target draw
RNG_SIM = 777             # shared per-cell simulator stream
T_MAX = 30.0

N_MC = int(os.environ.get("WP7_N_MC", 300))
DELAYS_MS = [int(x) for x in os.environ.get(
    "WP7_DELAYS", "0,10,20,50,100,125,150,175,200").split(",")]
SCOPES = os.environ.get("WP7_SCOPES", "pose,wheels,all").split(",")
TRIGGERS = os.environ.get("WP7_TRIGGERS", "oracle,measured").split(",")
NPROC = int(os.environ.get("WP7_NPROC", 5))

# built in main(), inherited by fork()ed workers
TARGETS = None
CTRLS = None


def load_net(name, head):
    import torch
    from ddwmr.pnn import GainMLP
    net = GainMLP(head=head)
    net.load_state_dict(torch.load(PASSPORT / "models" / name, weights_only=True))
    net.eval()
    return net


def build_controllers(targets):
    """name -> (ctrl_kind, gains, Kp array or None, Kth array or None).

    Gains are a pure function of the target in the robot frame; the robot starts
    at the origin with theta=0, so the policy input is the target itself - the
    same query as the earlier sweep.  In single-target mode the policy is evaluated
    once, so the gains do not depend on tau, scope or trigger mode.
    """
    from ddwmr.pnn import predict_gains
    n = len(targets)
    out = {}
    for label, fname, head in [("ours_bptt", "policy_bptt_seed9.pt", "sigmoid"),
                               ("ours_dr",   "policy_dr_seed0.pt",   "sigmoid"),
                               ("pnn_bf",    "pnn_wp0.pt",           "linear")]:
        g = predict_gains(load_net(fname, head), targets)
        out[label] = ("lyap", LyapGains(Kp=g[:, 0].copy(), Kth=g[:, 1].copy()),
                      g[:, 0].copy(), g[:, 1].copy())
    for label, (kp, kth) in [("lyap1", LYAP1), ("lyap2", LYAP2)]:
        out[label] = ("lyap", LyapGains(Kp=np.full(n, kp), Kth=np.full(n, kth)),
                      np.full(n, kp), np.full(n, kth))
    out["pid_table3"] = ("pid", PIDGains(), None, None)
    return out


def run_cell(job):
    """One (tau, scope, trigger, controller) cell -> (summary, per-run rows)."""
    tau_ms, scope, trigger, cname = job
    ck, gains, kp, kth = CTRLS[cname]
    dist = Disturbances(loop_delay_s=tau_ms / 1000.0,
                        loop_delay_scope=scope,
                        arrival_uses_measured=(trigger == "measured"))
    t0 = time.perf_counter()
    r = run_closedloop(TARGETS, ck, gains, T_max=T_MAX, dist=dist,
                       rng=np.random.default_rng(RNG_SIM))
    wall = time.perf_counter() - t0
    m = anchor_metric(r.e_l_mean, r.theta_end, r.t_end,
                      np.zeros((N_MC, 2)), TARGETS, C_l=1.0)
    d_gs = np.linalg.norm(TARGETS, axis=1)
    fail = (~r.arrived) | r.diverged | (r.t_end > 3.0 * K_T * d_gs)
    rows = []
    for i in range(N_MC):
        rows.append([tau_ms, scope, trigger, cname, i,
                     f"{TARGETS[i,0]:.6f}", f"{TARGETS[i,1]:.6f}",
                     int(r.arrived[i]), int(r.diverged[i]), int(fail[i]),
                     f"{r.t_end[i]:.4f}", f"{m['e_total'][i]:.3f}",
                     f"{r.e_l_rmse[i]:.6f}", f"{r.peak_V[i]:.4f}",
                     f"{r.sup_dv[i]:.5f}", f"{r.sup_dw[i]:.5f}",
                     "" if kp is None else f"{kp[i]:.6f}",
                     "" if kth is None else f"{kth[i]:.6f}"])
    summ = [tau_ms, scope, trigger, cname, N_MC,
            f"{100.0 * r.arrived.mean():.2f}", int(r.diverged.sum()),
            f"{100.0 * fail.mean():.2f}",
            f"{np.median(r.t_end):.4f}", f"{np.median(m['e_total']):.3f}",
            f"{np.median(r.e_l_rmse):.6f}", f"{np.median(r.peak_V):.4f}",
            f"{np.median(r.sup_dv):.5f}", f"{np.median(r.sup_dw):.5f}",
            f"{wall:.1f}"]
    return summ, rows


ROW_HDR = ["delay_ms", "scope", "trigger_mode", "controller", "run",
           "x_ref", "y_ref", "arrived", "diverged", "failed_3x_rule",
           "t_end_s", "e_total_Cl1m", "cross_track_rmse_m", "peak_V",
           "sup_dv", "sup_dw", "Kp", "Kth"]
SUM_HDR = ["delay_ms", "scope", "trigger_mode", "controller", "n",
           "arrival_pct", "n_diverged", "failed_3x_pct",
           "median_t_end_s", "median_e_total_Cl1m", "median_cross_track_rmse_m",
           "median_peak_V", "median_sup_dv", "median_sup_dw", "wall_s"]


def main():
    global TARGETS, CTRLS
    rng = np.random.default_rng(SEED)
    ang = rng.uniform(-0.7 * np.pi, 0.7 * np.pi, N_MC)      # fixed draw order
    rad = rng.uniform(0.1, 1.0, N_MC)
    TARGETS = np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)
    CTRLS = build_controllers(TARGETS)

    # --- gain summary: which gain each controller actually selected ---------
    gain_rows = []
    for cname, (ck, _g, kp, kth) in CTRLS.items():
        if kp is None:
            gain_rows.append([cname, ck, N_MC, "", "", "", ""])
        else:
            gain_rows.append([cname, ck, N_MC,
                              f"{np.median(kp):.6f}", f"{kp.max():.6f}",
                              f"{np.median(kth):.6f}", f"{kth.max():.6f}"])
    with open(RES / "wp7_delay_gains.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["controller", "ctrl_kind", "n_targets",
                    "median_Kp", "max_Kp", "median_Kth", "max_Kth"])
        w.writerows(gain_rows)
    for g in gain_rows:
        print(f"GAIN {g[0]:<11} median Kp={g[3] or 'n/a':<9} max Kp={g[4] or 'n/a':<9} "
              f"median Kth={g[5] or 'n/a':<9} max Kth={g[6] or 'n/a'}", flush=True)

    jobs = [(tau, sc, tr, cn)
            for tau in DELAYS_MS for sc in SCOPES
            for tr in TRIGGERS for cn in CTRLS]
    print(f"\n{len(jobs)} cells x {N_MC} runs, {NPROC} workers, "
          f"load at start {os.getloadavg()}", flush=True)

    t0 = time.perf_counter()
    all_rows = []; summaries = []; budgets = {}
    with mp.Pool(NPROC) as pool:
        for k, (summ, rows) in enumerate(pool.imap_unordered(run_cell, jobs), 1):
            summaries.append(summ); all_rows.extend(rows)
            budgets[f"{summ[0]}ms/{summ[1]}/{summ[2]}/{summ[3]}"] = float(summ[-1])
            print(f"[{k:3d}/{len(jobs)}] tau={summ[0]:3d}ms {summ[1]:<6} "
                  f"{summ[2]:<8} {summ[3]:<11} arr={float(summ[5]):6.2f}% "
                  f"div={summ[6]:3d} medE={float(summ[9]):8.1f} "
                  f"med_t={float(summ[8]):6.2f}s ({summ[-1]}s)", flush=True)
    wall = time.perf_counter() - t0

    order = {c: i for i, c in enumerate(CTRLS)}
    summaries.sort(key=lambda s: (s[0], SCOPES.index(s[1]),
                                  TRIGGERS.index(s[2]), order[s[3]]))
    all_rows.sort(key=lambda r: (r[0], SCOPES.index(r[1]),
                                 TRIGGERS.index(r[2]), order[r[3]], r[4]))

    with open(RES / "wp7_delay_singletarget.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(ROW_HDR); w.writerows(all_rows)
    with open(RES / "wp7_delay_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(SUM_HDR); w.writerows(summaries)

    (LOGS / "wp7_delay_meta.json").write_text(json.dumps(dict(
        script="scripts/wp7_delay.py", seed=SEED, sim_rng=RNG_SIM, n_mc=N_MC,
        T_max=T_MAX, delays_ms=DELAYS_MS, scopes=SCOPES, triggers=TRIGGERS,
        controllers={k: v[0] for k, v in CTRLS.items()},
        checkpoints=dict(ours_bptt="policy_bptt_seed9.pt",
                         ours_dr="policy_dr_seed0.pt", pnn_bf="pnn_wp0.pt",
                         lyap1=list(LYAP1), lyap2=list(LYAP2),
                         pid_table3="ddwmr.params.PID_TABLE3"),
        n_cells=len(jobs), nproc=NPROC, total_wall_s=round(wall, 1),
        loadavg_end=os.getloadavg(), machine=platform.processor(),
        cpu_count=os.cpu_count(), cell_wall_s=budgets), indent=1))
    print(f"\nWP7-4 DELAY SWEEP DONE  {len(jobs)} cells  {wall:.1f}s wall", flush=True)


if __name__ == "__main__":
    main()
