"""WP7-6 - velocity-bias study: the condition (C3) actually bounds.

Why this exists
---------------
An earlier version of this campaign recorded condition (C3) as tested. It was not.
scripts/wp3_robustness.py:96 registers the condition

    c.append(("bias_add", dict(bias_xy=0.005, bias_th=np.radians(1.0))))

and those two numbers enter the simulator through `meas_off`, i.e. as a
constant offset on the MEASURED POSE.  (C3) constrains delta_v, a bias on the
achieved body VELOCITY.  They are different disturbances: a pose bias moves
where the robot thinks the target is; a velocity bias means the robot keeps
drifting when the control law has already commanded zero.

The second defect compounds the first.  ARRIVAL_TOL is 1 cm and the injected
pose bias was 5 mm, so the biased equilibrium sat INSIDE the arrival ball.
100% arrival was structurally guaranteed - the test had no way to fail.

The threshold
-------------
    delta_v_bar  <  Kp_min * eps / 4  =  0.1 * 0.01 / 4  =  2.5e-4 m/s

This script injects a persistent additive velocity bias directly, via the new
Disturbances.vel_bias (default 0.0, so nothing existing changes), and sweeps it
by orders of magnitude across that threshold - below it, at it, and far above,
including the 0.027 m/s median measured in WP2.

Two passes per (delta_v, controller)
------------------------------------
ARRIVE pass - the campaign's own protocol: the rollout STOPS the first time the
    robot crosses inside the 1 cm ball.  This is what `arrived`, `t_end`,
    `final_d_m` and `e_total` report.  Note that a first crossing is cheap: a
    drifting robot can sweep through the ball on its way past, so this pass
    measures whether the target is ever REACHED, not whether it is HELD.

HOLD pass - identical rollout with the arrival test disabled (module-level
    ARRIVAL_TOL temporarily set negative inside this script only, so no row
    ever terminates early), run to the full horizon.  `hold_final_d_m` is the
    distance to the target at T_max, i.e. a sample of the residual set the
    biased loop settles into - the quantity (C3) actually bounds.  Without this
    pass the terminal-distance column would be pinned at the tolerance for
    every arriving run and could not discriminate between bias levels, which is
    the same structural blindness the original test had.

Interpretation of the numbers is deliberately left to the reader; this script
only reports arrival, divergence, arrival time, terminal distance and E.
"""
import csv, json, os, pathlib, sys, time

import numpy as np

CODE = pathlib.Path(__file__).resolve().parents[1]   # repository root
sys.path.insert(0, str(CODE / "src"))

import torch                                                  # noqa: E402
import ddwmr.sim as ddsim                                      # noqa: E402
from ddwmr.sim import run_closedloop, LyapGains, Disturbances  # noqa: E402
from ddwmr.metrics import anchor_metric                        # noqa: E402
from ddwmr.pnn import GainMLP, predict_gains                   # noqa: E402
from ddwmr.params import LYAP1, LYAP2, ARRIVAL_TOL, KP_MIN     # noqa: E402

PASSPORT = CODE / "passport"
OUT = CODE / "wp7"
RES, LOGS = OUT / "results", OUT / "logs"
for d in (RES, LOGS):
    d.mkdir(parents=True, exist_ok=True)

SEED = 60_2026                       # shared target draw
N_MC = int(os.environ.get("WP7_N_MC", 300))
T_MAX = float(os.environ.get("WP7_TMAX", 30.0))

# (C3): delta_v_bar < Kp_min * eps / 4
C3_THRESHOLD = KP_MIN * ARRIVAL_TOL / 4.0                     # 2.5e-4 m/s

# below (C3) | at (C3) | above, by orders of magnitude; 2e-2 straddles the
# 0.027 m/s median velocity bias measured in WP2.
VEL_BIASES = [float(x) for x in os.environ.get(
    "WP7_BIASES", "0,1e-4,2.5e-4,1e-3,5e-3,2e-2,5e-2").split(",")]


def best_bptt_seed():
    import pandas as pd
    se = pd.read_csv(PASSPORT / "results" / "wp1_sample_efficiency.csv")
    b = se[se.method == "bptt"]
    last = b[b.rollouts_per_target == b.rollouts_per_target.max()]
    return int(last.sort_values("median_E_ratio").iloc[0]["seed"])


def load_net(name, head):
    net = GainMLP(head=head)
    net.load_state_dict(torch.load(PASSPORT / "models" / name, weights_only=True))
    net.eval()
    return net


def build_targets():
    """Workspace target draw shared with the delay sweep (same order, same seed)."""
    rng = np.random.default_rng(SEED)
    ang = rng.uniform(-0.7 * np.pi, 0.7 * np.pi, N_MC)
    rad = rng.uniform(0.1, 1.0, N_MC)
    return np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)


def main():
    t_all = time.perf_counter()
    targets = build_targets()
    bseed = best_bptt_seed()
    net = load_net(f"policy_bptt_seed{bseed}.pt", "sigmoid")

    g_pol = predict_gains(net, targets)
    ctrls = {
        "ours_bptt": LyapGains(Kp=g_pol[:, 0].copy(), Kth=g_pol[:, 1].copy()),
        "lyap1": LyapGains(Kp=np.full(N_MC, LYAP1[0]), Kth=np.full(N_MC, LYAP1[1])),
        "lyap2": LyapGains(Kp=np.full(N_MC, LYAP2[0]), Kth=np.full(N_MC, LYAP2[1])),
    }

    print(f"(C3) threshold delta_v_bar < Kp_min*eps/4 = {KP_MIN}*{ARRIVAL_TOL}/4 "
          f"= {C3_THRESHOLD:.3e} m/s")
    print(f"arrival tolerance eps = {ARRIVAL_TOL} m; N = {N_MC} runs per level; "
          f"T_max = {T_MAX} s\n")
    print(f"{'delta_v [m/s]':>13} {'controller':<11} {'arrived%':>9} {'div':>4} "
          f"{'med t_end':>10} {'med final_d':>12} {'med hold_d':>12} "
          f"{'med e_total':>12}")

    rows, summary = [], []
    for vb in VEL_BIASES:
        dist = Disturbances(vel_bias=vb)
        for cname, gains in ctrls.items():
            t0 = time.perf_counter()
            # --- ARRIVE pass: the campaign protocol (stops at first crossing) ---
            r = run_closedloop(targets, "lyap", gains, T_max=T_MAX, dist=dist,
                               rng=np.random.default_rng(777))
            m = anchor_metric(r.e_l_mean, r.theta_end, r.t_end,
                              np.zeros((N_MC, 2)), targets, C_l=1.0)
            # --- HOLD pass: same rollout, arrival test disabled, full horizon ---
            tol_saved = ddsim.ARRIVAL_TOL
            ddsim.ARRIVAL_TOL = -1.0        # no row can ever satisfy dn < tol
            try:
                rh = run_closedloop(targets, "lyap", gains, T_max=T_MAX, dist=dist,
                                    rng=np.random.default_rng(777))
            finally:
                ddsim.ARRIVAL_TOL = tol_saved
            for i in range(N_MC):
                rows.append([
                    f"{vb:.6g}", cname, i,
                    f"{targets[i,0]:.4f}", f"{targets[i,1]:.4f}",
                    f"{gains.Kp[i]:.6f}", f"{gains.Kth[i]:.6f}",
                    int(r.arrived[i]), int(r.diverged[i]), f"{r.t_end[i]:.3f}",
                    f"{r.final_d[i]:.6f}",
                    f"{r.final_xy[i,0]:.5f}", f"{r.final_xy[i,1]:.5f}",
                    f"{rh.final_d[i]:.6f}", int(rh.diverged[i]),
                    f"{m['e_total'][i]:.2f}", f"{m['e_line'][i]:.2f}",
                    f"{m['e_th'][i]:.2f}", f"{m['e_time'][i]:.2f}",
                    f"{r.e_l_mean[i]:.6f}", f"{r.e_l_rmse[i]:.6f}",
                    f"{r.energy[i]:.4f}", f"{r.peak_V[i]:.3f}"])
            row = dict(vel_bias=vb, controller=cname,
                       arrived_pct=float(r.arrived.mean() * 100.0),
                       n_arrived=int(r.arrived.sum()),
                       n_diverged=int(r.diverged.sum()),
                       median_t_end=float(np.median(r.t_end)),
                       median_final_d=float(np.median(r.final_d)),
                       p95_final_d=float(np.percentile(r.final_d, 95)),
                       median_hold_final_d=float(np.median(rh.final_d)),
                       p95_hold_final_d=float(np.percentile(rh.final_d, 95)),
                       n_hold_diverged=int(rh.diverged.sum()),
                       median_e_total=float(np.median(m["e_total"])),
                       min_Kp=float(np.min(gains.Kp)),
                       wall_s=round(time.perf_counter() - t0, 1))
            summary.append(row)
            print(f"{vb:>13.4g} {cname:<11} {row['arrived_pct']:>8.1f}% "
                  f"{row['n_diverged']:>4d} {row['median_t_end']:>10.3f} "
                  f"{row['median_final_d']:>12.5f} "
                  f"{row['median_hold_final_d']:>12.5f} "
                  f"{row['median_e_total']:>12.1f}", flush=True)
        print("", flush=True)

    with open(RES / "wp7_bias.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["vel_bias", "controller", "run", "x_ref", "y_ref", "Kp", "Kth",
                    "arrived", "diverged", "t_end_s", "final_d_m",
                    "final_x", "final_y", "hold_final_d_m", "hold_diverged",
                    "e_total_Cl1m", "e_line", "e_th",
                    "e_time", "e_l_mean_m", "cross_track_rmse_m", "energy_J",
                    "peak_V"])
        w.writerows(rows)

    # first level at which arrival degrades, per controller (reported, not judged)
    first_deg = {}
    for cname in ctrls:
        base = next(s["arrived_pct"] for s in summary
                    if s["controller"] == cname and s["vel_bias"] == 0.0)
        hit = [s["vel_bias"] for s in summary
               if s["controller"] == cname and s["arrived_pct"] < base]
        first_deg[cname] = (min(hit) if hit else None)

    meta = dict(
        script="scripts/wp7_bias.py",
        defect_addressed=("wp3_robustness.py:96 injected a POSE bias "
                          "(bias_xy=0.005 m, bias_th=1 deg) via meas_off, not the "
                          "VELOCITY bias delta_v that (C3) constrains; and 5 mm < "
                          "the 1 cm arrival tolerance, so the biased equilibrium "
                          "lay inside the arrival ball"),
        injection=("Disturbances.vel_bias: additive offset on the achieved body "
                   "linear velocity in ddwmr.sim._deriv, i.e. the kinematics "
                   "integrate v_true + delta_v while the encoders, integrator and "
                   "observer keep reporting the unbiased wheel speeds. Default 0.0"),
        c3=dict(formula="delta_v_bar < Kp_min * eps / 4", Kp_min=KP_MIN,
                eps_m=ARRIVAL_TOL, threshold_m_s=C3_THRESHOLD,
                paper_measured_median_m_s=0.027),
        seed=SEED, n_runs_per_level=N_MC, T_max_s=T_MAX,
        vel_biases=VEL_BIASES, controllers=list(ctrls),
        ours_bptt_seed=bseed,
        target_construction=("rad ~ U(0.1,1.0), ang ~ U(-0.7pi,0.7pi), "
                             "default_rng(60_2026); shared with the delay sweep"),
        first_level_arrival_degrades=first_deg,
        passes=dict(
            arrive=("campaign protocol: rollout stops the first time the robot "
                    "crosses inside the 1 cm ball; source of arrived/t_end/"
                    "final_d_m/e_total"),
            hold=("identical rollout with ddwmr.sim.ARRIVAL_TOL patched negative "
                  "inside this script only, so no row terminates early; "
                  "hold_final_d_m is the distance to the target at T_max, a "
                  "sample of the residual set the biased loop settles into")),
        summary=summary,
        total_wall_s=round(time.perf_counter() - t_all, 1),
    )
    (LOGS / "wp7_bias_meta.json").write_text(json.dumps(meta, indent=1))

    print(f"first delta_v at which arrival drops below its delta_v=0 value:")
    for c, v in first_deg.items():
        print(f"   {c:<11} {v if v is not None else 'none in sweep'}")
    print(f"\nwrote {RES/'wp7_bias.csv'}  ({len(rows)} rows)")
    print(f"wrote {LOGS/'wp7_bias_meta.json'}")
    print(f"total wall {meta['total_wall_s']} s")


if __name__ == "__main__":
    main()
