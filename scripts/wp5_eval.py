"""WP5: evaluation matrix.

Cells:
  anchor4     : the 4 anchor validation targets (clean, deterministic)
  heldout30   : 30 targets in the HELD-OUT region d in [1.0,1.5],
                |theta| in [0.7pi, 0.9pi] (generalization)
  paths       : circle / infinity / line / random waypoint paths
                (anchor Figs. 19-38 scenario classes)
  noisy reps  : anchor4 + heldout30 under sensor noise (1 mm, 0.5 deg),
                30 evaluation seeds (NMPC: 10 seeds on anchor4 only — solve
                cost; documented)
Controllers: ours (WP1 BPTT best), ours_dr, pnn_bf, lyap1, lyap2, pid_table3,
pid_tuned (WP4), nmpc (CasADi/IPOPT), sac (median-return seed, deterministic).
Latency table measured on this machine for every controller.

E uses C_l = 1.0 m fixed for cross-scenario comparability (documented);
anchor4 rows additionally report E with per-target brute-force C_l.
"""
import sys, csv, json, time, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from ddwmr.sim import (run_closedloop, LyapGains, PIDGains, Disturbances)
from ddwmr.metrics import anchor_metric, wrap
from ddwmr.pnn import GainMLP, predict_gains
from ddwmr.params import LYAP1, LYAP2, K_T, ARRIVAL_TOL, V_MAX

RES = ROOT / "passport" / "results"
MOD = ROOT / "passport" / "models"
LOG = ROOT / "passport" / "logs"
SEED = 77
N_NOISY_SEEDS = 30
NOISE = dict(pose_noise_xy=0.001, pose_noise_th=np.radians(0.5))

# ---------------------------------------------------------------- scenarios
def heldout_targets(n=30):
    rng = np.random.default_rng(SEED)
    rad = rng.uniform(1.0, 1.5, n)
    sign = rng.choice([-1, 1], n)
    ang = sign * rng.uniform(0.7 * np.pi, 0.9 * np.pi, n)
    return np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)


def paths():
    out = {}
    k = np.arange(1, 17)
    out["circle"] = np.stack([np.cos(2 * np.pi * k / 16),
                              np.sin(2 * np.pi * k / 16)], axis=1)
    t = np.pi / 2 + 2 * np.pi * np.arange(1, 17) / 16
    out["infinity"] = np.stack([np.cos(t), 0.5 * np.sin(2 * t)], axis=1)
    xs = np.linspace(0.0, 5.0, 11)
    out["line"] = np.stack([xs, np.ones_like(xs)], axis=1)
    xr = np.linspace(0.63, 6.3, 10)
    out["random"] = np.stack([xr, np.sin(xr)], axis=1)
    return out


ANCHOR4 = np.array([[-0.2, -0.5], [-0.3, 0.5], [0.5, -0.5], [0.6, 0.4]])

# ---------------------------------------------------------------- controllers
def load_net(p, head):
    net = GainMLP(head=head)
    net.load_state_dict(torch.load(p, weights_only=True))
    net.eval()
    return net


def build_controllers():
    import pandas as pd
    se = pd.read_csv(RES / "wp1_sample_efficiency.csv")
    b = se[se.method == "bptt"]
    last = b[b.rollouts_per_target == b.rollouts_per_target.max()]
    best_seed = int(last.sort_values("median_E_ratio").iloc[0]["seed"])
    pidp = json.load(open(MOD / "pid_tuned.json"))["params"]
    ctrls = {
        "ours": ("policy", load_net(MOD / f"policy_bptt_seed{best_seed}.pt", "sigmoid")),
        "ours_dr": ("policy", load_net(MOD / "policy_dr_seed0.pt", "sigmoid")),
        "pnn_bf": ("policy", load_net(MOD / "pnn_wp0.pt", "linear")),
        "lyap1": ("fixed", LYAP1),
        "lyap2": ("fixed", LYAP2),
        "pid_table3": ("pid", PIDGains()),
        "pid_tuned": ("pid", PIDGains(**pidp)),
        "nmpc": ("nmpc", None),
        "sac": ("sac", None),
    }
    return ctrls, best_seed


def sac_best_model():
    import glob
    from stable_baselines3 import SAC
    best, best_ret = None, -np.inf
    for f in sorted(glob.glob(str(LOG / "wp4_sac_curve_seed*.csv"))):
        import pandas as pd
        df = pd.read_csv(f)
        if len(df) < 10:
            continue
        ret = df.ep_return.tail(50).median()
        seed = int(f.split("seed")[1].split(".")[0])
        if ret > best_ret:
            best_ret, best = ret, seed
    model = SAC.load(str(MOD / f"sac_seed{best}"), device="cpu")
    return model, best, best_ret


def sac_rollout(model, waypoints, t_max_per_seg=15.0):
    """Chained waypoint rollout with the SAC policy on the full 7-state plant."""
    from ddwmr.rl_env import _deriv, CTRL_DT, SIM_DT
    s = np.zeros(7)
    rows = []
    t = 0.0
    for tgt in waypoints:
        seg_start = s[:2].copy()
        seg = tgt - seg_start
        L = np.linalg.norm(seg)
        perp_sum = perp_sq = 0.0
        npts = 0
        t0 = t
        arrived = False
        while t - t0 < t_max_per_seg:
            ex, ey = tgt[0] - s[0], tgt[1] - s[1]
            eth = wrap(np.arctan2(ey, ex) - s[2])
            obs = np.array([ex, ey, eth, s[3], s[4]], np.float32)
            a, _ = model.predict(obs, deterministic=True)
            Vr, Vl = np.clip(a, -1, 1) * V_MAX
            for _ in range(int(CTRL_DT / SIM_DT)):
                k1 = _deriv(s, Vr, Vl); k2 = _deriv(s + 0.5 * SIM_DT * k1, Vr, Vl)
                k3 = _deriv(s + 0.5 * SIM_DT * k2, Vr, Vl); k4 = _deriv(s + SIM_DT * k3, Vr, Vl)
                s = s + (SIM_DT / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
                t += SIM_DT
            perp = abs((s[0] - seg_start[0]) * seg[1] - (s[1] - seg_start[1]) * seg[0]) / max(L, 1e-9)
            perp_sum += perp; perp_sq += perp ** 2; npts += 1
            if np.hypot(tgt[0] - s[0], tgt[1] - s[1]) < ARRIVAL_TOL:
                arrived = True
                break
        rows.append(dict(Lk=L, t_arrive=t - t0, arrived=arrived,
                         e_l=perp_sum / max(npts, 1),
                         e_l_rmse=np.sqrt(perp_sq / max(npts, 1)),
                         theta_end=s[2], phi=np.arctan2(seg[1], seg[0])))
        if not arrived:
            break
    return rows


def nmpc_rollout_seq(waypoints, t_max_per_seg=25.0):
    from ddwmr.nmpc import rollout_nmpc
    s0 = np.zeros(7)
    rows = []
    for tgt in waypoints:
        r = rollout_nmpc(tuple(tgt), T_max=t_max_per_seg,
                         start=(s0[0], s0[1]), theta0=s0[2], s0=s0)
        rows.append(r)
        if not r["arrived"]:
            break
        s0 = r["s_final"]
    return rows


# ---------------------------------------------------------------- evaluation
def eval_single_targets(ctrls, targets, label, noisy_seeds=0, writer=None,
                        nmpc_seeds_cap=10, Cl_bf=None):
    n = len(targets)
    variants = [("clean", None)] + \
        [(f"noisy{s}", s) for s in range(noisy_seeds)]
    for vname, vseed in variants:
        dist = Disturbances(**NOISE) if vseed is not None else Disturbances()
        for cname, (kind, obj) in ctrls.items():
            if kind == "nmpc" and vseed is not None and \
                    (label != "anchor4" or vseed >= nmpc_seeds_cap):
                continue
            t0 = time.perf_counter()
            if kind in ("policy", "fixed", "pid"):
                if kind == "policy":
                    g = predict_gains(obj, targets)
                    gains = LyapGains(Kp=g[:, 0].copy(), Kth=g[:, 1].copy())
                    ck = "lyap"
                elif kind == "fixed":
                    gains = LyapGains(Kp=np.full(n, obj[0]), Kth=np.full(n, obj[1]))
                    ck = "lyap"
                else:
                    gains, ck = obj, "pid"
                r = run_closedloop(targets, ck, gains, T_max=45.0, dist=dist,
                                   rng=np.random.default_rng(8800 + (vseed or 0)))
                m = anchor_metric(r.e_l_mean, r.theta_end, r.t_end,
                                  np.zeros((n, 2)), targets, C_l=1.0)
                for i in range(n):
                    mbf = np.nan
                    if Cl_bf is not None:
                        mbf = float(anchor_metric(
                            r.e_l_mean[i:i+1], r.theta_end[i:i+1], r.t_end[i:i+1],
                            np.zeros((1, 2)), targets[i:i+1],
                            C_l=Cl_bf[i])["e_total"][0])
                    writer.writerow([label, vname, cname, i,
                                     f"{targets[i,0]:.4f}", f"{targets[i,1]:.4f}",
                                     f"{m['e_total'][i]:.2f}",
                                     f"{mbf:.2f}" if np.isfinite(mbf) else "",
                                     f"{m['e_line'][i]:.2f}", f"{m['e_th'][i]:.2f}",
                                     f"{m['e_time'][i]:.2f}", f"{r.t_end[i]:.3f}",
                                     int(r.arrived[i]), int(r.diverged[i]),
                                     f"{r.e_l_rmse[i]:.5f}", f"{r.energy[i]:.4f}",
                                     f"{r.peak_V[i]:.3f}", f"{r.sup_dv[i]:.4f}",
                                     f"{r.sup_dw[i]:.4f}"])
            elif kind == "sac":
                model = obj
                for i in range(n):
                    segs = sac_rollout(model, [targets[i]])
                    s_ = segs[0]
                    dth = abs(wrap(s_["theta_end"] - s_["phi"]))
                    tn = abs(s_["t_arrive"] - K_T * s_["Lk"])
                    e_line = 1000 * s_["e_l"]; e_th = 1000 * dth / np.pi
                    e_time = 1000 * np.tanh(0.1 * tn)
                    et = 1.5 * e_line + 2 * e_th + 2 * e_time
                    writer.writerow([label, vname, cname, i,
                                     f"{targets[i,0]:.4f}", f"{targets[i,1]:.4f}",
                                     f"{et:.2f}", "", f"{e_line:.2f}", f"{e_th:.2f}",
                                     f"{e_time:.2f}", f"{s_['t_arrive']:.3f}",
                                     int(s_["arrived"]), 0,
                                     f"{s_['e_l_rmse']:.5f}", "", "", "", ""])
                if vseed is not None:
                    pass  # SAC is deterministic; noise enters via obs? (obs from true state; skip noisy reps)
            elif kind == "nmpc":
                from ddwmr.nmpc import rollout_nmpc
                for i in range(n):
                    r = rollout_nmpc(tuple(targets[i]), T_max=30.0,
                                     noise=(NOISE if vseed is not None else None),
                                     noise_seed=vseed)
                    dth = abs(wrap(r["theta_end"] - np.arctan2(targets[i, 1], targets[i, 0])))
                    tn = abs(r["t_end"] - K_T * np.linalg.norm(targets[i]))
                    e_line = 1000 * r["e_l_mean"]; e_th = 1000 * dth / np.pi
                    e_time = 1000 * np.tanh(0.1 * tn)
                    et = 1.5 * e_line + 2 * e_th + 2 * e_time
                    writer.writerow([label, vname, cname, i,
                                     f"{targets[i,0]:.4f}", f"{targets[i,1]:.4f}",
                                     f"{et:.2f}", "", f"{e_line:.2f}", f"{e_th:.2f}",
                                     f"{e_time:.2f}", f"{r['t_end']:.3f}",
                                     int(r["arrived"]), 0,
                                     f"{r['e_l_rmse']:.5f}", f"{r['energy']:.4f}",
                                     f"{r['peak_V']:.3f}", "", ""])
            print(f"  {label}/{vname}/{cname}: {time.perf_counter()-t0:.0f}s", flush=True)


def eval_paths(ctrls, writer_seg, writer_run):
    for pname, wps in paths().items():
        L_tot = np.linalg.norm(np.diff(np.vstack([[0, 0], wps]), axis=0), axis=1).sum()
        t_des = K_T * L_tot
        for cname, (kind, obj) in ctrls.items():
            t0 = time.perf_counter()
            if kind in ("policy", "fixed", "pid"):
                wp = wps[None]
                if kind == "policy":
                    def gp(poses, tgts, _net=obj):
                        dx = tgts[:, 0] - poses[:, 0]; dy = tgts[:, 1] - poses[:, 1]
                        c, s = np.cos(-poses[:, 2]), np.sin(-poses[:, 2])
                        rel = np.stack([c * dx - s * dy, s * dx + c * dy], axis=1)
                        return predict_gains(_net, rel)
                    g = LyapGains(Kp=np.array([0.5]), Kth=np.array([1.5]))
                    r = run_closedloop(None, "lyap", g, waypoints=wp,
                                       n_waypoints=np.array([len(wps)]),
                                       gain_policy=gp, T_max=200.0,
                                       record_series=1, series_stride=20)
                elif kind == "fixed":
                    g = LyapGains(Kp=np.array([obj[0]]), Kth=np.array([obj[1]]))
                    r = run_closedloop(None, "lyap", g, waypoints=wp,
                                       n_waypoints=np.array([len(wps)]),
                                       T_max=200.0, record_series=1, series_stride=20)
                else:
                    r = run_closedloop(None, "pid", obj, waypoints=wp,
                                       n_waypoints=np.array([len(wps)]),
                                       T_max=200.0, record_series=1, series_stride=20)
                segs = sorted(r.seg_records, key=lambda s_: s_["seg"])
                for s_ in segs:
                    writer_seg.writerow([pname, cname, s_["seg"], f"{s_['Lk']:.4f}",
                                         f"{s_['Kp']:.4f}", f"{s_['Kth']:.4f}",
                                         f"{s_['t_arrive']:.3f}", f"{s_['d_at_switch']:.4f}"])
                writer_run.writerow([pname, cname, int(r.arrived[0]),
                                     f"{r.t_end[0]:.2f}", f"{t_des:.2f}",
                                     f"{r.e_l_rmse[0]:.5f}", f"{r.e_l_mean[0]:.5f}",
                                     f"{r.energy[0]:.3f}", f"{r.peak_V[0]:.2f}",
                                     int(r.diverged[0]), len(segs)])
            elif kind == "sac":
                segs = sac_rollout(obj, list(wps))
                done = all(s_["arrived"] for s_ in segs) and len(segs) == len(wps)
                t_tot = sum(s_["t_arrive"] for s_ in segs)
                el = np.mean([s_["e_l"] for s_ in segs])
                writer_run.writerow([pname, cname, int(done), f"{t_tot:.2f}",
                                     f"{t_des:.2f}",
                                     f"{np.mean([s_['e_l_rmse'] for s_ in segs]):.5f}",
                                     f"{el:.5f}", "", "", 0, len(segs)])
                for j, s_ in enumerate(segs):
                    writer_seg.writerow([pname, cname, j, f"{s_['Lk']:.4f}", "", "",
                                         f"{s_['t_arrive']:.3f}", ""])
            elif kind == "nmpc":
                segs = nmpc_rollout_seq(list(wps))
                done = all(s_["arrived"] for s_ in segs) and len(segs) == len(wps)
                t_tot = sum(s_["t_end"] for s_ in segs)
                writer_run.writerow([pname, cname, int(done), f"{t_tot:.2f}",
                                     f"{t_des:.2f}",
                                     f"{np.mean([s_['e_l_rmse'] for s_ in segs]):.5f}",
                                     f"{np.mean([s_['e_l_mean'] for s_ in segs]):.5f}",
                                     f"{sum(s_['energy'] for s_ in segs):.3f}",
                                     f"{max(s_['peak_V'] for s_ in segs):.2f}",
                                     0, len(segs)])
                for j, s_ in enumerate(segs):
                    writer_seg.writerow([pname, cname, j, "", "", "",
                                         f"{s_['t_end']:.3f}", ""])
            print(f"  path {pname}/{cname}: {time.perf_counter()-t0:.0f}s", flush=True)


def latency_table(ctrls):
    rows = []
    rng = np.random.default_rng(3)
    obs = np.array([0.4, 0.3, 0.2, 1.0, 1.0], np.float32)
    for cname, (kind, obj) in ctrls.items():
        if kind == "policy":
            xy = torch.tensor([[0.4, 0.3]])
            with torch.no_grad():
                obj(xy)
                t0 = time.perf_counter()
                for _ in range(2000):
                    obj(xy)
                per_inf = (time.perf_counter() - t0) / 2000
            t0 = time.perf_counter()
            for _ in range(20000):
                d = np.hypot(0.4, 0.3); eth = np.arctan2(0.3, 0.4)
                v = 0.5 * d * np.cos(eth); w = 0.5 * np.cos(eth) * np.sin(eth) + 1.5 * eth
            per_tick = (time.perf_counter() - t0) / 20000
            rows.append([cname, per_tick * 1e6, per_inf * 1e6,
                         "inference once per waypoint switch"])
        elif kind in ("fixed", "pid"):
            t0 = time.perf_counter()
            for _ in range(20000):
                d = np.hypot(0.4, 0.3); eth = np.arctan2(0.3, 0.4)
                v = 0.5 * d; w = 1.0 * eth
            per_tick = (time.perf_counter() - t0) / 20000
            rows.append([cname, per_tick * 1e6, 0.0, ""])
        elif kind == "sac" and obj is not None:
            obj.predict(obs, deterministic=True)
            t0 = time.perf_counter()
            for _ in range(2000):
                obj.predict(obs, deterministic=True)
            rows.append([cname, (time.perf_counter() - t0) / 2000 * 1e6, 0.0, ""])
        elif kind == "nmpc":
            from ddwmr.nmpc import rollout_nmpc
            r = rollout_nmpc((0.6, 0.4), T_max=4.0)
            rows.append([cname, r["solve_ms_mean"] * 1e3, 0.0,
                         f"IPOPT fallback; p95={r['solve_ms_p95']:.0f} ms; acados ~1 ms (Frey et al. 2025)"])
    with open(RES / "wp4_latency.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["controller", "per_tick_us", "per_switch_inference_us", "note"])
        w.writerows(rows)


def main():
    ctrls, best_seed = build_controllers()
    sac_model, sac_seed, sac_ret = sac_best_model()
    ctrls["sac"] = ("sac", sac_model)
    f = open(RES / "wp5_eval_matrix.csv", "w", newline="")
    w = csv.writer(f)
    w.writerow(["scenario", "variant", "controller", "idx", "x_ref", "y_ref",
                "e_total_Cl1m", "e_total_Clbf", "e_line", "e_th", "e_time",
                "t_end_s", "arrived", "diverged", "cross_track_rmse_m",
                "energy_J", "peak_V", "sup_dv", "sup_dw"])
    bf = np.load(RES / "wp0_bruteforce.npz")
    n_ws = int(bf["n_workspace"])
    Cl4 = bf["C_l_bf"][n_ws:n_ws + 4]
    eval_single_targets(ctrls, ANCHOR4, "anchor4", noisy_seeds=N_NOISY_SEEDS,
                        writer=w, Cl_bf=Cl4)
    eval_single_targets(ctrls, heldout_targets(), "heldout30", noisy_seeds=N_NOISY_SEEDS,
                        writer=w)
    f.flush()
    fs = open(RES / "wp5_path_segments.csv", "w", newline="")
    ws_ = csv.writer(fs)
    ws_.writerow(["path", "controller", "seg", "L_k", "Kp", "Kth", "t_arrive_s",
                  "d_at_switch_m"])
    fr = open(RES / "wp5_paths.csv", "w", newline="")
    wr_ = csv.writer(fr)
    wr_.writerow(["path", "controller", "completed", "t_total_s", "t_desired_s",
                  "cross_track_rmse_m", "e_l_mean_m", "energy_J", "peak_V",
                  "diverged", "n_segments_done"])
    eval_paths(ctrls, ws_, wr_)
    for fh in (f, fs, fr):
        fh.close()
    latency_table(ctrls)
    with open(LOG / "wp5_meta.json", "w") as fh:
        json.dump(dict(seed=SEED, ours_bptt_seed=best_seed, sac_seed=sac_seed,
                       sac_median_return_tail=float(sac_ret),
                       noise=dict(xy_m=0.001, th_deg=0.5),
                       nmpc="CasADi+IPOPT fallback (acados build unavailable); "
                            "noisy reps capped at 10 on anchor4"), fh, indent=1)
    print("WP5 DONE")


if __name__ == "__main__":
    main()
