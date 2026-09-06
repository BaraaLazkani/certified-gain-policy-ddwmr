"""WP7-5 - one latency harness, one statistic, every controller.

Why this exists
---------------
Two defects made the earlier latency measurement in this repository not well posed.

1.  Different statistics.  An earlier measurement reported the MINIMUM over 7
    repeats for the Lyapunov/PID stack ("least contaminated by scheduler
    noise"), while the NMPC entry is carried over from
    passport/results/wp4_latency.csv as a MEAN of 75888.7 us whose own note
    records p95 = 52 ms.  A mean above the p95 is outlier-dominated by
    definition.  Dividing an outlier-inflated mean by a best-case minimum is
    what produces the 7.2e3x ratio; it is an artefact of the two statistics,
    not a measurement.

2.  Misassigned inference.  The earlier table gave every policy row the
    same 44.9 us inference figure, including "PNN (brute force)", whose own
    measured inference in wp4_latency.csv is 27.5 us.  The linear-head PNN and
    the sigmoid-head certified policy are different forward passes and are
    timed separately here.

What this script does
---------------------
Every controller is measured by ONE harness: each individual invocation is
timed with time.perf_counter() and the SAME four statistics are reported for
all of them - median, p95, mean, n.  Nothing is carried over from a stored
scalar; the NMPC number comes from actually running ddwmr.nmpc.rollout_nmpc
and keeping the per-solve distribution.

Measured units (column `unit`):
  control_tick  a complete per-tick control computation, i.e. everything that
                must run between two samples on the robot
  inference     a gain-policy forward pass, which runs ONCE per waypoint
                segment, not once per tick
  partial_tick  the outer-loop algebra alone - what the original 1.8 us figure
                measured; kept only so the two are comparable

The Lyapunov control tick uses the earlier construction, which is
correct: pose error -> NPC law (Eqs. 25-26) -> wheel-speed geometry ->
integral action (Eqs. 44-45) -> observer update (Eqs. 50-51) -> inner LQR +
saturation (Eqs. 46-49).  It does not depend on WHICH Kp,Kth are used, so the
whole Lyapunov family (ours, ours_dr, pnn_bf, lyap1, lyap2) shares it; what
distinguishes them is the per-segment inference, which is timed per policy.

These are NumPy/PyTorch/CasADi numbers on one desktop CPU: an upper bound on a
compiled embedded implementation, reported as measured.
"""
import csv, glob, json, os, pathlib, platform, subprocess, sys, time

import numpy as np

CODE = pathlib.Path(__file__).resolve().parents[1]   # repository root
sys.path.insert(0, str(CODE / "src"))

import torch                                                  # noqa: E402
from ddwmr.params import (A4_NOM, B4_NOM, K2, KE, DT, NOMINAL, V_MAX,   # noqa: E402
                          PID_TABLE3, LYAP1, LYAP2)
from ddwmr.pnn import GainMLP                                 # noqa: E402

PASSPORT = CODE / "passport"
OUT = CODE / "wp7"
RES, LOGS = OUT / "results", OUT / "logs"
for d in (RES, LOGS):
    d.mkdir(parents=True, exist_ok=True)

# ---- one harness, one set of knobs -----------------------------------------
BUDGET_S = float(os.environ.get("WP7_LAT_BUDGET", 4.0))   # wall time per unit
N_MIN = int(os.environ.get("WP7_LAT_NMIN", 300))          # samples floor
N_MAX = int(os.environ.get("WP7_LAT_NMAX", 30_000))       # samples ceiling
WARMUP = int(os.environ.get("WP7_LAT_WARMUP", 200))       # untimed calls first
NMPC_TARGETS = int(os.environ.get("WP7_LAT_NMPC_TGT", 6))
NMPC_TMAX = float(os.environ.get("WP7_LAT_NMPC_TMAX", 6.0))


def stats(us):
    """The SAME four statistics for every controller (plus min for contrast)."""
    a = np.asarray(us, float)
    return dict(median_us=float(np.median(a)), p95_us=float(np.percentile(a, 95)),
                mean_us=float(a.mean()), min_us=float(a.min()),
                max_us=float(a.max()), std_us=float(a.std(ddof=1)) if a.size > 1 else 0.0,
                n=int(a.size))


def bench(fn, *, budget_s=BUDGET_S, n_min=N_MIN, n_max=N_MAX, warmup=WARMUP):
    """Time every individual invocation. Stop at the wall budget, bounded by
    [n_min, n_max] samples. Same primitive for ticks, inference and solves."""
    for _ in range(warmup):
        fn()
    ts = np.empty(n_max)
    t_start = time.perf_counter()
    k = 0
    while k < n_max:
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        ts[k] = t1 - t0
        k += 1
        if k >= n_min and (t1 - t_start) >= budget_s:
            break
    return ts[:k] * 1e6


# --------------------------------------------------------------- control ticks

def make_lyap_tick(Kp=0.5, Kth=1.5):
    """Complete tick (verbatim construction):
    error -> NPC -> geometry -> integrator -> observer -> LQR + saturation."""
    n = NOMINAL
    st = dict(xhat=np.zeros(4), eps=np.zeros(2))
    tgt = np.array([0.4, 0.3])
    pose = np.array([0.05, 0.02, 0.1])
    wheels = np.array([1.2, 1.1])               # wr, wl measured

    def tick():
        ex = tgt[0] - pose[0]; ey = tgt[1] - pose[1]
        d = np.hypot(ex, ey)
        eth = np.arctan2(ey, ex) - pose[2]
        eth = (eth + np.pi) % (2 * np.pi) - np.pi
        ce = np.cos(eth); se = np.sin(eth)
        v_ref = Kp * d * ce
        w_ref = Kp * ce * se + Kth * eth
        wr_ref = (v_ref + 0.5 * n.d * w_ref) / n.r
        wl_ref = (v_ref - 0.5 * n.d * w_ref) / n.r
        st["eps"] = st["eps"] + DT * np.array([wr_ref - wheels[0],
                                               wl_ref - wheels[1]])
        xhat = st["xhat"]
        V = np.clip(st["eps"] - K2 @ xhat, -V_MAX, V_MAX)
        innov = wheels - xhat[:2]
        st["xhat"] = xhat + DT * (A4_NOM @ xhat + B4_NOM @ V + KE @ innov)
        return V

    return tick


def make_npc_only_tick(Kp=0.5, Kth=1.5):
    """Outer-loop algebra alone: what the original 1.8 us figure measured."""
    def tick():
        d = np.hypot(0.4, 0.3); eth = np.arctan2(0.3, 0.4)
        v = Kp * d * np.cos(eth)
        w = Kp * np.cos(eth) * np.sin(eth) + Kth * eth
        return v, w
    return tick


def make_pid_tick():
    """Complete tick for the dual-PID outer loop (same inner loop)."""
    n = NOMINAL
    g = PID_TABLE3
    st = dict(xhat=np.zeros(4), eps=np.zeros(2), Id=0.0, Ieth=0.0)
    tgt = np.array([0.4, 0.3]); pose = np.array([0.05, 0.02, 0.1])
    wheels = np.array([1.2, 1.1])

    def tick():
        ex = tgt[0] - pose[0]; ey = tgt[1] - pose[1]
        d = np.hypot(ex, ey)
        eth = np.arctan2(ey, ex) - pose[2]
        eth = (eth + np.pi) % (2 * np.pi) - np.pi
        v_act = (n.r / 2.0) * (wheels[0] + wheels[1])
        w_act = (n.r / n.d) * (wheels[0] - wheels[1])
        st["Id"] += DT * d; st["Ieth"] += DT * eth
        ddot = -v_act * np.cos(eth)
        ethdot = v_act * np.sin(eth) / max(d, 1e-6) - w_act
        v_ref = g["KP1"] * d + g["KI1"] * st["Id"] + g["KD1"] * ddot
        w_ref = g["KP2"] * eth + g["KI2"] * st["Ieth"] + g["KD2"] * ethdot
        wr_ref = (v_ref + 0.5 * n.d * w_ref) / n.r
        wl_ref = (v_ref - 0.5 * n.d * w_ref) / n.r
        st["eps"] = st["eps"] + DT * np.array([wr_ref - wheels[0],
                                               wl_ref - wheels[1]])
        xhat = st["xhat"]
        V = np.clip(st["eps"] - K2 @ xhat, -V_MAX, V_MAX)
        innov = wheels - xhat[:2]
        st["xhat"] = xhat + DT * (A4_NOM @ xhat + B4_NOM @ V + KE @ innov)
        return V
    return tick


# ------------------------------------------------------------------ policies

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


def make_infer(net):
    xy = torch.tensor([[0.4, 0.3]], dtype=torch.float32)

    def infer():
        with torch.no_grad():
            return net(xy)
    return infer


def best_sac_seed():
    import pandas as pd
    best, best_ret = None, -np.inf
    for f in sorted(glob.glob(str(PASSPORT / "logs" / "wp4_sac_curve_seed*.csv"))):
        df = pd.read_csv(f)
        if len(df) < 10:
            continue
        ret = float(df.ep_return.tail(50).median())
        seed = int(f.split("seed")[1].split(".")[0])
        if ret > best_ret:
            best_ret, best = ret, seed
    return best, best_ret


def make_sac_predict():
    """SAC predict() IS the whole controller: obs -> voltages, no outer loop."""
    from stable_baselines3 import SAC
    seed, ret = best_sac_seed()
    if seed is None:
        return None, None, None
    model = SAC.load(str(PASSPORT / "models" / f"sac_seed{seed}"), device="cpu")
    obs = np.array([0.4, 0.3, 0.2, 1.0, 1.0], np.float32)

    def predict():
        return model.predict(obs, deterministic=True)
    return predict, seed, ret


# ---------------------------------------------------------------------- NMPC

def nmpc_solve_times():
    """Per-solve distribution measured by RUNNING the closed loop.

    ddwmr.nmpc.rollout_nmpc is called for real; NMPCController.step is wrapped
    so that the same perf_counter primitive used everywhere else brackets the
    whole per-tick controller call (warm-start assembly + IPOPT solve), not
    just the solve() line that nmpc.py times internally.  Both are recorded.
    """
    import ddwmr.nmpc as nm
    outer, inner, per_run = [], [], []
    orig_step = nm.NMPCController.step

    def timed_step(self, state7):
        t0 = time.perf_counter()
        out = orig_step(self, state7)
        outer.append(time.perf_counter() - t0)
        return out

    nm.NMPCController.step = timed_step
    rng = np.random.default_rng(60_2026)
    ang = rng.uniform(-0.7 * np.pi, 0.7 * np.pi, NMPC_TARGETS)
    rad = rng.uniform(0.1, 1.0, NMPC_TARGETS)
    try:
        for i in range(NMPC_TARGETS):
            tgt = (float(rad[i] * np.cos(ang[i])), float(rad[i] * np.sin(ang[i])))
            n0 = len(outer)
            t0 = time.perf_counter()
            r = nm.rollout_nmpc(tgt, T_max=NMPC_TMAX)
            per_run.append(dict(target=[round(tgt[0], 4), round(tgt[1], 4)],
                                arrived=bool(r["arrived"]), t_end=round(r["t_end"], 3),
                                n_solves=int(r["n_solves"]),
                                wall_s=round(time.perf_counter() - t0, 2)))
            inner.append(r)
            print(f"  nmpc target ({tgt[0]:+.3f},{tgt[1]:+.3f}) "
                  f"solves={len(outer)-n0} arrived={r['arrived']} "
                  f"{per_run[-1]['wall_s']}s", flush=True)
    finally:
        nm.NMPCController.step = orig_step
    return np.array(outer) * 1e6, per_run


# ---------------------------------------------------------------------- main

def cpu_model():
    try:
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def main():
    t_all = time.perf_counter()
    load_before = os.getloadavg()
    rows = []

    def add(unit_id, unit, s, note, extra=""):
        rows.append([unit_id, unit,
                     f"{s['median_us']:.4f}", f"{s['p95_us']:.4f}",
                     f"{s['mean_us']:.4f}", f"{s['min_us']:.4f}",
                     f"{s['max_us']:.4f}", f"{s['std_us']:.4f}", s["n"],
                     extra, note])
        print(f"{unit_id:<28} median={s['median_us']:>11.3f}  p95={s['p95_us']:>11.3f}  "
              f"mean={s['mean_us']:>11.3f}  min={s['min_us']:>10.3f}  n={s['n']}",
              flush=True)

    print("== control ticks ==", flush=True)
    s_npc = stats(bench(make_npc_only_tick()))
    add("npc_law_only", "partial_tick", s_npc,
        "outer-loop algebra alone; the quantity the original 1.8 us figure measured")
    s_lyap = stats(bench(make_lyap_tick()))
    add("lyap_family_full_tick", "control_tick", s_lyap,
        "complete tick: error, NPC law, geometry, integral action, observer, "
        "inner LQR + saturation; shared by ours/ours_dr/pnn_bf/lyap1/lyap2",
        "ours,ours_dr,pnn_bf,lyap1,lyap2")
    s_pid = stats(bench(make_pid_tick()))
    add("pid_full_tick", "control_tick", s_pid,
        "complete tick, dual-PID outer loop, same inner loop",
        "pid_table3,pid_tuned")

    print("== gain-policy inference (once per waypoint segment) ==", flush=True)
    bseed = best_bptt_seed()
    s_sig = stats(bench(make_infer(load_net(f"policy_bptt_seed{bseed}.pt", "sigmoid"))))
    add("gainpolicy_sigmoid_inference", "inference", s_sig,
        f"GainMLP head=sigmoid (the certified BPTT policy, seed {bseed}); "
        "forward pass under no_grad", "ours,ours_dr")
    s_lin = stats(bench(make_infer(load_net("pnn_wp0.pt", "linear"))))
    add("gainpolicy_linear_pnn_inference", "inference", s_lin,
        "GainMLP head=linear (the anchor's supervised PNN, pnn_wp0.pt); "
        "measured separately - it is NOT the sigmoid figure", "pnn_bf")

    print("== SAC ==", flush=True)
    sac_seed = sac_ret = None
    s_sac = None
    try:
        predict, sac_seed, sac_ret = make_sac_predict()
        if predict is not None:
            s_sac = stats(bench(predict))
            add("sac_predict", "control_tick", s_sac,
                f"stable-baselines3 SAC.predict(deterministic=True), seed {sac_seed}; "
                "this IS the whole controller (obs -> voltages), CPU", "sac")
    except Exception as e:                                   # pragma: no cover
        print(f"  SAC unavailable: {type(e).__name__}: {e}", flush=True)

    print("== NMPC (measured by running rollout_nmpc) ==", flush=True)
    nm_us, nm_runs = nmpc_solve_times()
    s_nmpc = stats(nm_us)
    add("nmpc_solve", "control_tick", s_nmpc,
        f"CasADi/IPOPT per control tick (warm start + solve) at Ts=20 ms, N=50, "
        f"measured over {len(nm_runs)} closed-loop rollouts via rollout_nmpc; "
        "NOT carried over from wp4_latency.csv", "nmpc")

    # --- derived: per-tick cost of a policy controller, inference amortised ---
    seg_s = seg_ticks = None
    f_seg = PASSPORT / "results" / "wp2_segments.csv"
    if f_seg.exists():
        import pandas as pd
        df = pd.read_csv(f_seg)
        halt = df[df["condition"].str.startswith("halt1cm")] if "condition" in df else df
        t = halt["t_arrive_s"].to_numpy(float)
        t = t[np.isfinite(t) & (t > 0)]
        seg_s = float(np.median(t)); seg_ticks = int(round(seg_s / DT))
    for uid, s_inf, who in [("ours_amortised_tick", s_sig, "ours,ours_dr"),
                            ("pnn_bf_amortised_tick", s_lin, "pnn_bf")]:
        if seg_ticks:
            d = {k: s_lyap[k] + (s_inf[k] / seg_ticks if k.endswith("_us") else 0)
                 for k in s_lyap}
            d["n"] = min(s_lyap["n"], s_inf["n"])
            rows.append([uid, "derived",
                         f"{d['median_us']:.4f}", f"{d['p95_us']:.4f}",
                         f"{d['mean_us']:.4f}", f"{d['min_us']:.4f}",
                         f"{d['max_us']:.4f}", "", d["n"], who,
                         f"DERIVED, not measured: full tick + inference/{seg_ticks} "
                         f"ticks (median segment {seg_s:.2f} s at dt={DT*1e3:.0f} ms)"])
            print(f"{uid:<28} median={d['median_us']:>11.3f}  p95={d['p95_us']:>11.3f}  "
                  f"mean={d['mean_us']:>11.3f}  (derived)", flush=True)

    with open(RES / "wp7_latency.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unit_id", "unit", "median_us", "p95_us", "mean_us",
                    "min_us", "max_us", "std_us", "n", "controllers", "note"])
        w.writerows(rows)

    # ---------------- meta -------------------------------------------------
    try:
        cores_phys = int(subprocess.run(
            ["lscpu", "-p=Core,Socket"], capture_output=True, text=True
        ).stdout.strip().count("\n"))
    except Exception:
        cores_phys = None
    meta = dict(
        script="scripts/wp7_latency.py",
        purpose=("one harness, one statistic, every controller; replaces the "
                 "min-vs-mean comparison used earlier"),
        harness=dict(primitive="time.perf_counter() around each individual call",
                     statistics=["median", "p95", "mean", "n"],
                     budget_s=BUDGET_S, n_min=N_MIN, n_max=N_MAX,
                     warmup_calls=WARMUP,
                     note=("repeat count n is set by a per-unit wall-clock budget "
                           "bounded by [n_min, n_max]; the STATISTIC is identical "
                           "for every unit, which is the point")),
        machine=dict(cpu_model=cpu_model(), cpu_count_logical=os.cpu_count(),
                     cpu_cores_physical=cores_phys,
                     platform=platform.platform(), machine=platform.machine(),
                     loadavg_1_5_15_before=list(load_before),
                     loadavg_1_5_15_after=list(os.getloadavg()),
                     load_note=("timings are wall-clock on a shared desktop; a "
                                "1-minute load average approaching or exceeding "
                                "the logical core count inflates the upper tail "
                                "(p95, mean) of EVERY unit here, which is why the "
                                "median is reported alongside them")),
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      torch=torch.__version__,
                      torch_num_threads=int(torch.get_num_threads())),
        nmpc=dict(n_targets=NMPC_TARGETS, T_max_s=NMPC_TMAX, Ts_s=0.02, N_hor=50,
                  solver="CasADi/IPOPT", runs=nm_runs,
                  total_solves=int(s_nmpc["n"]),
                  measured_here=True,
                  carried_over_mean_us_wp4=75888.70495612141,
                  carried_over_note=("wp4_latency.csv recorded mean=75888.7 us with "
                                     "p95=52 ms in the same row; mean > p95 means "
                                     "that figure was outlier-dominated")),
        sac=dict(available=s_sac is not None, seed=sac_seed,
                 median_return_tail=sac_ret),
        gain_policy=dict(bptt_seed=bseed,
                         sigmoid_ckpt=f"policy_bptt_seed{bseed}.pt",
                         linear_ckpt="pnn_wp0.pt",
                         prior_defect=("an earlier timing table assigned the "
                                       "44.9 us sigmoid figure to the PNN row; "
                                       "the linear PNN is timed separately here")),
        segment=dict(median_segment_s=seg_s, ticks_per_segment=seg_ticks, dt_s=DT),
        gains_note=dict(lyap1=LYAP1, lyap2=LYAP2,
                        note="the tick cost does not depend on the gain values"),
        total_wall_s=round(time.perf_counter() - t_all, 1),
    )
    if s_sac is not None:
        meta["sac"].update(s_sac)
    (LOGS / "wp7_latency_meta.json").write_text(json.dumps(meta, indent=1))

    print(f"\nwrote {RES/'wp7_latency.csv'}")
    print(f"wrote {LOGS/'wp7_latency_meta.json'}")
    print(f"total wall {meta['total_wall_s']} s")
    # like-for-like ratios, both directions, so the artefact is visible
    print("\n-- NMPC / ours, same statistic each time --")
    for k in ("median_us", "p95_us", "mean_us"):
        print(f"   {k:<10} {s_nmpc[k]/s_lyap[k]:10.1f}x")
    print(f"   MIXED (NMPC mean / ours min) = {s_nmpc['mean_us']/s_lyap['min_us']:.1f}x "
          f"<- the earlier style of comparison")


if __name__ == "__main__":
    main()
