"""Regenerate every campaign figure from passport/results/* only.

Each figure function is guarded by artifact existence, so this script can be
re-run at any stage; outputs PDF + 300-dpi PNG into passport/figures/.
(WP0's two figures are produced by scripts/wp0_verify.py deterministically
from wp0_bruteforce.npz + the saved PNN; noted in README.)
"""
import pathlib, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- print legibility (IJCAS two-column: 80 mm column, 166 mm text width) ----
# Figures are rendered at their FINAL printed width so LaTeX applies no
# down-scaling; type is then genuinely 8-9 pt on the page.
COL_IN, TEXT_IN = 3.35, 6.9
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "lines.linewidth": 1.8, "lines.markersize": 4,
    "axes.linewidth": 0.8, "grid.linewidth": 0.5,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "legend.frameon": False, "figure.constrained_layout.use": True,
    "savefig.bbox": None,   # honour figsize exactly so LaTeX never rescales
})

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "passport" / "results"
FIG = ROOT / "passport" / "figures"
FIG.mkdir(exist_ok=True, parents=True)


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", name)


def wp1_curves():
    f = RES / "wp1_sample_efficiency.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    fig, axes = plt.subplots(2, 1, figsize=(COL_IN, 4.0))
    style = {"bf": ("tab:grey", "-", "grid"), "bptt": ("tab:blue", "-", "BPTT"),
             "bo": ("tab:orange", "-", "BO"),
             "bo_distilled": ("tab:red", "--", "BO distilled")}
    for m, g in df.groupby("method"):
        col, ls, lab = style.get(m, ("k", "-", m))
        piv = g.groupby("rollouts_per_target")["median_E_ratio"].median()
        axes[0].plot(piv.index.values, piv.values, ls, color=col, label=lab, lw=1.6)
        if "frac_within_2pct" in g.columns:
            p2 = g.groupby("rollouts_per_target")["frac_within_2pct"].median()
            axes[1].plot(p2.index.values, p2.values, ls, color=col, lw=1.6)
    axes[0].axhline(1.0, color="k", lw=0.6, ls=":")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_ylabel("$E/E_\\mathrm{grid}$")
    axes[0].set_xlabel("")
    axes[0].tick_params(labelbottom=False)
    axes[0].legend(ncol=4, fontsize=6.5, loc="upper right", frameon=False)
    axes[1].set_xscale("log")
    axes[1].set_ylabel("frac. within 2%")
    axes[1].set_xlabel("rollouts per target")
    for a in axes:
        a.grid(alpha=0.3, which="both", lw=0.4)
    save(fig, "wp1_sample_efficiency")


def wp2_figs():
    f = RES / "wp2_vseries.npz"
    if f.exists():
        z = np.load(f)
        t = z["halt1cm_pnn_t"]; dv = z["halt1cm_pnn_dv"]; d = z["halt1cm_pnn_d"]
        fig, axes = plt.subplots(2, 1, figsize=(COL_IN, 3.6), sharex=True)
        n = min(12, dv.shape[1])
        for i_ in range(n):
            axes[0].plot(t, np.abs(dv[:, i_]), lw=0.6, alpha=0.55)
        axes[0].set_ylabel(r"$|\delta_v|$ [m/s]")
        axes[0].set_title("inner-loop velocity error, %d runs" % n)
        axes[0].grid(alpha=0.3, lw=0.4)
        near = np.abs(dv[:, :n]).copy()
        near[d[:, :n] >= 0.05] = np.nan          # terminal approach only
        for i_ in range(n):
            axes[1].plot(t, near[:, i_], lw=0.7, alpha=0.7)
        axes[1].axhline(2.5e-4, color="r", ls="--", lw=1.2,
                        label=r"(C3) threshold $2.5\times10^{-4}$")
        axes[1].set_yscale("log")
        axes[1].set_ylabel(r"$|\delta_v|$, $d<5$ cm")
        axes[1].set_xlabel("t [s]")
        axes[1].legend(fontsize=6.5, loc="upper right")
        axes[1].grid(alpha=0.3, which="both", lw=0.4)
        save(fig, "wp2_v_traces")

    fseg = RES / "wp2_segments.csv"
    if fseg.exists():
        df = pd.read_csv(fseg)
        d = df[df.condition == "halt1cm_pnn"].dropna(subset=["T_bound_s"])
        fig, axes = plt.subplots(2, 1, figsize=(COL_IN, 4.4))
        axes[0].scatter(d["T_bound_s"], d["t_arrive_s"], s=3, alpha=0.25,
                        label="segments (n=%d)" % len(d))
        lim = [0, max(d["T_bound_s"].max(), d["t_arrive_s"].max()) * 1.05]
        axes[0].plot(lim, lim, "r-", lw=1.4, label="$t_k = T_k$ (bound)")
        axes[0].set_xlabel("theoretical bound $T_k$ [s]")
        axes[0].set_ylabel("measured arrival $t_k$ [s]")
        frac = (d["t_arrive_s"] <= d["T_bound_s"]).mean()
        axes[0].set_title("%.2f%% within bound; min slack %.2f s"
                          % (frac * 100, (d["T_bound_s"] - d["t_arrive_s"]).min()))
        axes[0].legend(loc="upper left")
        dfr = pd.read_csv(RES / "wp2_switching.csv")
        summ = dfr.groupby("condition").agg(
            completed=("completed", "mean"), diverged=("diverged", "mean")).reset_index()
        lbl = {"halt1cm_pnn": "1 cm halt", "tol5cm_pnn": "5 cm switch",
               "timer_pnn": "2 s timer", "halt1cm_fixedL1": "fixed gain"}
        names = [lbl.get(c, c) for c in summ.condition]
        axes[1].bar(names, 100 * summ.completed, color="tab:blue", label="completed")
        axes[1].bar(names, 100 * summ.diverged, color="tab:red", label="diverged")
        axes[1].set_ylabel("% of 1000 runs"); axes[1].set_ylim(0, 108)
        axes[1].tick_params(axis="x", rotation=15)
        axes[1].legend(ncol=2, loc="lower right")
        axes[1].set_title("completion / divergence by switching rule")
        save(fig, "wp2_dwell_and_outcomes")


def wp3_figs():
    f = RES / "wp3_robustness.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    conds = list(df.condition.unique())
    ctrls = ["ours_nominal", "ours_dr", "pnn_bf", "lyap1", "lyap2", "pid_table3"]
    nice = {"ours_nominal": "ours (BPTT)", "ours_dr": "ours (DR)",
            "pnn_bf": "PNN [1]", "lyap1": "Lyap 1", "lyap2": "Lyap 2",
            "pid_table3": "PID [1]"}
    fig, axes = plt.subplots(2, 1, figsize=(COL_IN, 4.6), sharex=True)
    width = 0.13
    xs = np.arange(len(conds))
    for j, c in enumerate(ctrls):
        med = [df[(df.condition == cd) & (df.controller == c)].e_total_Cl1m.median()
               for cd in conds]
        fail = [df[(df.condition == cd) & (df.controller == c)].failed_3x_rule.mean() * 100
                for cd in conds]
        axes[0].bar(xs + (j - 2.5) * width, med, width, label=nice.get(c, c))
        axes[1].bar(xs + (j - 2.5) * width, fail, width)
    axes[0].set_ylabel("median $E$  ($C_l$ = 1 m)")
    axes[0].set_title("500 Monte-Carlo runs per condition")
    axes[0].legend(ncol=3, fontsize=6.5)
    axes[1].set_ylabel("timing-spec violation [%]")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(conds, rotation=30, ha="right", fontsize=6.5)
    save(fig, "wp3_robustness")


def wp5_figs():
    f = RES / "wp5_eval_matrix.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    ho = df[(df.scenario == "heldout30") & (df.variant == "clean")]
    if len(ho):
        order = ho.groupby("controller").e_total_Cl1m.median().sort_values()
        data = [ho[ho.controller == c].e_total_Cl1m.values for c in order.index]
        fig, ax = plt.subplots(figsize=(COL_IN, 2.8))
        nice = {"ours": "ours (BPTT)", "ours_dr": "ours (DR)", "pnn_bf": "PNN [1]",
                "lyap1": "Lyap 1", "lyap2": "Lyap 2", "pid_table3": "PID [1]",
                "pid_tuned": "PID tuned", "nmpc": "NMPC", "sac": "SAC"}
        ax.boxplot(data, tick_labels=[nice.get(o, o) for o in order.index],
                   showfliers=False, widths=0.55)
        ax.set_yscale("log")
        ax.set_ylabel("$E$ ($C_l$ = 1 m)")
        ax.set_title("held-out targets")
        ax.tick_params(axis="x", rotation=40, labelsize=6.5)
        for lab in ax.get_xticklabels():
            lab.set_ha("right")
        ax.grid(alpha=0.3, axis="y", which="both", lw=0.4)
        save(fig, "wp5_heldout_box")

    WP7 = ROOT / "wp7" / "results"
    lat_f, ree_f = WP7 / "wp7_latency.csv", WP7 / "wp7_reeval.csv"
    if lat_f.exists() and ree_f.exists():
        lat = pd.read_csv(lat_f).set_index("unit_id")["median_us"]
        ree = pd.read_csv(ree_f)
        ree = ree[ree.scenario == "heldout30"].set_index("controller")["E_median_Cl1m"]
        ev = pd.read_csv(RES / "wp5_eval_matrix.csv")
        ev = ev[(ev.scenario == "heldout30") & (ev.variant == "clean")]
        ev = ev.groupby("controller")["e_total_Cl1m"].median()
        LY, PID = lat["lyap_family_full_tick"], lat["pid_full_tick"]
        pts = [  # (label, latency us, held-out E, certified?)
            ("ours (BPTT)",  LY, ree.get("ours_bptt"),        True),
            ("ours (BO)",    LY, ree.get("ours_bo_sigmoid"),  True),
            ("PNN [1]",      LY, ree.get("pnn_bf_linear"),    True),
            ("Lyap 1",       LY, ree.get("lyap1"),            True),
            ("Lyap 2",       LY, ree.get("lyap2"),            True),
            ("PID [1]",      PID, ree.get("pid_table3"),      False),
            ("PID tuned",    PID, ev.get("pid_tuned"),        False),
            ("SAC",          lat["sac_predict"], ev.get("sac"),   False),
            ("NMPC",         lat["nmpc_solve"],  ev.get("nmpc"),  False),
        ]
        fig, ax = plt.subplots(figsize=(COL_IN, 3.2))
        good = [q for q in pts if q[2] is not None and np.isfinite(q[2])]
        for name, x, y, cert in good:
            ax.scatter(x, y, s=42, marker="o" if cert else "^",
                       facecolor="tab:blue" if cert else "none",
                       edgecolor="tab:blue" if cert else "tab:red", zorder=3)
        # the certified controllers share one latency, so their labels would
        # stack; fan them out to the right in y order instead.
        cluster = sorted([q for q in good if q[3]], key=lambda q: q[2])
        for i, (name, x, y, _) in enumerate(cluster):
            ax.annotate(name, (x, y), textcoords="offset points",
                        xytext=(10, -3 + 11 * (i - (len(cluster) - 1) / 2)),
                        fontsize=6.2, va="center",
                        arrowprops=dict(arrowstyle="-", lw=0.4, color="grey",
                                        shrinkA=0, shrinkB=2))
        for name, x, y, cert in good:
            if not cert:
                # label to the right near the left axis, otherwise to the left
                right = x < 1e3
                ax.annotate(name, (x, y), textcoords="offset points",
                            xytext=(7 if right else -6, 7), fontsize=6.2,
                            ha="left" if right else "right")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(5, 1.2e5); ax.margins(y=0.18)
        ax.set_xlabel("per-tick control computation [µs], median")
        ax.set_ylabel("held-out median $E$")
        ax.set_title("filled = carries the certificate")
        ax.grid(alpha=0.3, which="both", lw=0.4)
        save(fig, "wp5_tradeoff")


if __name__ == "__main__":
    wp1_curves()
    wp2_figs()
    wp3_figs()
    wp5_figs()

def wp7_figs():
    """Figures for the WP7 correction campaign: delay by feedback path, and
    the 2x2 head x label-source ablation."""
    WP7 = ROOT / "wp7" / "results"

    # --- delay, each feedback path separately (R1.11, R3.1, R3.2) ---
    f = WP7 / "wp7_delay_summary.csv"
    if f.exists():
        d = pd.read_csv(f)
        d = d[d.trigger_mode == "measured"]
        nice = {"ours_bptt": "ours (BPTT)", "ours_dr": "ours (DR)",
                "pnn_bf": "PNN [1]", "lyap1": "Lyap 1", "lyap2": "Lyap 2",
                "pid_table3": "PID [1]"}
        fig, axes = plt.subplots(1, 3, figsize=(TEXT_IN, 2.5), sharey=True)
        for ax, scope, title in zip(axes, ["pose", "wheels", "all"],
                                    ["pose path only", "encoder path only",
                                     "both simultaneously"]):
            sub = d[d.scope == scope]
            for c in nice:
                g = sub[sub.controller == c].sort_values("delay_ms")
                if len(g):
                    ax.plot(g.delay_ms, g.arrival_pct, marker="o", ms=3,
                            lw=1.4, label=nice[c])
            ax.set_title(title); ax.set_xlabel(r"delay $\tau$ [ms]")
            ax.set_ylim(-4, 108); ax.grid(alpha=0.3, lw=0.4)
        axes[0].set_ylabel("arrival rate [%]")
        axes[2].legend(fontsize=6, ncol=2, loc="lower left")
        save(fig, "wp7_delay")

    # --- 2x2 ablation on the held-out ring ---
    f = WP7 / "wp7_ablation_heldout_ring.csv"
    if f.exists():
        a = pd.read_csv(f)
        fig, ax = plt.subplots(figsize=(COL_IN, 2.5))
        cells = [("linear", "bf"), ("linear", "bo"),
                 ("sigmoid", "bf"), ("sigmoid", "bo")]
        data = [a[(a["head"] == h) & (a["labels"] == l)]["E"].values for h, l in cells]
        bp = ax.boxplot(data, tick_labels=["lin\nBF", "lin\nBO",
                                           "sig\nBF", "sig\nBO"],
                        showfliers=False, patch_artist=True, widths=0.55)
        for patch, (h, l) in zip(bp["boxes"], cells):
            patch.set_facecolor("tab:blue" if h == "sigmoid" else "lightgrey")
            patch.set_alpha(0.65)
        ax.set_ylabel("held-out median $E$")
        ax.set_title("10 seeds per cell; blue = certified output box")
        ax.grid(alpha=0.3, axis="y", lw=0.4)
        save(fig, "wp7_ablation")

wp7_figs()
