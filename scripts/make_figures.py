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

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "passport" / "results"
FIG = ROOT / "passport" / "figures"
FIG.mkdir(exist_ok=True, parents=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", name)


def wp1_curves():
    f = RES / "wp1_sample_efficiency.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = dict(bf="tab:gray", bptt="tab:blue", bo="tab:orange",
                  bo_distilled="tab:red")
    for m, g in df.groupby("method"):
        piv = g.pivot_table(index="rollouts_per_target", values="median_E_ratio",
                            aggfunc=["median", lambda x: np.percentile(x, 25),
                                     lambda x: np.percentile(x, 75)])
        x = piv.index.values
        med = piv.iloc[:, 0].values
        lo, hi = piv.iloc[:, 1].values, piv.iloc[:, 2].values
        ls = "--" if m == "bo_distilled" else "-"
        axes[0].plot(x, med, ls, color=colors.get(m, "k"), label=m)
        axes[0].fill_between(x, lo, hi, color=colors.get(m, "k"), alpha=0.15)
        piv2 = g.pivot_table(index="rollouts_per_target", values="frac_within_2pct",
                             aggfunc="median")
        axes[1].plot(piv2.index.values, piv2.values.ravel(), ls,
                     color=colors.get(m, "k"), label=m)
    axes[0].axhline(1.02, color="k", lw=0.6, ls=":")
    axes[0].set_xscale("log"); axes[0].set_ylim(0.9, 2.5)
    axes[0].set_xlabel("rollouts per target"); axes[0].set_ylabel("median E / E*_bf")
    axes[0].set_title("H1: sample efficiency (10 seeds, IQR band)")
    axes[0].legend()
    axes[1].axhline(1.0, color="k", lw=0.6, ls=":")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("rollouts per target")
    axes[1].set_ylabel("fraction of targets within 2% of brute-force optimum")
    axes[1].legend()
    save(fig, "wp1_sample_efficiency")


def wp2_figs():
    f = RES / "wp2_vseries.npz"
    if not f.exists():
        return
    z = np.load(f)  # self-generated npz of plain float arrays; no pickle needed
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
    for ax, cond in zip(axes, ["halt1cm_pnn", "tol5cm_pnn", "timer_pnn"]):
        t = z[f"{cond}_t"]
        V = z[f"{cond}_V_lyap"]
        for j in range(min(12, V.shape[1])):
            ax.semilogy(t, V[:, j], lw=0.6, alpha=0.7)
        ax.set_title(f"V(t) = (d² + e_θ²)/2 — {cond} (12 of 50 stored runs)")
        ax.set_ylabel("V")
    axes[-1].set_xlabel("t [s]")
    save(fig, "wp2_v_traces")

    fseg = RES / "wp2_segments.csv"
    if fseg.exists():
        df = pd.read_csv(fseg)
        d = df[df.condition == "halt1cm_pnn"].dropna(subset=["T_bound_s"])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        axes[0].scatter(d["T_bound_s"], d["t_arrive_s"], s=4, alpha=0.25)
        lim = [0, max(d["T_bound_s"].max(), d["t_arrive_s"].max()) * 1.05]
        axes[0].plot(lim, lim, "r-", lw=1, label="t = T_bound")
        axes[0].set_xlabel("theoretical bound T_k [s]")
        axes[0].set_ylabel("measured arrival t_k [s]")
        frac = (d["t_arrive_s"] <= d["T_bound_s"]).mean()
        axes[0].set_title(f"halt1cm: arrival vs dwell bound "
                          f"({frac*100:.2f}% within bound)")
        axes[0].legend()
        dfr = pd.read_csv(RES / "wp2_switching.csv")
        summ = dfr.groupby("condition").agg(
            completed=("completed", "mean"), diverged=("diverged", "mean"),
            min_dwell=("min_dwell_s", "min")).reset_index()
        axes[1].bar(summ.condition, 100 * summ.completed, color="tab:blue",
                    label="completed %")
        axes[1].bar(summ.condition, 100 * summ.diverged, color="tab:red",
                    label="diverged %")
        axes[1].set_ylabel("% of 1000 runs")
        axes[1].tick_params(axis="x", rotation=20)
        axes[1].legend()
        axes[1].set_title("H2: completion / divergence by switching policy")
        save(fig, "wp2_dwell_and_outcomes")


def wp3_figs():
    f = RES / "wp3_robustness.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    conds = [c for c in df.condition.unique()]
    ctrls = ["ours_nominal", "ours_dr", "pnn_bf", "lyap1", "lyap2", "pid_table3"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    width = 0.13
    xs = np.arange(len(conds))
    for j, c in enumerate(ctrls):
        med = [df[(df.condition == cd) & (df.controller == c)].e_total_Cl1m.median()
               for cd in conds]
        fail = [df[(df.condition == cd) & (df.controller == c)].failed_3x_rule.mean() * 100
                for cd in conds]
        axes[0].bar(xs + (j - 2.5) * width, med, width, label=c)
        axes[1].bar(xs + (j - 2.5) * width, fail, width, label=c)
    axes[0].set_ylabel("median E (C_l = 1 m)")
    axes[0].set_title("H3: robustness across labeled perturbation conditions (500 MC runs each)")
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].set_ylabel("failure rate [%] (3x rule or divergence)")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(conds, rotation=25, ha="right")
    save(fig, "wp3_robustness")


def wp5_figs():
    f = RES / "wp5_eval_matrix.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    ho = df[(df.scenario == "heldout30") & (df.variant == "clean")]
    if len(ho):
        order = ho.groupby("controller").e_total_Cl1m.median().sort_values()
        fig, ax = plt.subplots(figsize=(9, 4.5))
        data = [ho[ho.controller == c].e_total_Cl1m.values for c in order.index]
        ax.boxplot(data, labels=list(order.index), showfliers=True)
        ax.set_yscale("log")
        ax.set_ylabel("E (C_l = 1 m), held-out region")
        ax.set_title("Generalization: held-out workspace d∈[1.0,1.5], |θ|∈[0.7π,0.9π]")
        ax.tick_params(axis="x", rotation=20)
        save(fig, "wp5_heldout_box")

    flat = RES / "wp4_latency.csv"
    if flat.exists():
        lat = pd.read_csv(flat)
        an = df[(df.scenario == "anchor4") & (df.variant == "clean")]
        med = an.groupby("controller").e_total_Cl1m.median()
        guar = dict(ours=1, ours_dr=1, pnn_bf=1, lyap1=1, lyap2=1,
                    pid_table3=0, pid_tuned=0, nmpc=0, sac=0)
        fig, ax = plt.subplots(figsize=(8, 5))
        for _, row in lat.iterrows():
            c = row.controller
            if c not in med.index:
                continue
            l_us = row.per_tick_us
            ax.scatter(l_us, med[c],
                       marker="o" if guar.get(c) else "x",
                       s=90, label=c)
        ax.set_xscale("log")
        ax.set_xlabel("per-tick decision latency [µs] (this machine)")
        ax.set_ylabel("median E on anchor targets")
        ax.set_title("H4 trade-off: performance vs latency "
                     "(o = formal stability guarantee, x = none)")
        ax.legend(fontsize=8)
        save(fig, "wp5_tradeoff")


if __name__ == "__main__":
    wp1_curves()
    wp2_figs()
    wp3_figs()
    wp5_figs()
