"""WP5 statistics: Wilcoxon signed-rank, ours vs each baseline, paired by
target (heldout30 clean / anchor4+heldout30 noisy), alpha=0.05 with Holm
correction across baselines within each (cell, metric) family.
Output: results/wp5_stats.csv
"""
import sys, csv, pathlib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "passport" / "results"

METRICS = ["e_total_Cl1m", "cross_track_rmse_m", "e_th", "e_time", "energy_J",
           "peak_V", "t_end_s"]


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    mx = 0.0
    for rank, i in enumerate(order):
        mx = max(mx, (m - rank) * pvals[i])
        adj[i] = min(1.0, mx)
    return adj


def main():
    df = pd.read_csv(RES / "wp5_eval_matrix.csv")
    rows = []
    cells = [("heldout30", ["clean"]), ("anchor4", None), ("heldout30", None)]
    seen = set()
    for scenario in ["heldout30", "anchor4"]:
        for variant_sel, label in [(["clean"], "clean"), (None, "noisy_all")]:
            d = df[df.scenario == scenario]
            if variant_sel is not None:
                d = d[d.variant.isin(variant_sel)]
            else:
                d = d[d.variant != "clean"]
            if not len(d):
                continue
            key = (scenario, label)
            if key in seen:
                continue
            seen.add(key)
            ours = d[d.controller == "ours"]
            if not len(ours):
                continue
            baselines = [c for c in d.controller.unique() if c != "ours"]
            for metric in METRICS:
                pvals, items = [], []
                for b in baselines:
                    join = ours.merge(d[d.controller == b],
                                      on=["scenario", "variant", "idx"],
                                      suffixes=("_o", "_b"))
                    a = pd.to_numeric(join[f"{metric}_o"], errors="coerce")
                    bb = pd.to_numeric(join[f"{metric}_b"], errors="coerce")
                    ok = a.notna() & bb.notna()
                    a, bb = a[ok].values, bb[ok].values
                    if len(a) < 6 or np.allclose(a, bb):
                        continue
                    try:
                        stat, p = wilcoxon(a, bb)
                    except ValueError:
                        continue
                    pvals.append(p)
                    items.append([scenario, label, metric, b, len(a),
                                  float(np.median(a)), float(np.median(bb)),
                                  float(np.median(a - bb)), p])
                if pvals:
                    adj = holm(np.array(pvals))
                    for it, pa in zip(items, adj):
                        rows.append(it + [float(pa), int(pa < 0.05)])
    with open(RES / "wp5_stats.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "variant", "metric", "baseline", "n_pairs",
                    "median_ours", "median_baseline", "median_paired_diff",
                    "p_raw", "p_holm", "significant_05"])
        w.writerows(rows)
    print(f"wrote {len(rows)} stat rows")


if __name__ == "__main__":
    main()
