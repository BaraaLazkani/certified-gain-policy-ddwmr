"""WP0 V0 gate: train PNN on brute-force labels (anchor Table 2 recipe),
reproduce Tables 4-7, compare rankings, render Fig-8 surface comparison.

Outputs:
  passport/results/wp0_verification.csv     (per-table per-controller rows +
                                             per-criterion PASS/FAIL rows)
  passport/results/wp0_tables_raw.csv       (raw per-run numbers)
  passport/models/pnn_wp0.pt
  passport/figures/wp0_fig8_surface.(pdf|png)
  passport/figures/wp0_trajectories.(pdf|png)
  passport/logs/wp0_pnn_training.csv
"""
import sys, pathlib, csv, json
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from ddwmr.sim import run_closedloop, LyapGains, PIDGains
from ddwmr.metrics import anchor_metric
from ddwmr.pnn import train_supervised, predict_gains, GainMLP
from ddwmr.params import LYAP1, LYAP2

SEED = 20260612
RES = ROOT / "passport" / "results"
FIG = ROOT / "passport" / "figures"
MOD = ROOT / "passport" / "models"
LOG = ROOT / "passport" / "logs"
for p in (RES, FIG, MOD, LOG):
    p.mkdir(parents=True, exist_ok=True)

# anchor Tables 4-7: {target: {controller: (e_total, e_line, e_th, e_time)}}
ANCHOR = {
    (-0.2, -0.5): dict(PNN=(1042.7, 12.6, 94.2, 229.6), Lyap1=(1103.7, 15.9, 79.8, 222.4),
                       Lyap2=(1459.3, 13.0, 36.0, 488.8), PID=(1864.7, 68.4, 514.4, 128.4)),
    (-0.3, 0.5): dict(PNN=(1103.8, 12.4, 48.9, 321.9), Lyap1=(1364.6, 25.9, 88.7, 216.2),
                      Lyap2=(1583.5, 17.8, 43.0, 489.6), PID=(2013.6, 75.6, 603.6, 124.3)),
    (0.5, -0.5): dict(PNN=(516.2, 18.0, 19.7, 63.1), Lyap1=(731.7, 22.0, 27.5, 123.7),
                      Lyap2=(973.4, 5.1, 0.1464, 436.6), PID=(1057.8, 48.8, 92.9, 12.0)),
    (0.6, 0.4): dict(PNN=(468.2, 14.8, 6.1, 49.2), Lyap1=(716.7, 18.7, 23.9, 109.2),
                     Lyap2=(967.5, 4.4, 1.5, 429.0), PID=(958.2, 40.0, 34.8, 3.4)),
}
TABLE_NO = {(-0.2, -0.5): 4, (-0.3, 0.5): 5, (0.5, -0.5): 6, (0.6, 0.4): 7}

bf = np.load(RES / "wp0_bruteforce.npz")
n_ws = int(bf["n_workspace"])
ws_targets = bf["targets"][:n_ws]
ws_labels = bf["labels"][:n_ws]
val_targets = bf["val_targets"]
all_targets = bf["targets"]

# ---- 1. PNN training (anchor Table 2 recipe, labels from corrected box) ----
torch.manual_seed(SEED)
net, hist = train_supervised(ws_targets, ws_labels, seed=SEED)
torch.save(net.state_dict(), MOD / "pnn_wp0.pt")
with open(LOG / "wp0_pnn_training.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["epoch", "train_mse", "val_mse"])
    w.writerows(hist)
print(f"PNN trained: final train_mse={hist[-1][1]:.6f} val_mse={hist[-1][2]:.6f}")

# ---- 2. run the four controllers on the four validation targets ----
def run_one(tgt, ctrl, gains):
    r = run_closedloop(np.array([tgt]), ctrl, gains, T_max=40.0)
    return r

rows_raw = []
ours = {}
for ti, tgt in enumerate(map(tuple, val_targets)):
    # C_l for e_total: per-target brute-force max e_l (the anchor's apparent
    # e_total convention, reverse-engineered: Lyap1/Lyap2 e_totals in Tables
    # 4-7 are reproduced by C_l = max_j e_l_j; components use C_l = 1 m)
    C_l_bf = float(bf["C_l_bf"][n_ws + ti])
    runs = {}
    runs["Lyap1"] = run_one(tgt, "lyap", LyapGains(Kp=np.array([LYAP1[0]]), Kth=np.array([LYAP1[1]])))
    runs["Lyap2"] = run_one(tgt, "lyap", LyapGains(Kp=np.array([LYAP2[0]]), Kth=np.array([LYAP2[1]])))
    runs["PID"] = run_one(tgt, "pid", PIDGains())
    g = predict_gains(net, [tgt])[0]
    runs["PNN"] = run_one(tgt, "lyap", LyapGains(Kp=np.array([g[0]]), Kth=np.array([g[1]])))
    ours[tgt] = {}
    for name, r in runs.items():
        m1 = anchor_metric(r.e_l_mean, r.theta_end, r.t_end, [[0, 0]], [tgt], C_l=1.0)
        mbf = anchor_metric(r.e_l_mean, r.theta_end, r.t_end, [[0, 0]], [tgt], C_l=C_l_bf)
        ours[tgt][name] = dict(e_total=float(mbf["e_total"][0]), e_line=float(m1["e_line"][0]),
                               e_th=float(m1["e_th"][0]), e_time=float(m1["e_time"][0]),
                               t_end=float(r.t_end[0]), arrived=bool(r.arrived[0]),
                               gains=(float(g[0]), float(g[1])) if name == "PNN" else None)
        rows_raw.append([TABLE_NO[tgt], f"({tgt[0]},{tgt[1]})", name,
                         ours[tgt][name]["e_total"], ours[tgt][name]["e_line"],
                         ours[tgt][name]["e_th"], ours[tgt][name]["e_time"],
                         float(r.t_end[0]), float(r.e_l_mean[0]), float(r.theta_end[0]),
                         bool(r.arrived[0]), bool(r.diverged[0]), C_l_bf])

with open(RES / "wp0_tables_raw.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["anchor_table", "target", "controller", "e_total_Clbf", "e_line_Cl1m",
                "e_th", "e_time", "t_end_s", "e_l_raw_m", "theta_end_rad",
                "arrived", "diverged", "C_l_bf_m"])
    w.writerows(rows_raw)

# ---- 3. ranking comparison ----
def ranks(d, key):
    names = sorted(d, key=lambda n: d[n][key] if isinstance(d[n], dict) else d[n])
    return {n: i + 1 for i, n in enumerate(names)}

def anchor_ranks(tgt, col_idx):
    vals = {n: ANCHOR[tgt][n][col_idx] for n in ANCHOR[tgt]}
    names = sorted(vals, key=vals.get)
    return {n: i + 1 for i, n in enumerate(names)}

crit_rows = []
agree_cells = 0; total_cells = 0; kendall_sum = 0.0
from itertools import combinations
for tgt in map(tuple, val_targets):
    for ci, col in enumerate(["e_total", "e_line", "e_th", "e_time"]):
        ar = anchor_ranks(tgt, ci)
        orr = ranks(ours[tgt], col)
        conc = disc = 0
        for a, b in combinations(ar, 2):
            s_a = np.sign(ar[a] - ar[b]); s_o = np.sign(orr[a] - orr[b])
            conc += (s_a == s_o); disc += (s_a != s_o)
        tau = (conc - disc) / (conc + disc)
        kendall_sum += tau
        match = all(ar[n] == orr[n] for n in ar)
        agree_cells += match; total_cells += 1
        crit_rows.append([f"T{TABLE_NO[tgt]}", col,
                          ">".join(sorted(ar, key=ar.get)),
                          ">".join(sorted(orr, key=orr.get)),
                          f"{tau:.2f}", "MATCH" if match else "DIFFER"])

# blueprint-named gate criteria
g1 = all(ranks(ours[t], "e_total")["PNN"] == 1 for t in map(tuple, val_targets))
g2 = ours[(0.6, 0.4)]["Lyap2"]["e_line"] == min(v["e_line"] for v in ours[(0.6, 0.4)].values())
g3 = ranks(ours[(0.6, 0.4)], "e_time")["PID"] == 1
mean_tau = kendall_sum / total_cells

# ---- 4. Fig 8 surface comparison ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig8_el = bf["fig8_anchorbox_e_l"]; fig8_t = bf["fig8_anchorbox_t_end"]
fig8_th = bf["fig8_anchorbox_th_end"]; G8 = bf["fig8_gains"]
kp8 = bf["kp_grid_fig8"]; kth = bf["kth_grid"]
tgt8 = (0.45, 0.77)
C8 = float(fig8_el.max())
m8 = anchor_metric(fig8_el, fig8_th, fig8_t, np.zeros((len(G8), 2)),
                   np.tile(tgt8, (len(G8), 1)), C_l=C8)
E8 = m8["e_total"].reshape(50, 50)
fig = plt.figure(figsize=(11, 4.5))
ax = fig.add_subplot(121, projection="3d")
KPm, KTm = np.meshgrid(kp8, kth, indexing="ij")
ax.plot_surface(KPm, KTm, E8, cmap="jet", linewidth=0)
ax.set_xlabel("Kp"); ax.set_ylabel("Kth"); ax.set_zlabel("error metric")
ax.set_title(f"Reproduced error surface, target {tgt8}\n(anchor box Kp in [0,1] — Fig. 8 shape check)")
ax.view_init(elev=25, azim=-130)
ax2 = fig.add_subplot(122)
cs = ax2.contourf(KPm, KTm, E8, levels=30, cmap="jet")
plt.colorbar(cs, ax=ax2)
bi = np.unravel_index(E8.argmin(), E8.shape)
ax2.plot(kp8[bi[0]], kth[bi[1]], "w*", ms=14, mec="k")
ax2.set_xlabel("Kp"); ax2.set_ylabel("Kth"); ax2.set_title("contour + argmin")
fig.tight_layout()
fig.savefig(FIG / "wp0_fig8_surface.pdf"); fig.savefig(FIG / "wp0_fig8_surface.png", dpi=300)
plt.close(fig)

# surface descriptors for the gate record
plateau = E8[:5].mean()      # low-Kp wall
valley = E8[25:, 20:].min()
mono = E8.mean(axis=1)       # vs Kp
fig8_shape_ok = (plateau > 3 * valley) and (mono[0] > mono[-1]) and \
    (bi[0] > 25)             # minimum at high Kp, matching anchor Fig 8

# ---- 5. trajectory overlays (like anchor Figs. 15-18) ----
fig, axes = plt.subplots(2, 2, figsize=(10, 9))
for ax, tgt in zip(axes.ravel(), map(tuple, val_targets)):
    for name, ctrl, gains, colr in [
            ("Lyap1", "lyap", LyapGains(Kp=np.array([LYAP1[0]]), Kth=np.array([LYAP1[1]])), "tab:blue"),
            ("Lyap2", "lyap", LyapGains(Kp=np.array([LYAP2[0]]), Kth=np.array([LYAP2[1]])), "tab:red"),
            ("PID", "pid", PIDGains(), "tab:orange")]:
        r = run_closedloop(np.array([tgt]), ctrl, gains, T_max=40.0,
                           record_series=1, series_stride=20)
        ax.plot(r.series["x"][:, 0], r.series["y"][:, 0], color=colr, label=name, lw=1.2)
    g = predict_gains(net, [tgt])[0]
    r = run_closedloop(np.array([tgt]), "lyap",
                       LyapGains(Kp=np.array([g[0]]), Kth=np.array([g[1]])),
                       T_max=40.0, record_series=1, series_stride=20)
    ax.plot(r.series["x"][:, 0], r.series["y"][:, 0], color="black", label="PNN", lw=1.4)
    ax.plot(*tgt, "r o", ms=8, mfc="none")
    ax.set_title(f"target {tgt}"); ax.set_aspect("equal"); ax.legend(fontsize=8)
fig.suptitle("WP0: trajectories across controllers (cf. anchor Figs. 15-18)")
fig.tight_layout()
fig.savefig(FIG / "wp0_trajectories.pdf"); fig.savefig(FIG / "wp0_trajectories.png", dpi=300)
plt.close(fig)

# ---- 6. verdict + CSV ----
with open(RES / "wp0_verification.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["record", "table", "metric", "anchor_value", "our_value",
                "anchor_rank_order", "our_rank_order", "kendall_tau", "status"])
    for tgt in map(tuple, val_targets):
        for ci, col in enumerate(["e_total", "e_line", "e_th", "e_time"]):
            for n in ["PNN", "Lyap1", "Lyap2", "PID"]:
                w.writerow(["value", f"T{TABLE_NO[tgt]}", f"{col}/{n}",
                            ANCHOR[tgt][n][ci], f"{ours[tgt][n][col]:.1f}", "", "", "", ""])
    for r_ in crit_rows:
        w.writerow(["ranking", r_[0], r_[1], "", "", r_[2], r_[3], r_[4], r_[5]])
    w.writerow(["criterion", "ALL", "PNN best e_total in all 4 tables", "", "",
                "", "", "", "PASS" if g1 else "FAIL"])
    w.writerow(["criterion", "T7", "Lyap2 best e_line", "", "", "", "", "",
                "PASS" if g2 else "FAIL"])
    w.writerow(["criterion", "T7", "PID best e_time", "", "", "", "", "",
                "PASS" if g3 else "FAIL"])
    w.writerow(["criterion", "ALL", "Fig8 surface shape (low-Kp plateau >> high-Kp valley, argmin at high Kp)",
                "", f"plateau={plateau:.0f} valley={valley:.0f}", "", "", "",
                "PASS" if fig8_shape_ok else "FAIL"])
    w.writerow(["criterion", "ALL", f"mean Kendall tau across 16 ranking cells",
                "", f"{mean_tau:.3f}", "", "", "",
                "PASS" if mean_tau > 0.6 else "FAIL"])

print(json.dumps(dict(
    pnn_e_total_best_all4=bool(g1), lyap2_best_eline_T7=bool(g2),
    pid_best_etime_T7=bool(g3), fig8_shape=bool(fig8_shape_ok),
    mean_kendall_tau=round(mean_tau, 3),
    exact_rank_cells=f"{agree_cells}/{total_cells}"), indent=1))
for r_ in crit_rows:
    print(r_)
print("\nPNN gains chosen:", {str(t): ours[tuple(t)]["PNN"]["gains"] for t in val_targets})
