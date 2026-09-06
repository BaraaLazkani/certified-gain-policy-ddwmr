"""WP7-3: re-evaluate the campaigns with the CORRECT certified policy.

An earlier version of this campaign labelled a result row "Ours (BO policy)" that is loaded
from policy_bptt_seed*.pt (wp5_eval.py:78).  The BO policy is evaluated
nowhere, and the BO-distilled checkpoints carry head="linear", so they violate
Assumption 1 and Corollary 1 does not apply to them.

WP7-2 trains the four head x label-source combinations.  This script evaluates
the certified cell (sigmoid + BO labels) on the same scenarios the campaign
reports, alongside the BPTT policy that was actually evaluated before, so the
mislabelling is corrected in the open rather than quietly.

Every controller row is written with the checkpoint filename it came from, so
no reported row can again be traced to the wrong artefact.
"""
import sys, os, csv, json, pathlib
import numpy as np

CODE = pathlib.Path(__file__).resolve().parents[1]   # repository root
sys.path.insert(0, str(CODE / "src")); sys.path.insert(0, str(CODE / "scripts"))
import torch
from ddwmr.sim import run_closedloop, LyapGains, PIDGains
from ddwmr.metrics import anchor_metric
from ddwmr.pnn import GainMLP, predict_gains
from ddwmr.params import LYAP1, LYAP2

PASSPORT = CODE / "passport"
OUT = CODE / "wp7"
RES, MOD = OUT / "results", OUT / "models"


def best_seed_from_ablation(head, labels):
    """Pick the seed by held-out median E, and say so in the output."""
    import pandas as pd
    # the ring file uses the campaign's actual held-out construction
    # (|theta| in [0.7pi,0.9pi]); wp7_ablation.csv used a wrong one and is not
    # comparable.  Note df["head"] not df.head -- the latter is a DataFrame method.
    # Seed chosen on the TRAINING workspace, never on the held-out set:
    # selecting by held-out error would be selecting on the test set.
    ind = pd.read_csv(RES / "wp7_ablation.csv")          # in-distribution columns
    sub = ind[(ind["head"] == head) & (ind["labels"] == labels)]
    if sub.empty:
        raise SystemExit(f"no ablation rows for {head}/{labels}; run WP7-2 first")
    seed = int(sub.sort_values("E_over_Egrid_median").iloc[0]["seed"])
    ring = pd.read_csv(RES / "wp7_ablation_heldout_ring.csv")
    r = ring[(ring["head"] == head) & (ring["labels"] == labels)
             & (ring["seed"] == seed)]
    return seed, (float(r["E"].iloc[0]) if len(r) else float("nan"))


def load(path, head):
    n = GainMLP(head=head); n.load_state_dict(torch.load(path, weights_only=True))
    n.eval(); return n


def heldout_targets(n=30, seed=77):
    """EXACTLY wp5_eval.heldout_targets: a ring OUTSIDE the trained angular
    range. An earlier version of this script drew ang ~ U(-0.9pi, 0.9pi),
    which contains the training range [-0.7pi, 0.7pi], so nothing extrapolated."""
    rng = np.random.default_rng(seed)
    rad = rng.uniform(1.0, 1.5, n)
    sign = rng.choice([-1, 1], n)
    ang = sign * rng.uniform(0.7 * np.pi, 0.9 * np.pi, n)
    return np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)


ANCHOR4 = np.array([[-0.2, -0.5], [-0.3, 0.5], [0.5, -0.5], [0.6, 0.4]])


def main():
    s_bo, e_bo = best_seed_from_ablation("sigmoid", "bo")
    s_bf, e_bf = best_seed_from_ablation("sigmoid", "bf")
    print(f"certified BO policy : sigmoid/bo seed {s_bo} (held-out E {e_bo:.1f})")
    print(f"box-only control    : sigmoid/bf seed {s_bf} (held-out E {e_bf:.1f})")

    ctrls = {
        "ours_bo_sigmoid":  ("policy", load(MOD / f"wp7_sigmoid_bo_seed{s_bo}.pt", "sigmoid"),
                             f"wp7_sigmoid_bo_seed{s_bo}.pt"),
        "ours_bf_sigmoid":  ("policy", load(MOD / f"wp7_sigmoid_bf_seed{s_bf}.pt", "sigmoid"),
                             f"wp7_sigmoid_bf_seed{s_bf}.pt"),
        "ours_bptt":        ("policy", load(PASSPORT / "models" / "policy_bptt_seed9.pt", "sigmoid"),
                             "policy_bptt_seed9.pt  (what was actually evaluated earlier)"),
        "pnn_bf_linear":    ("policy", load(PASSPORT / "models" / "pnn_wp0.pt", "linear"),
                             "pnn_wp0.pt  (anchor's PNN; linear head, NOT certified)"),
        "lyap1":            ("fixed", LYAP1, "fixed gains (0.8, 1.7)"),
        "lyap2":            ("fixed", LYAP2, "fixed gains (0.5, 2.5)"),
        "pid_table3":       ("pid", None, "anchor Table 3 gains"),
    }
    rows = []
    for scen, T in [("anchor4", ANCHOR4), ("heldout30", heldout_targets())]:
        for name, (kind, obj, prov) in ctrls.items():
            if kind == "policy":
                g = predict_gains(obj, T)
                gains = LyapGains(Kp=g[:, 0].copy(), Kth=g[:, 1].copy()); ck = "lyap"
            elif kind == "fixed":
                gains = LyapGains(Kp=np.full(len(T), obj[0]),
                                  Kth=np.full(len(T), obj[1])); ck = "lyap"; g = None
            else:
                gains = PIDGains(); ck = "pid"; g = None
            r = run_closedloop(T, ck, gains, T_max=40.0, rng=np.random.default_rng(777))
            m = anchor_metric(r.e_l_mean, r.theta_end, r.t_end,
                              np.zeros((len(T), 2)), T, C_l=1.0)
            rows.append([scen, name, prov, f"{np.median(m['e_total']):.2f}",
                         f"{np.median(r.e_l_rmse):.5f}", f"{r.arrived.mean():.4f}",
                         int(r.diverged.sum()),
                         f"{np.median(g[:,0]):.3f}" if g is not None else "",
                         f"{np.median(g[:,1]):.3f}" if g is not None else ""])
            print(f"  {scen:<10} {name:<17} E {np.median(m['e_total']):8.1f}  "
                  f"arr {r.arrived.mean()*100:5.1f}%", flush=True)
    with open(RES / "wp7_reeval.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "controller", "checkpoint", "E_median_Cl1m",
                    "xt_rmse_median_m", "arrival", "diverged", "Kp_median", "Kth_median"])
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> wp7_reeval.csv")


if __name__ == "__main__":
    main()
