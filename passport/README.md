# Materials Passport — Gain-policy learning with switching-stability guarantees for waypoint-halt DDWMR navigation

**Produced by:** experiment-agent (run mode) · 2026-06-12
**Blueprint:** `../../research/methodology_blueprint.md` + binding `../SPEC_AMENDMENTS.md`
**Environment:** `environment/environment.json` (Python 3.12.3, i5-13450HX 16-core, Linux 6.17),
lockfile `environment/requirements_lock.txt`. Code repo: this tree (`git log` for hashes).
**Anchor paper:** Deeb, Alsaleh & Hatem (in press, IJCAS) — all Eq./Table numbers refer to it.

---

## V0 GATE VERDICT: **PASS WITH DOCUMENTED DEVIATION**

Evidence (`results/wp0_verification.csv`, full forensics `logs/wp0_pid_forensics.md`):

| Check | Result |
|---|---|
| LQR gains from Q,R (Eqs. 46-47) reproduce published K1,K2 (Eqs. 48-49) | PASS — max diff 4.6e-5 |
| Eqs. 12-15 coefficients vs first-principles M^-1 | PASS — identical |
| RK4(1 ms) vs adaptive RK45(rtol 1e-8) | PASS — 4e-11 state diff |
| Lyap1/Lyap2 component values, Tables 4-7 | PASS — e.g. e_line 18.66/18.7, 4.41/4.4; e_th 23.9/23.9; e_time 109.1/109.2, 428.8/429 |
| PNN best e_total in all four tables | PASS |
| Lyap2 best e_line at (0.6,0.4) | PASS |
| Fig. 8 error-surface shape (anchor box) | PASS — low-Kp plateau >> high-Kp valley, argmin high-Kp |
| PID path shape (e_line, e_th) all 4 targets | PASS — 38.9/40, 67.4/68.4, 74.5/75.6, 48.5/48.8 |
| **PID arrival time** | **DEVIATION** — anchor PID ~2.2x faster than Table-3 gains produce under any standard PID; anchor's own e_total column is internally inconsistent with their components (Lyap rows reconcile with C_l=0.062 m; PID rows with no C_l). All e_time rank differences trace to this one cause. |

Reverse-engineered anchor constants (validated to 0.1%): **K_t = 3.5 s/m**;
e_total normalization **C_l = per-target brute-force max e_l**; component columns use C_l = 1 m.

---

## Hypothesis verdicts

**H1 (sample efficiency, `results/wp1_sample_efficiency.csv`, `wp1_final.csv`) — PARTIAL.**
Brute-force yardstick: 515,000 rollouts, 907 s wall (`logs/wp0_bruteforce_log.txt`).
- Contextual BO: **median E ratio 1.004, 76.8% of targets within 2% of the brute-force
  optimum, at 25 rollouts/target = 100x fewer. H1 SUPPORTED.**
- BPTT/diff-sim: converges by ~75 iters but **plateaus at ratio 1.125** (5% within 2%)
  at 150 rollouts/target (16.7x). **H1 NOT met at the 2% bar.** Cause: the anchor metric is
  only piecewise-smooth (arrival-halt event); the straight-through surrogate's bias near
  the optimum (~10-20% at aggressive gains) bounds achievable accuracy.
- BO labels distilled into the anchor MLP: ratio 1.062 (distillation gap, mirrors anchor's PNN).

**H2 (switching stability, `results/wp2_switching.csv`, `wp2_segments.csv`, `wp2_vseries.npz`) — SUPPORTED.**
4,000 sequence runs (1,000 x {1 cm halt, 5 cm switch, 2 s timer, fixed-gain control}):
**zero divergences; 100% completion.** Theory-verification data: **12,509/12,509 segments
arrive within the dwell bound T_k = ln4/Kth + (2/Kp) ln((L_k+eps)/eps); minimum slack 3.01 s**
(bound never tight). Min observed dwell 2.30 s (halt), 1.09 s (5 cm). Full V(t), delta_v(t),
delta_w(t) at 100 Hz for runs 0-49 per condition; sup/RMS for all runs. The dwell-violating
policies did not destabilize the system — the guarantee boundary was not reachable in this
gain box (reported as-is; supports conservativeness of the bound).

**H3 (robustness/DR, `results/wp3_robustness.csv`, 39,000 runs) — SPLIT.**
- Graceful degradation: **SUPPORTED** — nominal policy +9.9% median E at ±20% (bound: 30%),
  +20.3% at ±30%; DR policy flatter still (+3.6%, +6.1%).
- Strict dominance of DR: **REJECTED** — DR pays ~190 E clean-performance premium never
  recovered within ±30% (DR better on ≤10% of paired runs). Interpretation: the
  Lyapunov+LQR structure is already intrinsically robust; DR conservatism buys little here.
- **Zero divergences in all 39,000 runs.** Failure rules (both reported): 3x-desired-time
  rule = timing-spec violation rate (fails slow controllers even unperturbed — see
  `failed_3x_rule` column); 3x-own-nominal rule (post hoc from per-run t_end) = **0.0%
  everywhere**.

**H4 (fair baselines, `results/wp5_eval_matrix.csv`, `wp5_paths.csv`, `wp4_latency.csv`, `wp5_stats.csv`) — SUPPORTED (trade-off form).**
- NMPC (CasADi/IPOPT, QSS internal model): **best tracking everywhere** — E 86 (anchor4) /
  137 (held-out), 1-4 mm path RMSE — at **~76 ms/tick measured (p95 52 ms)** vs **1.8 µs/tick
  + 46 µs/switch for our policy (~43,000x)**, with 12 V saturation and ~7x energy on paths.
  Reported unfiltered per the honesty rule.
- Tuned PID (same 5,000-rollout BO budget; meanE 670 vs anchor Table-3 PID 2,322 —
  removes the anchor's weak-baseline flaw): competitive in-distribution (E 242) but
  **catastrophically brittle**: held-out E 21,422 with 20% arrival, and **diverges on all
  four waypoint paths** (only controller to diverge anywhere in the campaign).
- SAC (500k steps x 5 seeds, budgets in `logs/wp4_sac_budget_seed*.json`): never reaches
  the 1 cm tolerance; E 2,517-2,870. Budget-limited result, reported as such.
- Our learned-gain controller: best generalization among guarantee-carrying real-time
  controllers (held-out E 288, 100% arrival; sigmoid gain box prevents the extrapolation
  blow-up that hits the linear-head PNN: 633). All Lyapunov-family controllers complete
  all paths with zero divergences.
- Stats: 155 Wilcoxon/Holm rows in `results/wp5_stats.csv`.

---

## Artifact map

| WP | Results | Configs | Models | Figures | Logs |
|---|---|---|---|---|---|
| WP0 | wp0_verification.csv, wp0_tables_raw.csv, wp0_bruteforce.npz, wp1_bruteforce_labels.csv | wp0_*.yaml | pnn_wp0.pt | wp0_fig8_surface, wp0_trajectories | wp0_bruteforce_log.txt, wp0_pnn_training.csv, wp0_pid_forensics.md |
| WP1 | wp1_sample_efficiency.csv, wp1_final.csv | wp1_train.yaml | policy_bptt_seed0-9.pt, pnn_bo_distilled_seed0-9.pt | wp1_sample_efficiency | wp1_budgets.json |
| WP2 | wp2_switching.csv, wp2_segments.csv, wp2_vseries.npz | wp2_switching.yaml | (uses pnn_wp0) | wp2_v_traces, wp2_dwell_and_outcomes | wp2_budgets.json |
| WP3 | wp3_robustness.csv | wp3_robustness.yaml | policy_dr_seed0-2.pt | wp3_robustness | wp3_dr_training_seed*.csv, wp3_budgets.json |
| WP4 | wp4_pid_tuned.csv, wp4_latency.csv | wp4_baselines.yaml | pid_tuned.json, sac_seed0-4.zip | (in wp5_tradeoff) | wp4_sac_curve_seed*.csv, wp4_sac_budget_seed*.json |
| WP5 | wp5_eval_matrix.csv, wp5_paths.csv, wp5_path_segments.csv, wp5_stats.csv | wp5_eval.yaml | — | wp5_heldout_box, wp5_tradeoff | wp5_meta.json |

Reproduction: every figure regenerates via `python scripts/make_figures.py` from
`results/*` only (WP0's two figures via `scripts/wp0_verify.py`, deterministic from the
saved npz + model). One YAML per experiment ID with seeds. No silent drops: failures and
divergences appear as rows (`arrived`, `diverged`, `failed_*` columns).

## Documented deviations from the blueprint

1. **Gain box** Kp ∈ [0.1, 1.0] per SPEC_AMENDMENTS #1, applied from the start (no reruns
   needed). Fig-8 shape check rendered on the anchor's original [0,1] box, labeled.
2. **acados unavailable** (source build denied by environment policy) → blueprint-sanctioned
   CasADi+IPOPT fallback; NMPC internal model uses quasi-steady-state current elimination
   (La/Ra = 0.67 ms << Ts = 20 ms); evaluated plant always full-order. Frey et al. 2025
   acados recipe cited for ~1 ms deployment latency.
3. **Diff-sim integrator**: exact-ZOH discretization of the (linear) inner loop + 2nd-order
   pose update instead of literal torch-RK4 — more accurate on the stiff electrical states,
   ~25x faster; cross-validated vs the RK4 eval sim (t to 3 ms, e_l to 0.1 mm). The exact
   non-smooth metric in the numpy sim is the only reported metric.
4. **SAC budget** 500k steps, train_freq=2 (vs blueprint 1M): CPU-only throughput
   (18 fps measured); budgets + learning curves logged.
5. **NMPC noisy replicates** capped at 10 seeds on anchor4 (IPOPT cost); clean cells full.
6. **WP3 evaluation** uses DR seed 0 (training curves for seeds 0-2 in logs).
7. **Failure rule** reported under both readings of "3x nominal time" (desired-time and
   own-nominal); the blueprint phrase is ambiguous.
8. **Anchor PID timing** not reproducible from their published spec (V0 deviation, above).
