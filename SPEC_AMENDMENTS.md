# Binding spec amendments (from ARS theory side, 2026-06-12)

Received before any compute was spent — no reruns required.

1. **Gain box correction**: K_p ∈ [0.1, 1.0] (NOT [0, 1]); K_θ ∈ [0.5, 3.0] unchanged.
   Applies to: brute-force grid, BPTT policy squashing, BO bounds, PNN labels, PID tuning analog.
   Exception: WP0 Fig-8 surface *shape comparison* is rendered over the anchor's original
   [0,1]×[0.5,3] box (verification artifact only, labeled as such in README).
2. **WP0 gate**: explicit PASS/FAIL on anchor Tables 4–7 qualitative rankings + Fig 8 surface
   shape; comparison numbers in results/wp0_verification.csv. On FAIL: stop, report, do not
   proceed to WP1–WP5 conclusions.
3. **Theory-verification data (theory_section.tex P1–P5, conditions C1–C3)**:
   - WP2 per-segment rows: segment length L_k, K_p, K_θ, measured arrival time, ε, and the
     theoretical bound T_k = ln(4)/K_θ + (2/K_p)·ln((L_k+ε)/ε).
   - Full V(t) time series for WP2 switching policies (a) 1 cm halt, (b) 5 cm switch,
     (c) fixed timer — not just summaries.
   - Inner-loop tracking errors δ_v(t) = v − v_ref, δ_ω(t) = ω − ω_ref: sup + RMS per run for
     ALL WP2/WP3 runs; full time series for a documented subset.
   - WP3 separate labeled conditions: common_mode_slip, differential_slip, additive_bias,
     zero_mean_noise (theory predicts qualitatively different outcomes per condition).
4. **Reproducibility**: figures regenerable from results/*.csv via included script; one YAML
   config + seed per experiment ID; lockfile + Python version + CPU model + git commit hash in
   environment/; latency measured on this same machine for all controllers.
5. **Honesty**: failures/divergences as rows, never dropped; raw per-run data, not only
   aggregates; NMPC/RL wins reported unfiltered.
