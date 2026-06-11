# WP0 PID timing forensics

Anchor PID (Table 3) reproduces path shape but not arrival time.

Evidence:
- e_line ours/anchor: 38.9/40, 67.4/68.4, 74.5/75.6, 48.5/48.8 (all 4 targets, <3%)
- e_th  ours/anchor: 34.1/34.8, 513.7/514.4, 596.7/603.6, 96.9/92.9 (<2%)
- t_end ours vs anchor-implied (K_t=3.5, validated to 0.001 s on Lyap1/Lyap2):
  5.53/2.52-2.56, 7.06/3.18, 7.30/3.29, 5.78/2.60  -> consistent ratio ~2.2x
Hypotheses tested and rejected:
1. Integral gain per-sample scaling (KI x5..x100): degrades or destabilizes; never
   reproduces timing (this file's companion test in conversation log).
2. Swapped channel assignment (PID2->v, PID1->w): diverges on all 4 targets.
3. cm-units error / voltage-saturated sprint: implies arrival ~0.6-0.8 s and
   e_time ~178 at T7; anchor reports 3.4. Rejected.
Additional anomaly: anchor e_total != 1.5*e_line + 2*e_th + 2*e_time for ANY row of
Tables 4-7 using their own components; Lyap rows reconcile exactly with
C_l = 0.062 m (= per-target brute-force max e_l, our reverse-engineered value);
PID rows reconcile with no C_l. Conclusion: anchor PID rows contain a
bookkeeping/implementation artifact; our plant reproduces every quantity that
is reproducible from the published spec.
