# Certified Gain-Policy Learning for Waypoint-Halt DDWMR Control

Simulation code, experiment configurations, per-run results, and figure-regeneration
scripts for:

> B. Lazkani, "Certified Gain-Policy Learning with Switching-Stability Guarantees for
> Waypoint-Halt Control of Differential-Drive Mobile Robots," *International Journal of
> Control, Automation, and Systems* (IJCAS), under review.

Everything is pure Python: NumPy/SciPy plant, PyTorch differentiable rollout, contextual
Bayesian optimization, CasADi/IPOPT NMPC, Stable-Baselines3 SAC.

The plant model, task metric, and verification scenarios reimplement — from the published
specification alone — the system studied in:
A. Deeb, B. Alsaleh, and I. Hatem, "AI-Driven Control Strategy for DDWMR: Neural
Network-Based Parameter Optimization and Real-Time Stabilization for Multi-Waypoint
Navigation," *IJCAS*, in press. Equation and table numbers in code comments refer to it.

---

## Start here: the WP7 correction campaign (newest material)

Results live in **two** campaigns. `passport/` is the original WP0–WP5 study; **`wp7/` is a
later correction campaign that re-ran several measurements and changed two conclusions.**
Where the two disagree, **WP7 supersedes WP0–WP5.**

| What it establishes | Script | Data |
|---|---|---|
| Head × label-source ablation; trains the certified BO policy | `scripts/wp7_ablation.py` | `wp7/results/wp7_ablation*.csv`, `wp7/models/` |
| Held-out re-evaluation with the correct policy | `scripts/wp7_reeval.py` | `wp7/results/wp7_reeval.csv` |
| Loop transport delay, separated by feedback path | `scripts/wp7_delay.py` | `wp7/results/wp7_delay_*.csv` |
| Latency: every controller, one harness, one statistic | `scripts/wp7_latency.py` | `wp7/results/wp7_latency.csv` |
| Training cost at matched parallelism and native batching | `scripts/wp7_cost_native.py` | `wp7/logs/wp7_cost_native.json` |
| Velocity-bias sweep across the (C3) threshold | `scripts/wp7_bias.py` | `wp7/results/wp7_bias.csv` |

### What WP7 found

**1. The evaluated controller was mislabelled.** Every downstream campaign loaded
`passport/models/policy_bptt_seed*.pt` while the results were reported as the
Bayesian-optimization policy (`scripts/wp5_eval.py`, controller key `ours`). The BO policy
was evaluated nowhere. The distilled BO networks were also trained with `head="linear"`, so
their output is unbounded and the gain-box certification assumption does not hold for them;
a certified BO policy had to be trained from scratch (`scripts/wp7_ablation.py`).

**2. Correcting it inverted the conclusion.** On the held-out ring the differentiable-rollout
policy reaches median task error **288** against **542** for Bayesian optimization and 633
for the brute-force-label network. BO reproduces grid quality better on the training
workspace (E/E_grid 1.004 against 1.125) and transfers worse.

**3. The transport-delay finding was withdrawn.** The earlier sweep delayed every feedback
signal together, including the wheel encoders, which on a real platform are wired to the
motor-controller board. Delaying each path separately (`scripts/wp7_delay.py`), there is
**no arrival loss on either path alone at any delay up to 200 ms**, for any controller. A
failure appears only with both paths delayed 200 ms at once — a configuration hardware does
not produce.

**4. The cost comparison was measured at mismatched parallelism** — the grid on 8 cores, the
learned methods on 5. Re-measured serially with each method in its own batching form
(`scripts/wp7_cost_native.py`), BO costs **337 s against the grid's 3,718 s**: 11× cheaper
at 100× fewer rollouts per target, not more expensive as the earlier numbers suggested.

Two earlier cost scripts were discarded rather than kept: they reimplemented the grid as one
gain pair across many targets, whereas `wp0_bruteforce.run_target` batches all 2,500 gain
pairs for a single target — about twelve times fewer calls — which inflated the grid's cost
in the opposite direction. `wp7_cost_native.py` times each method in its native form.

### Code changed by WP7

| Path | Change |
|---|---|
| `src/ddwmr/sim.py` | loop transport delay on the feedback path (`Disturbances.loop_delay_s`, `loop_delay_scope`) and an option to trigger arrival from the measured rather than the true pose. **Both default to off**; `tests/test_delay.py::test_zero_delay_is_bit_identical` asserts the WP0–WP5 numbers are unchanged. |
| `scripts/make_figures.py` | figures rendered at final print width; trade-off figure rebuilt on the re-measured latencies |

---

## Original campaign (WP0–WP5)

| Result | Script | Data |
|---|---|---|
| Reproduction gate against the anchor system | `scripts/wp0_verify.py` | `passport/results/wp0_verification.csv` |
| Exhaustive gain grid (the yardstick) | `scripts/wp0_bruteforce.py` | `passport/results/wp0_bruteforce.npz` |
| Sample efficiency: grid vs BO vs differentiable rollout | `scripts/wp1_train.py` | `passport/results/wp1_*.csv`, `passport/logs/wp1_budgets.json` |
| Switching stability, dwell bound, 12,509 segments | `scripts/wp2_switching.py` | `passport/results/wp2_*.csv` |
| Robustness, 39,000 perturbed runs | `scripts/wp3_robustness.py` | `passport/results/wp3_robustness.csv` |
| Tuned PID and SAC baselines | `scripts/wp4_pid_tune.py`, `scripts/wp4_rl_train.py` | `passport/results/wp4_*.csv` |
| Held-out evaluation and paths | `scripts/wp5_eval.py` | `passport/results/wp5_*.csv` |
| All figures | `scripts/make_figures.py` | `passport/figures/` |

## Layout

```
src/ddwmr/    plant model (6th-order voltage-level DDWMR), NPC + LQR controllers,
              task metric, trainers (BO, differentiable rollout), baselines
scripts/      one entry point per work package
tests/        plant, metric and loop-delay unit tests
passport/     WP0-WP5 record
  environment/    lockfile, Python version, CPU model
  configs/        one YAML (with seed) per experiment
  results/*.csv   per-run rows behind every table and figure
  figures/        figures (PDF + PNG), regenerable from results/
  models/         trained policies and baselines
  logs/           training curves, solver stats, documented deviations
wp7/          correction campaign: results/, models/ (40 ablation checkpoints
              plus BO labels), logs/ (machine and parallelism for every timing run)
```

Every result row carries its configuration and seed; no result exists without them.
Failures and divergences appear as rows, never dropped. WP0-WP5 keep one YAML per
experiment in `passport/configs/`; WP7 records the same information — seed, checkpoints
loaded, sweep grid, machine, core count and load average — in a `*_meta.json` beside
each result in `wp7/logs/`.

## Reproducing

```bash
python -m venv .venv && .venv/bin/pip install -r passport/environment/requirements_lock.txt
.venv/bin/python tests/test_sim.py     # plant, metric, RK4-vs-RK45 cross-check
.venv/bin/python tests/test_diffsim.py # differentiable rollout and gradients
.venv/bin/python tests/test_delay.py   # loop-delay path; zero delay is bit-identical

# original campaign
.venv/bin/python scripts/wp0_verify.py       # reproduction gate; run this first

# correction campaign
.venv/bin/python scripts/wp7_ablation.py     # BO labels + 2x2 ablation
.venv/bin/python scripts/wp7_reeval.py       # held-out re-evaluation (needs the above)
.venv/bin/python scripts/wp7_delay.py        # delay by feedback path
.venv/bin/python scripts/wp7_bias.py         # velocity-bias sweep
.venv/bin/python scripts/wp7_latency.py      # run ALONE - see below
.venv/bin/python scripts/wp7_cost_native.py  # run ALONE - see below
.venv/bin/python scripts/make_figures.py     # regenerate every figure
```

**Timing experiments must run with nothing else active.** An earlier latency run was taken
while a training job saturated five cores and reported a maximum of 15,030 µs for an
operation whose median is 3.6 µs. `wp7/logs/` records the machine, core count and load
average for every timing measurement; treat any timing result without them as unusable.

## License

MIT — see [LICENSE](LICENSE).
