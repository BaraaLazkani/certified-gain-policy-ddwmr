# Certified Gain-Policy Learning for Waypoint-Halt DDWMR Navigation

Simulation code, experiment configurations, per-run results, and figure-regeneration
scripts accompanying the manuscript:

> B. Lazkani, "Certified Gain-Policy Learning with Switching-Stability Guarantees for
> Waypoint-Halt Navigation of Differential-Drive Mobile Robots," submitted to the
> *International Journal of Control, Automation, and Systems* (IJCAS).

All experiments are pure Python (NumPy/SciPy plant, PyTorch differentiable rollout,
BoTorch contextual Bayesian optimization, CasADi/IPOPT NMPC, Stable-Baselines3 SAC).

## Layout

```
src/ddwmr/    plant model (6th-order voltage-level DDWMR), NPC + LQR controllers,
              task metric, trainers (BO, BPTT), baselines
scripts/      one entry point per work package (WP0 verification ... WP5 evaluation)
tests/        plant and metric unit tests
passport/     materials passport: every result in the paper
  README.md          maps every artifact to its experiment ID
  environment/       lockfile, Python version, CPU, commit hash
  configs/           one YAML (with seed) per experiment
  results/*.csv      per-run rows behind every table and figure
  figures/           paper figures (PDF + PNG), regenerable from results/
  logs/              training curves, solver stats, documented deviations
```

## Reproducing

```bash
python -m venv .venv && .venv/bin/pip install -r passport/environment/requirements_lock.txt
.venv/bin/python scripts/wp0_verify.py      # verification gate vs. the anchor paper
.venv/bin/python scripts/wp1_train.py ...   # see passport/configs/ for exact settings
```

Every figure in the paper is regenerable from `passport/results/*.csv` via the scripts
in `scripts/`; no result exists without its config and seed.

## License

MIT — see [LICENSE](LICENSE).
