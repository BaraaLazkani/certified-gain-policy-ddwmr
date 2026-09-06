"""WP7-1d: training cost with each method in its NATIVE batching form.

Three earlier attempts are superseded, and it is worth recording why, because
each failed in a different direction:

  1. The earlier numbers timed the grid at Pool(8) and BO at Pool(5).
     Different core counts -> the wall times were never comparable, and the
     2.97 ms/rollout break-even derived from them does not stand.
  2. wp7_cost.py timed both serially, but reimplemented the grid as one gain
     pair evaluated across N targets. wp0_bruteforce.run_target does the
     opposite: all 2500 gain pairs for ONE target per call. The
     reimplementation incurs ~12x more call overhead, inflating the grid's
     cost and flattering BO.
  3. wp7_cost_width.py repeated (2) at a larger target batch; same defect.

Here each method is timed exactly as its own script implements it:

  grid : 2500-gain batch per target (wp0_bruteforce.run_target), 205 targets
  BO   : 25 sequential iterations, 200-target batch each (wp1_train.bo_worker)

Both serial, one process, same machine, nothing else running.
"""
import sys
import time
import json
import pathlib

import numpy as np

CODE = pathlib.Path(__file__).resolve().parents[1]   # repository root
sys.path.insert(0, str(CODE / "src"))
sys.path.insert(0, str(CODE / "scripts"))
import wp1_train as W1                                    # noqa: E402

PASSPORT = CODE / "passport"
W1.RES, W1.MOD, W1.LOG = (PASSPORT / "results", PASSPORT / "models",
                          PASSPORT / "logs")
from ddwmr.sim import run_closedloop, LyapGains           # noqa: E402

N_TARGETS_TIMED = 5          # of 205; each call is a full 2500-gain batch
N_TARGETS_TOTAL = 205        # 200 workspace + 4 validation + 1 fig-8
T_MAX = 40.0
BO_SERIAL_S = 337.0          # measured end-to-end, WP7-1 Pool(1), 200 targets
BO_ROLLOUTS = 200 * 25


def main():
    targets, C_l, E_surf, gains, _ = W1.load_bf()
    B = len(gains)
    print(f"grid, native batching: {B}-gain batch per target, "
          f"{N_TARGETS_TOTAL} targets total, timing {N_TARGETS_TIMED}",
          flush=True)
    t0 = time.perf_counter()
    for i in range(N_TARGETS_TIMED):
        tg = np.tile(targets[i], (B, 1))
        run_closedloop(tg, "lyap",
                       LyapGains(Kp=gains[:, 0].copy(), Kth=gains[:, 1].copy()),
                       T_max=T_MAX)
        print(f"  target {i + 1}/{N_TARGETS_TIMED}: "
              f"{time.perf_counter() - t0:.0f}s", flush=True)
    w = time.perf_counter() - t0

    per_target = w / N_TARGETS_TIMED
    grid_full = per_target * N_TARGETS_TOTAL
    grid_rollouts = N_TARGETS_TOTAL * B

    print(f"\n  grid: {per_target:.1f} s/target -> {grid_full:.0f} s serial "
          f"for {grid_rollouts} rollouts "
          f"({grid_full / grid_rollouts * 1e3:.2f} ms/rollout)")
    print(f"  BO  : {BO_SERIAL_S:.0f} s serial for {BO_ROLLOUTS} rollouts "
          f"({BO_SERIAL_S / BO_ROLLOUTS * 1e3:.2f} ms/rollout incl. surrogate)")
    print(f"\n  rollout saving : {grid_rollouts / BO_ROLLOUTS:6.1f}x")
    print(f"  COST saving    : {grid_full / BO_SERIAL_S:6.1f}x  "
          f"(serial, native batching)")

    out = CODE.parent / "wp7" / "logs" / "wp7_cost_native.json"
    out.write_text(json.dumps(dict(
        grid_targets_timed=N_TARGETS_TIMED, grid_batch=B,
        grid_targets_total=N_TARGETS_TOTAL, grid_s_per_target=per_target,
        grid_full_serial_s=grid_full, grid_rollouts=grid_rollouts,
        grid_ms_per_rollout=grid_full / grid_rollouts * 1e3,
        bo_serial_s=BO_SERIAL_S, bo_rollouts=BO_ROLLOUTS,
        rollout_ratio=grid_rollouts / BO_ROLLOUTS,
        cost_ratio=grid_full / BO_SERIAL_S,
        note=("Each method timed in the batching form its own script uses. "
              "Supersedes wp7_cost.py and wp7_cost_width.py, which "
              "reimplemented the grid with ~12x more call overhead, and the "
              "earlier Pool(8)-vs-Pool(5) comparison.")), indent=1))
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    main()
