"""Tests for the loop-transport-delay path added for WP7 (Section 5.5).

The delay support changes `_deriv` and `run_closedloop`, which every earlier
work package also uses, so the first and most important test is that the
zero-delay path is untouched.
"""
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ddwmr.sim import run_closedloop, LyapGains, Disturbances   # noqa: E402
from ddwmr.params import LYAP1                                   # noqa: E402

TGT = np.array([[0.6, 0.4], [-0.3, 0.5], [0.5, -0.5]])


def _gains(n):
    return LyapGains(Kp=np.full(n, LYAP1[0]), Kth=np.full(n, LYAP1[1]))


def test_zero_delay_is_bit_identical():
    """loop_delay_s=0 must reproduce the no-delay code path exactly.

    This is the regression that protects WP0-WP5: those campaigns were run
    before delay support existed and their numbers must not move.
    """
    a = run_closedloop(TGT, "lyap", _gains(len(TGT)), T_max=30.0)
    b = run_closedloop(TGT, "lyap", _gains(len(TGT)), T_max=30.0,
                       dist=Disturbances(loop_delay_s=0.0))
    assert np.array_equal(a.t_end, b.t_end), (a.t_end, b.t_end)
    assert np.array_equal(a.e_l_mean, b.e_l_mean)
    assert np.array_equal(a.arrived, b.arrived)
    print("zero-delay identical to the no-delay path: t_end", a.t_end)


def test_delay_must_be_a_multiple_of_dt():
    """A transport delay that is not an integer number of steps is refused
    rather than silently rounded."""
    try:
        run_closedloop(TGT, "lyap", _gains(len(TGT)), T_max=5.0,
                       dist=Disturbances(loop_delay_s=0.0015))
    except ValueError as e:
        assert "multiple" in str(e)
        print("non-integer delay correctly refused:", e)
        return
    raise AssertionError("expected ValueError for a non-integer-step delay")


def test_delay_scopes_are_distinguishable():
    """pose / wheels / all must actually differ, or the scope argument is
    doing nothing and Section 5.5's separation is meaningless."""
    out = {}
    for scope in ("pose", "wheels", "all"):
        r = run_closedloop(TGT, "lyap", _gains(len(TGT)), T_max=30.0,
                           dist=Disturbances(loop_delay_s=0.100,
                                             loop_delay_scope=scope))
        out[scope] = r.t_end.copy()
    assert not np.array_equal(out["pose"], out["wheels"]), \
        "pose-only and wheels-only delay gave identical results"
    assert not np.array_equal(out["all"], out["pose"]), \
        "delaying both paths matched delaying pose alone"
    print("scopes differ at tau=100 ms:",
          {k: np.round(v, 4).tolist() for k, v in out.items()})


def test_delay_degrades_monotonically():
    """Arrival time should not improve as transport delay grows."""
    ts = []
    for tau in (0.0, 0.02, 0.05, 0.10):
        r = run_closedloop(TGT[:1], "lyap", _gains(1), T_max=30.0,
                           dist=Disturbances(loop_delay_s=tau))
        ts.append(float(r.t_end[0]))
    assert all(b >= a - 1e-9 for a, b in zip(ts, ts[1:])), ts
    print("arrival time non-decreasing in tau:", [round(t, 4) for t in ts])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
