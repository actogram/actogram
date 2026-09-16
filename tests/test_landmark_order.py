"""
Regression tests for landmark order and loop closure.

Before the fix, a long inactive stretch covering an activity peak (typically
non-wear) was accepted as the night both before and after that peak, putting
wake-up after the peak and bedtime before it. The night minimum was not
checked against its night, and the loop-closing fill for a missing last
bedtime added one day instead of one loop period.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

from actogram import intervalts

from test_synthetic import DAY, SLEEP, square_wave_week


LANDMARKS = ["t_start", "t_min", "t_onset", "t_max", "t_offset"]
HOUR = DAY // 24
NHANES = Path.home() / "work/NHANES/NPZ/nhanes_steps.npz"

# NHANES userids that broke under the previous code, by failure
NHANES_CASES = {
    31157: "bedtime before peak",
    31731: "bedtime before peak, full week",
    31206: "wake-up after peak",
    33766: "wake-up after peak, full week",
    31867: "night minimum after wake-up",
    33031: "night minimum after wake-up, full week",
    36204: "night minimum before night start, full week",
    31159: "loop not closed",
    32218: "loop not closed",
    39587: "loop not closed, one valid day",
}


def assert_well_formed(out, n):
    """Landmarks ordered per day, days contiguous, loop closed."""
    t = np.stack([np.asarray(out[key]) for key in LANDMARKS])
    assert t.shape[0] == len(LANDMARKS)
    if t.shape[1] == 0:
        return
    assert np.all(np.diff(t, axis=0) >= 0), t
    np.testing.assert_array_equal(out["t_start"][1:], out["t_offset"][:-1])
    assert abs(out["t_offset"][-1] - (out["t_start"][0] + n)) <= HOUR


def run_strict(x, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        return intervalts.timestamps(x, **kwargs)


def quiet_days_with_burst(burst=60):
    """Square-wave week with two inactive days holding one short activity burst."""
    x = square_wave_week()
    x[2 * DAY + SLEEP:4 * DAY + SLEEP] = 0
    x[3 * DAY + 800:3 * DAY + 800 + burst] = 1
    return x


@pytest.mark.parametrize("burst", [30, 60, 120])
def test_inactive_stretch_around_peak_keeps_order(burst):
    x = quiet_days_with_burst(burst)
    assert_well_formed(run_strict(x, loop=True), len(x))


@pytest.mark.parametrize("n_active", [60, 240, DAY - SLEEP])
def test_single_active_block_loop_closes(n_active):
    x = np.zeros(7 * DAY, dtype=int)
    x[3 * DAY + SLEEP:3 * DAY + SLEEP + n_active] = 1
    out = run_strict(x, loop=True)
    assert len(out["t_max"]) == 1
    assert_well_formed(out, len(x))


@pytest.mark.skipif(not NHANES.exists(), reason="NHANES steps file not available")
@pytest.mark.parametrize("prec", [False, True])
@pytest.mark.parametrize("userid", list(NHANES_CASES))
def test_nhanes_landmarks_well_formed(userid, prec):
    d = np.load(NHANES)
    x = (d["steps"][d["userid"] == userid][0] >= 5).astype(int)
    assert_well_formed(run_strict(x, prec=prec, loop=True), len(x))
