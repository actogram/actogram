"""
Regression tests for nights without a qualifying sleep interval.

Before the fix, such nights left NaN onset / offset values that were cast to
integers (0 or duplicated landmarks), dropped days in loop mode, or raised
ValueError in _select_intervals. Days with no selected landmarks also returned
sleep intervals spanning the whole padding.
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


def restless_week(nights, burst=2, step=10):
    """Square-wave week where given nights have short activity bursts
    every `step` minutes, so no rest interval reaches nmin."""
    x = square_wave_week()
    for night in nights:
        n = night * DAY
        x[n:n + SLEEP] = 0
        for s in range(n, n + SLEEP, step):
            x[s:s + burst] = 1
    return x


def run_strict(x, **kwargs):
    """Run intervalts.timestamps turning RuntimeWarnings (NaN casts) into errors."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        return intervalts.timestamps(x, **kwargs)


def assert_consistent_days(out, n_days):
    for key in LANDMARKS:
        assert len(out[key]) == n_days, (key, out[key])
    t = np.stack([out[key] for key in LANDMARKS])
    assert np.all(np.diff(t, axis=0) >= 0), t
    np.testing.assert_array_equal(out["t_start"][1:], out["t_offset"][:-1])


@pytest.mark.parametrize("loop", [True, False])
@pytest.mark.parametrize("nights", [[3], [2, 3]])
def test_restless_nights_keep_all_days(nights, loop):
    out = run_strict(restless_week(nights), loop=loop)
    assert_consistent_days(out, 7)
    for night in nights:
        assert abs(out["t_offset"][night - 1] - night * DAY) <= 2 * HOUR
        assert abs(out["t_onset"][night] - (night * DAY + SLEEP)) <= 2 * HOUR
    assert len(out["sleep"]) == 7 - len(nights)


@pytest.mark.parametrize("night", [0, 6])
def test_restless_edge_night_loop_closes(night):
    x = restless_week([night])
    out = run_strict(x, loop=True)
    assert_consistent_days(out, 7)
    assert abs(out["t_offset"][-1] - (out["t_start"][0] + len(x))) <= HOUR


def test_no_selected_days_drops_intervals():
    sleep = np.array([[-2 * DAY, 9 * DAY]])
    rest = np.array([[100, 200]])
    empty = np.zeros(0, dtype=int)
    t_start, t_min, sleep, rest, is_closed = intervalts._select_intervals(
        np.zeros(7 * DAY), DAY, empty, empty, empty, empty, sleep, rest, True)
    assert sleep.shape == (0, 2) and rest.shape == (0, 2)
    assert not is_closed


@pytest.mark.skipif(not NHANES.exists(), reason="NHANES steps file not available")
@pytest.mark.parametrize("userid", [31248, 32553, 41026, 41377])
@pytest.mark.parametrize("prec", [False, True])
def test_nhanes_former_nan_subjects(userid, prec):
    d = np.load(NHANES)
    x = (d["steps"][d["userid"] == userid][0] >= 5).astype(int)
    out = run_strict(x, prec=prec, loop=True)
    n_days = len(out["t_max"])
    for key in LANDMARKS:
        assert len(out[key]) == n_days
        assert out[key].dtype.kind == "i"
    if n_days == 0:
        assert len(out["sleep"]) == 0 and len(out["rest"]) == 0
