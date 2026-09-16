import numpy as np

from actogram import intervalts, peakts

from test_synthetic import DAY, SLEEP, square_wave_week


LANDMARKS = ["t_start", "t_min", "t_onset", "t_max", "t_offset"]
HOUR = DAY // 24


def _closed_gap(out, n):
    return abs(out["t_offset"][-1] - (out["t_start"][0] + n))


def test_loop_short_recordings_have_matching_landmark_lengths():
    for n_days in [1, 2, 3, 5, 7]:
        x = square_wave_week(n_days)
        out = peakts.timestamps(x, window=DAY, prec=True, loop=True)
        n_max = len(out["t_max"])
        assert n_days - 1 <= n_max <= n_days + 1
        for key in LANDMARKS:
            assert len(out[key]) == n_max, (n_days, key, len(out[key]), n_max)


def test_loop_t_start_is_rolled_t_offset_and_closes():
    for n_days in [1, 2, 3, 5, 7]:
        x = square_wave_week(n_days)
        out = peakts.timestamps(x, window=DAY, prec=True, loop=True)
        assert len(out["t_start"]) >= 1
        if len(out["t_start"]) > 1:
            np.testing.assert_array_equal(out["t_start"][1:], out["t_offset"][:-1])
        assert _closed_gap(out, len(x)) <= HOUR


def test_loop_false_drops_padded_extra_tmin():
    x = square_wave_week(7)
    out = peakts.timestamps(x, window=DAY, prec=True, loop=False)
    assert len(out["t_min"]) == len(out["t_max"])
    assert len(out["t_start"]) == len(out["t_max"])


def test_intervalts_loop_one_day_does_not_crash():
    x = square_wave_week(1)
    out = intervalts.timestamps(x, window=DAY, prec=True, loop=True)
    assert len(out["t_max"]) >= 1
    assert out["sleep"].ndim == 2 and out["sleep"].shape[1] == 2
    assert _closed_gap(out, len(x)) <= HOUR


def test_intervalts_loop_short_recordings_match_nights():
    for n_days in [1, 2, 3, 7]:
        x = square_wave_week(n_days)
        out = intervalts.timestamps(x, window=DAY, prec=True, loop=True)
        assert len(out["t_start"]) == n_days
        for key in LANDMARKS:
            assert len(out[key]) == n_days
        np.testing.assert_allclose(out["t_start"], np.arange(n_days) * DAY, atol=HOUR)
        np.testing.assert_allclose(out["t_onset"], np.arange(n_days) * DAY + SLEEP, atol=HOUR)
        assert _closed_gap(out, len(x)) <= HOUR
        sleep = out["sleep"]
        assert len(sleep) == n_days
        night_minutes = 0
        for i0, i1 in sleep:
            i0 = max(int(i0), 0)
            i1 = min(int(i1), len(x))
            if i1 > i0:
                night_minutes += int(np.sum(x[i0:i1] == 0))
        assert night_minutes > n_days * SLEEP / 2


def test_intervalts_loop_truncated_week_still_closes():
    x = square_wave_week(7)[:-3 * HOUR]
    out = intervalts.timestamps(x, window=DAY, prec=True, loop=True)
    assert len(out["t_start"]) >= 6
    assert _closed_gap(out, len(x)) <= HOUR
    for key in LANDMARKS:
        assert len(out[key]) == len(out["t_max"])
