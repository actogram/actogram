import numpy as np

from actogram import intervalts, peakts


DAY = 1440
SLEEP = 8 * 60
WAKE = DAY - SLEEP
WEEK_DAYS = 7


def square_wave_week(n_days=WEEK_DAYS):
    """8 h inactive, then 16 h active, repeated. Minute sampling."""
    day = np.concatenate([np.zeros(SLEEP, dtype=int), np.ones(WAKE, dtype=int)])
    return np.tile(day, n_days)


def test_peakts_finds_about_one_peak_per_day():
    x = square_wave_week()
    out = peakts.timestamps_and_scores(x, window=DAY, prec=True)
    n_max = len(out["t_max"])
    assert 5 <= n_max <= 8
    for key in ["t_start", "t_min", "t_onset", "t_offset"]:
        assert 5 <= len(out[key]) <= 9
        assert abs(len(out[key]) - n_max) <= 1


def test_peakts_onsets_are_about_a_day_apart():
    x = square_wave_week()
    out = peakts.timestamps(x, window=DAY, prec=True)
    onset = np.sort(out["t_onset"])
    gaps = np.diff(onset)
    assert np.all(np.abs(gaps - DAY) < DAY / 2)


def test_night_scores_lower_than_day():
    x = square_wave_week()
    out = peakts.timestamps_and_scores(x, window=DAY, prec=True)
    pad = 2 * DAY
    scores = out["scores"][pad : pad + len(x)]
    assert scores[x == 0].mean() < scores[x == 1].mean()


def test_quiet_week_returns_empty_timestamps():
    x = np.zeros(WEEK_DAYS * DAY, dtype=int)
    out = peakts.timestamps(x, window=DAY, prec=True)
    for key in ["t_start", "t_min", "t_onset", "t_max", "t_offset"]:
        assert len(out[key]) == 0


def test_intervalts_sleep_covers_nights():
    x = square_wave_week()
    out = intervalts.timestamps(x, window=DAY, prec=True, nmin=30)
    sleep = out["sleep"]
    assert sleep.ndim == 2 and sleep.shape[1] == 2
    assert len(sleep) >= 5
    night_minutes = 0
    for i0, i1 in sleep:
        i0 = max(int(i0), 0)
        i1 = min(int(i1), len(x))
        if i1 > i0:
            night_minutes += int(np.sum(x[i0:i1] == 0))
    assert night_minutes > 5 * SLEEP / 2


def test_peakts_biobank_stacks_samples():
    x = np.stack([square_wave_week(), square_wave_week()])
    out = peakts.timestamps_biobank(x, window=DAY, prec=True)
    assert out["t_max"].shape[0] == 2
    assert out["t_max"].shape[1] == 10
    assert np.isfinite(out["t_max"][0]).sum() >= 5


def test_intervalts_biobank_stacks_samples():
    x = np.stack([square_wave_week(), square_wave_week()])
    out = intervalts.timestamps_biobank(x, window=DAY, prec=True)
    assert out["sleep"].shape[0] == 2
    assert np.isfinite(out["sleep"][0]).sum() >= 5


def test_normalize_scores_keeps_high_plateau():
    x = np.ones(21)
    t_max = np.array([5, 15])
    t_plateau = np.array([[4, 6], [14, 16]])
    scores = peakts._normalize_scores(x, t_max, t_plateau)
    assert scores[5] > 0.9
    assert scores[15] > 0.9


def test_normalize_scores_quiet_flat_is_rest():
    x = np.concatenate([np.ones(5), np.full(11, 0.2), np.ones(5)])
    t_max = np.array([2, 18])
    t_plateau = np.zeros((0, 2), dtype=int)
    scores = peakts._normalize_scores(x, t_max, t_plateau)
    assert scores[10] < 0.5
    assert scores[2] > scores[10]
    assert scores[18] > scores[10]
