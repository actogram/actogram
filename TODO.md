# sleep_intervals → actogram handover

This file is a snapshot of the project recap and to-do list from a Cursor session
on 16 Sep 2026, plus a note for the next agent. The author is moving this
code into a new GitHub project named **actogram**.

---

## Handover (read this first)

You are continuing work that started in `actogram/sleep_intervals` (local path
was `/Users/timpyrkov/core/sleep_intervals`). Tim paused the library in Feb
2025, then resumed in Sep 2026 to refresh the codebase and fix known bugs
before copying it into the new **actogram** repo.

**Do not revert** the `_normalize_scores` change in `peakts.py`. Tim chose to
**keep** the new condition:

```python
if (1 - np.min(scores[i0:i1])) > epsilon:
    scores[i0:i1] -= scores[i0:i1].min()
```

instead of `np.std(scores[i0:i1]) > epsilon`. Plain-language reason: a quiet
flat stretch (low activity, almost no wiggle) must stay rest (score → 0). The
old `std` check treated “flat” as “already done”, then later division flipped
that night into a fake peak (score → 1). A busy high plateau still stays at 1
under both rules.

None of the Sep 2026 work was committed in `sleep_intervals` (still three
commits on `master`, last: `d77054c` “fix t_start”, 6 Feb 2025). If this
handover lands in a fresh **actogram** repo, treat the working tree as the
source of truth and check `git status` before assuming anything is committed.

**Next real task:** looped biobank day-boundary / `t_start` behavior. That was
the last committed work in 2025 and the only open item on the restart list.
Do not casually rewrite the detector; add tests around `loop=True` short
recordings (≤7 days) first.

Run tests from the package root:

```bash
python -m pytest tests/test_synthetic.py -v
```

Public import after `pip install -e ".[dev]"` from the repo root:

```python
from actogram import peakts, intervalts, plot
```

Package layout is `src/actogram/` (installable as `actogram`). Tests import
`actogram`, not the old `sleep_intervals` name.

---

## What this library does

Turn a 1D **binarized** activity series (`1` = active, `0` = inactive;
typically minute-level wearable steps) into circadian landmarks:

| Key | Meaning | How it is found |
|---|---|---|
| `t_start` | Day boundary (unlabeled in the plot) | Previous `t_offset`, rolled; first one is synthesized from `t_min` |
| `t_min` | Night / min activity | Argmin of normalized scores between peaks (search reversed) |
| `t_onset` | Wake up | Last sample before `t_max` where score < `level` (default 0.5) |
| `t_max` | Max activity | Local max / plateau midpoint after Hann smoothing |
| `t_offset` | Go to sleep | First sample after `t_max` where score < `level` |
| `sleep` / `rest` | Night sleep / daytime nap | `intervalts` only: zero-runs scored against the trough vs peak |
| `scores` | Daytime-ness curve (0–1) | Smoothed activity, normalized between neighboring peaks |

Day shape: night/sleep (`t_start` → `t_min` → `t_onset`) then day/activity
(`t_onset` → `t_max` → `t_offset`).

`loop=True` wraps short biobank recordings so the first and last days are not
truncated.

---

## Architecture

```
src/actogram/
  checks.py  →  peakts.py  →  intervalts.py  →  plot.py
```

Pipeline:

```
binarized x  →  checks (coerce + warn)  →  peakts (density peaks)
                                              ↓
                                         intervalts (sleep vs rest)
                                              ↓
                                         plot.plot_week
```

| File | Role |
|---|---|
| `src/actogram/peakts.py` | Activity-density peak timestamps. Public: `timestamps`, `timestamps_and_scores`, `timestamps_biobank`. Helpers: `pad`, `smoother`, `find_peaks`, `find_plateaus`, `pad_timestamps`. |
| `src/actogram/intervalts.py` | Rest/sleep intervals on top of peaks. Same public names, plus `nmin`, `atol`, `rtol`, `durations`. Helpers: `find_intervals`, `fill_gaps`. |
| `src/actogram/checks.py` | Input sanitizer decorator wrapping the public timestamp functions. |
| `src/actogram/plot.py` | One-week matplotlib overlay (`plot_week`). |
| `src/actogram/__init__.py` | Re-exports `peakts`, `intervalts`, and `plot`. |

Pipeline: pad (or wrap) → optional hourly downsample if `prec=False` → Hann
smooth → peaks and plateaus → normalize scores 0–1 → `t_min` / onset / offset
→ (`intervalts`) fill activity gaps, find rest runs, label sleep vs nap →
unpad and keep the cycle set that best overlaps the original series.

### Parameters

| Param | Default | Meaning |
|---|---|---|
| `window` | `1440` | One day in samples (minutes). `24` = hourly. |
| `level` | `0.5` | Score cutoff for onset / offset, and sleep vs rest. |
| `prec` | `False` | `False` = downsample to hourly, then upsample indices. |
| `loop` | `False` | Wrap the series; use for short biobank recordings. |
| `nmin` | `30` | Min rest-interval length (minutes). |
| `atol` | `5` | Fill activity gaps this short before finding rest. |
| `rtol` | `1` | Fill rest gaps this short when merging intervals. |
| `durations` | `False` | If `True`, `sleep`/`rest` become per-day minute totals. |

Input contract: equispaced binary ints. `checks.sanity_check` coerces lists,
binarizes values > 1, and only clips NaN/Inf on non-numeric dtypes. NaN/Inf on
a float array is documented as forbidden and is **not fully guarded**.

`scores` is **not** shifted/trimmed by `_shift_timestamps`; it still includes
padding. `plot.plot_week` accounts for a 2-day pad.

---

## Suggested to-do list

Status as of 16 Sep 2026, after the resume session.

- [x] Decide on the `_normalize_scores` change, then keep or revert it.
  **Kept** `(1 - min) > epsilon`. Do not revert.
- [x] Fix `timestamps_biobank`: iterate `dcts` (not empty `dct`), use
  `peakts.pad_timestamps` in `intervalts`, pass `npad` as a keyword.
- [x] Repair `_add_tmin` so the `loop=True` miss path returns updated
  `t_onset` / `t_max` / `t_offset` (it used undefined `t_offset` and dropped
  slices only locally).
- [x] Add a synthetic 7-day square-wave test (16 h on / 8 h off) for both
  `peakts` and `intervalts`. See `tests/test_synthetic.py` (9 tests).
- [x] Add `pyproject.toml` so this is importable (`numpy`, `tqdm`,
  `matplotlib`; pytest in optional `dev`). Package name is `actogram`,
  modules live under `src/actogram/`, install with `pip install -e ".[dev]"`.
- [x] Fix `plot.py` stale docstring (`timestamps_*_dict`, unused `colors`)
  and remove unused `tqdm` import.
- [x] **Revisit looped biobank day-boundary / `t_start` cases** — last
  committed work in Feb 2025 (`d77054c`). Tests in `tests/test_loop_tstart.py`.
  Kept rolled-`t_offset` / overlap-via-`t_offset`. Fixes: skip empty overlap
  masks; fill a missing last `t_offset` when `loop=True`; do not shrink retry
  `npad` below 2; trim extra padded `t_min` for `loop=False`.

### Still known, not on the original restart list

- `intervalts._timestamps` now calls private `_timestamps_and_scores` (was
  accidentally calling the public decorated wrapper). Keep that.

---

## Issues from the recap (all the crashers were fixed)

These were found in the Sep 2026 review. The first four are done in the
working tree; do not “fix” them again unless git history in the new repo
does not include these edits.

1. **Fixed.** `timestamps_biobank` stacked `for d in dct` (empty dict) instead
   of `dcts`.
2. **Fixed.** `intervalts.timestamps_biobank` called `pad_timestamps` without
   importing it.
3. **Fixed.** `intervalts._timestamps_and_scores` passed `npad` positionally
   into the `durations` slot, so extra padding on unclosed loops never ran.
4. **Fixed.** `peakts._add_tmin` `loop=True` miss path: undefined `t_offset`,
   local-only slices.
5. **Fixed.** `plot.py` docstring / unused import.
6. **Fixed.** `checks.return_dict` unused; removed.

---

## Where Tim left off in 2025

Last commit `d77054c` “fix t_start” rewired day-boundary selection for looped
samples: `t_start` is a rolled `t_offset`, overlap is measured with `t_offset`
not `t_start`, and empty `sleep`/`rest` arrays became shape `(0, 2)`.

The Sep 2026 scoring tweak is a continuation of that same edge/scoring work,
not a random edit. Looped biobank `t_start` tests and the short-recording
crash/alignment fixes are in the working tree (`tests/test_loop_tstart.py`).
