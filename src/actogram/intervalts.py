#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
from tqdm import tqdm
from . import checks, peakts

"""
Copyright © 2025 Actogram LTD (https://actogram.ai)

This module (interval time stamps) contains functions to extract circadian 
rhythm timestamps based on REST INTERVALS DETECTION.
"""


def find_intervals(x, tol=0, nmin=None, sort=False):
    """
    Finds continuous positive intervals in 1d-array.
    
    Parameters
    ----------
    x : ndarray
        1D array of non-negative numeric values
    tol : int, default 0
        Gap duration tolerance
    nmin : int or None, default None
        Minimal length of intervals
    sort : bool, default False
        If False - keep interval order by index in the array 
        If True - sort descending by interval duration

    Returns
    -------
    ndarray
        2D array - N intervals x 2 indices (start, end)

    """
    assert x.ndim == 1 and not any(x < 0)
    x_ = fill_gaps(x, tol)
    x_ = (x_ > 0).astype(int)
    pad = np.zeros((1))
    x_ = np.concatenate([pad, x_, pad])
    x_ = np.diff(x_)
    idx = np.arange(len(x_))
    i0 = idx[x_ == 1]
    i1 = idx[x_ == -1]
    idx = np.stack([i0,i1]).T
    if nmin is not None:
        duration = np.diff(idx, axis=-1).flatten()
        idx = idx[duration >= nmin]
    if sort:
        duration = np.diff(idx, axis=-1).flatten()
        idx = idx[np.argsort(duration)[::-1]]
    return idx


def fill_gaps(x, gap=None, fill=1):
    """
    Fills zero gaps in 1d-array.
    
    Parameters
    ----------
    x : ndarray
        1D array of non-negative numeric values
    gap : int or None, default None
        Max gap duration (if None - fill all gaps)
    fill : float, default 1
        Value to fill zeros

    Returns
    -------
    ndarray
        1D array - arrray with all gaps <= gap filled

    """
    assert x.ndim == 1 and not any(x < 0)
    x_ = np.copy(x)
    if gap is None:
        x_[~(x>0)] = fill
    elif gap > 0:
        x__ = (np.nan_to_num(x) == 0).astype(int)
        idx = find_intervals(x__)
        duration = np.diff(idx, axis=-1).flatten()
        idx = idx[duration <= gap]
        for i0, i1 in idx:
            x_[i0:i1] = fill
    return x_


def _interval_score(idx, scores, nmin):
    """
    Utility function to score likelihood of being 
    either night sleep or daytime rest / nap for each interval
    
    Parameters
    ----------
    idx : ndarray
        2D array of intervals start-end indices
    scores : ndarray
        1D array of scores that a point is daytime (close to max)
    nmin : int
        Minimum length of intervals

    Returns
    -------
    idx : ndarray
        2D array of intervals sorted descending by score
    iscore : ndarray
        1D array of likelihood each interval is a night sleep in range (0,1)
    inumber : ndarray
        1D array of interval id

    """
    lscore = 2 * nmin / np.diff(idx, axis=1).flatten()
    iscore = np.array([min(scores[i0:i1]) for (i0, i1) in idx])
    iscore = np.min(np.stack([iscore, lscore]), axis=0)
    iscore = np.clip(iscore, 0, 1)
    inumber = np.arange(len(idx))
    return idx, iscore, inumber


def _interval_dct(x, window, idx, loop):
    """
    Utility function to make dictionary of
    matching sleep / rest intervals if loop == True

    Parameters
    ----------
    x : ndarray
        Padded array of equispaced time series data
    window : int
        Window size
    idx : ndarray
        2D array of intervals start-end indices
    loop : bool, default False
        Flag to loop time series (intended to use for biobanks)

    Returns
    -------
    dict
        Dictionary of looped interval indices

    """
    dct = {}
    if loop and len(idx):
        for ax in range(2):
            d = idx[:,ax]
            d = np.stack([d] * len(d))
            d = np.abs(d - (d + x.size).T)
            mask = np.min(d, axis=1) == 0
            if np.sum(mask):
                i = np.arange(len(d))[mask]
                d = np.argmin(d, axis=1)[mask]
                dct.update(dict(zip(i, d)))
    return dct


def _empty_arrays(durations=False):    
    """
    Utility function to init empty timestamp arrays
    
    Parameters
    ----------
    durations : bool, default False
        If True - return "sleep" and "rest" as 1D-arrays

    Returns
    -------
    dict
        Empty arrays for t_onset, t_max, t_offset, t_min, sleep, rest

    """
    dct = {}
    for key in ["t_start", "t_min", "t_onset", "t_max", "t_offset", "sleep", "rest"]:
        dct[key] = np.zeros((0), dtype=int)
    if not durations:
        dct["sleep"] = np.zeros((0,2), dtype=int)
        dct["rest"] = np.zeros((0,2), dtype=int)
    return dct


def _sleep_rest_durations(dct):
    """
    Utility function to calculate durations of sleep and rest

    Parameters
    ----------
    dct : dict
        Dictionary of t_start, t_min, t_onset, t_max, t_offset, sleep, rest

    Returns
    -------
    dct : dict
        Dictionary of t_start, t_min, t_onset, t_max, t_offset, sleep, rest

    """
    t_start, t_offset = dct["t_start"], dct["t_offset"]
    sleep, rest = dct["sleep"], dct["rest"]
    n = len(t_offset)
    d_sleep = np.zeros((n))
    d_rest = np.zeros((n))
    if len(t_offset) and len(t_start):
        # Extend t_offset
        t_ext = np.concatenate([(t_start[0],), t_offset])
        if len(sleep):
            d = np.diff(sleep, axis=1).flatten()
            t = np.sum(sleep, axis=1) / 2
            for i in range(n):
                mask = (t > t_ext[i]) & (t < t_ext[i+1])
                d_sleep[i] = np.sum(d[mask])
        if len(rest):
            d = np.diff(rest, axis=1).flatten()
            t = np.sum(rest, axis=1) / 2
            for i in range(n):
                mask = (t > t_ext[i]) & (t < t_ext[i+1])
                d_rest[i] = np.sum(d[mask])
    dct["sleep"] = d_sleep.astype(int)
    dct["rest"] = d_rest.astype(int)
    return dct


def _initialize_intervals(x, tol=0, nmin=None, idx=None):
    """
    Utility function to initializes continuous positive intervals in 1d-array.
    
    Parameters
    ----------
    x : ndarray
        1D array of non-negative numeric values
    tol : int, default 0
        Gap duration tolerance
    nmin : int or None, default None
        Minimal length of intervals
    idx : ndarray or None, default None
        Optional initialization of sleep/nap intervals

    Returns
    -------
    ndarray
        2D array - N intervals x 2 indices (start, end)

    """
    idxs = find_intervals(x, tol, nmin)
    if idx is not None:
        exclude = np.zeros((len(idxs))).astype(bool)
        for i, (i0, i1) in enumerate(idxs):
            exclude[i] = np.any((idx > i0) & (idx < i1))
        idxs = np.vstack([idxs[~exclude], idx])
        i = np.argsort(idxs[:,0])
        idxs = idxs[i]
    return idxs


def _select_intervals(x, window, t_start, t_min, t_onset, t_offset, sleep, rest, loop):
    """
    Utility function to optimize overlap of intervals with the original array

    Parameters
    ----------
    x : ndarray
        Array of equispaced time series data    
    t_start : ndarray
        Indices of peak starts
    t_min : ndarray
        Indices of peak minima
    t_onset : ndarray
        Indices of peak onsets
    t_offset : ndarray
        Indices of peak offsets
    sleep : ndarray
        2D array of N intervals x 2 indices (start, end)
    rest : ndarray
        2D array of N intervals x 2 indices (start, end)
    loop : bool, default False
        Flag to loop time series (intended to use for biobanks)

    Returns
    -------
    t_start : ndarray
        Indices of peak starts
    t_min : ndarray
        Indices of peak minima
    sleep : ndarray
        2D array of N intervals x 2 indices (start, end)
    rest : ndarray
        2D array of N intervals x 2 indices (start, end)
    is_closed : bool
        True if loop == True and t_offset[-1] = t_start[0] + x.size

    """
    if len(t_start) == 0 or len(t_offset) == 0:
        # No selected days: drop intervals, they would span the whole padding
        return t_start, t_min, sleep[:0], rest[:0], False
    delta = window // 24
    is_closed = loop & (abs(t_start[0] + x.size - t_offset[-1]) <= delta)
    # Select sleep intervals
    if len(sleep):
        mask = (sleep[:,1] > t_start[0]) & (sleep[:,0] < t_offset[-1])
        sleep = sleep[mask]
    # Select rest intervals
    if len(rest):
        mask = (rest[:,1] > t_start[0]) & (rest[:,0] < t_offset[-1])
        rest = rest[mask]
    # Adjust t_start
    if loop:
        if is_closed:
            t_start[0] = t_offset[-1] - x.size
            if len(sleep):
                sleep[0,0] = t_start[0]
        else:
            if len(sleep):
                t_start[0] = sleep[0,0]
    else:
        t_start[0] = t_onset[0] - 8 * delta
        if len(sleep):
            sleep[0,0] = t_start[0]
    if len(t_min) and len(t_onset):
        t_min[0] = int((t_start[0] + t_onset[0]) // 2)
    return t_start, t_min, sleep, rest, is_closed


def _find_timestamps_and_scores(x, window=1440, level=0.5, nmin=30, 
    atol=5, rtol=1, prec=False, loop=False, durations=False, npad=2, idx=None):
    """
    Utility function to assign peak times for full (padded) array of data
    NaN, Inf values NOT allowed

    Parameters
    ----------
    x : ndarray
        1D array of equispaced binarized integer time series data
    window : int, default 1440
        Window size
    level : float, default 0.5
        Cutoff level for peak onset and offset between max and min in range 0.0 - 1.0
    nmin : int, default 30
        Minimum length of intervals
    atol : int, default 5
        Gap tolerance for activity intervals [minutes]
    rtol : int, default 1
        Gap tolerance for rest intervals [minutes]
    prec : bool, dfault False
        Flag to find precise timestamps (default False sets precision to 1 hour)
    loop : bool, default False
        Flag to loop time series (intended to use for biobanks)
    durations : bool, default False
        If True - return "sleep" and "rest" as 1D-arrays
    npad : int, default 2
        Number of windows to pad in both directions
    idx : ndarray or None, default None
        Optional initialization of sleep/nap intervals

    Returns
    -------
     dct : dict
        't_start' : 1D array of indices of peak starts
        't_min' : 1D array of indices of peak minima
        't_onset' : 1D array of indices of peak onsets
        't_max' : 1D array of indices of peak maxima
        't_offset' : 1D array of indices of peak offsets
        'sleep' : 2D array of N intervals x 2 indices (start, end)
        'rest' : 2D array of N intervals x 2 indices (start, end)
        'scores' : Scores of time point falling between onset and offset (close to max)
    is_closed : bool
        True if loop == True and t_offset[-1] = t_start[0] + x.size

    """
    # Preprocessing: Guess peaks min / max indices based on smoothed and shrinked profile
    x_ = peakts.pad(x, window, loop, npad)
    dct = peakts._find_timestamps_and_scores(x_, window, level, prec, npad=0)
    t_min, t_onset, t_max = dct["t_min"], dct["t_onset"], dct["t_max"]
    t_offset, scores = dct["t_offset"], dct["scores"]
    # Prepcossing: Find and fill activity intervals
    epsilon = 1e-5
    for i0, i1 in find_intervals(x_ > epsilon, atol):
        x_[i0:i1] = 1
    # Find candidate rest / sleep interval indices
    idx = _initialize_intervals(x_ < epsilon, rtol, nmin, idx)
    # Return empty arrays in no peak max or no intervals were found
    if len(t_max) == 0 or len(idx) == 0:
        dct = _empty_arrays(durations)
        return dct, False
    # Otherwise adjust t_onset / t_offset to start / end indices of rest / sleep intervals
    idx, iscore, inumber = _interval_score(idx, scores, nmin)
    dct = _interval_dct(x, window, idx, loop)
    ext_max = np.concatenate([(-1,), t_max, (len(x_),)])
    ext_onset = np.concatenate([t_onset, (len(x_) - 1,)])
    ext_offset = np.concatenate([(0,), t_offset])
    # Keep peak-based onset / offset as fallback for nights without sleep intervals
    peak_onset = np.asarray(t_onset, dtype=float)
    peak_offset = np.asarray(t_offset, dtype=float)
    t_onset = np.zeros((len(t_onset))) * np.nan
    t_offset = np.zeros((len(t_offset))) * np.nan
    sleep = np.zeros((len(idx))).astype(bool)
    for k in range(len(ext_onset)):
        mask = (idx[:,1] > ext_offset[k]) & (idx[:,0] < ext_onset[k])
        mask = mask & ((idx[:,0]>ext_max[k]) | (idx[:,1]<ext_max[k+1]))
        if np.sum(mask):
            t0, t1 = np.nan, np.nan
            for i in inumber[mask]:
                if sleep[i] or iscore[i] < level:
                    sleep[i] = True
                    if i in dct:
                        sleep[dct[i]] = True
                    t0 = np.nanmin((t0, idx[i,0]))
                    t1 = np.nanmax((t1, idx[i,1]))
            if k < len(t_onset):
                t_onset[k] = t1
            if k > 0:
                t_offset[k-1] = t0
    rest = idx[~sleep]
    sleep = idx[sleep]
    # Close loop: the trailing pad may have no night after the last peak
    t_offset = np.asarray(t_offset, dtype=float)
    t_onset = np.asarray(t_onset, dtype=float)
    if loop and len(t_offset) and np.isnan(t_offset[-1]):
        finite = t_offset[np.isfinite(t_offset)]
        if len(finite):
            t_offset[-1] = finite[-1] + window
    # Nights without a qualifying sleep interval keep the peak-based onset / offset
    missing = np.isnan(t_onset)
    t_onset[missing] = peak_onset[missing]
    missing = np.isnan(t_offset)
    t_offset[missing] = peak_offset[missing]
    t_onset, t_offset = t_onset.astype(int), t_offset.astype(int)
    # Select timestamps by optimal overlap with original data array
    t_start, t_min, t_onset, t_max, t_offset = peakts._select_timestamps(x, 
        t_min, t_onset, t_max, t_offset, window, loop, npad)
    # Select intervals by optimal overlap with original data array
    t_start, t_min, sleep, rest, is_closed = _select_intervals(x, 
        window, t_start, t_min, t_onset, t_offset, sleep, rest, loop)
    dct = {"t_start": t_start, "t_min": t_min, "t_onset": t_onset, "t_max": t_max, 
           "t_offset": t_offset, "sleep": sleep, "rest": rest, "scores": scores}
    return dct, is_closed


def _timestamps_and_scores(x, window=1440, level=0.5, nmin=30, atol=5, rtol=1, prec=False, loop=False, durations=False):
    """
    Finds peak timestamps for array of time series data

    Parameters
    ----------
    x : ndarray
        1D array of equispaced binarized integer time series data
    window : int, default 1440
        Window size, positive int
    level : float, default 0.5
        Cutoff level for peak onset and offset between max and min in range 0.0 - 1.0
    nmin : int, default 30
        Minimum length of intervals, positive int
    atol : int, default 5
        Gap tolerance for activity intervals, non-negative int
    rtol : int, default 1
        Gap tolerance for rest intervals, non-negative int
    prec : bool, dfault False
        Flag to find precise timestamps (default False sets precision to 1 hour)
    loop : bool, default False
        Flag to loop time series (intended to use for biobanks)
    durations : bool, default False
        If True - return "sleep" and "rest" as 1D-arrays

    Returns
    -------
     dict
        't_start' : 1D array of indices of peak starts
        't_min' : 1D array of indices of peak minima
        't_onset' : 1D array of indices of peak onsets
        't_max' : 1D array of indices of peak maxima
        't_offset' : 1D array of indices of peak offsets
        'sleep' : 2D array of N intervals x 2 indices (start, end)
        'rest' : 2D array of N intervals x 2 indices (start, end)
        'scores' : Scores of time point falling between onset and offset (close to max)

    """
    npad = 2
    dct, is_closed = _find_timestamps_and_scores(
        x, window, level, nmin, atol, rtol, prec, loop,
        durations=durations, npad=npad)
    # In some cases rest intervals may be very large and require increased padding by full weeks:
    if loop and not is_closed:
        npad = max(npad, x.size // window)
        dct, is_closed = _find_timestamps_and_scores(
            x, window, level, nmin, atol, rtol, prec, loop,
            durations=durations, npad=npad)
    # Shift timestamps to original (non-padded) data sample
    dct = peakts._shift_timestamps(dct, window, npad)
    # If durations == True, replace sleep / rest indices with durations
    if durations:
        dct = _sleep_rest_durations(dct)
    return dct


def _timestamps(x, window=1440, level=0.5, nmin=30, atol=5, rtol=1, prec=False, loop=False, durations=False):
    """
    Finds peak timestamps for array of time series data

    Parameters
    ----------
    x : ndarray
        1D array of equispaced binarized integer time series data
    window : int, default 1440
        Window size, positive int
    level : float, default 0.5
        Cutoff level for peak onset and offset between max and min in range 0.0 - 1.0
    nmin : int, default 30
        Minimum length of intervals, positive int
    atol : int, default 5
        Gap tolerance for activity intervals, non-negative int
    rtol : int, default 1
        Gap tolerance for rest intervals, non-negative int
    prec : bool, dfault False
        Flag to find precise timestamps (default False sets precision to 1 hour)
    loop : bool, default False
        Flag to loop time series (intended to use for biobanks)
    durations : bool, default False
        If True - return "sleep" and "rest" as 1D-arrays

    Returns
    -------
     dict
        't_start' : 1D array of indices of peak starts
        't_min' : 1D array of indices of peak minima
        't_onset' : 1D array of indices of peak onsets
        't_max' : 1D array of indices of peak maxima
        't_offset' : 1D array of indices of peak offsets
        'sleep' : 2D array of N intervals x 2 indices (start, end)
        'rest' : 2D array of N intervals x 2 indices (start, end)

    """
    dct = _timestamps_and_scores(
        x, window, level, nmin, atol, rtol, prec, loop, durations)
    dct = {k: v for k, v in dct.items() if k != "scores"}
    return dct


def _timestamps_biobank(x, window=1440, level=0.5, nmin=30, atol=5, rtol=1, prec=False):
    """
    Finds peak timestamps for arrays of time series data

    Notes
    -----
    - Input "x" as binarized (1 - active / 0 - inactive) steps, bpm, sleep
    - Loop = True to keep edge start / end days of short samples (<= 7 days, typically from biobanks)
    - Durations = True to keep dimensions of all arrays = 2

    Parameters
    ----------
    x : ndarray
        2D array of samples x equispaced binarized integer time series data
    window : int, default 1440
        Window size, positive int
    level : float, default 0.5
        Cutoff level for peak onset and offset between max and min in range 0.0 - 1.0
    nmin : int, default 30
        Minimum length of intervals, positive int
    atol : int, default 5
        Gap tolerance for activity intervals, non-negative int
    rtol : int, default 1
        Gap tolerance for rest intervals, non-negative int
    prec : bool, dfault False
        Flag to find precise timestamps (default False sets precision to 1 hour)

    Returns
    -------
    dict
        't_start' : 2D array of N samples x indices of peak starts
        't_min' : 2D array of N samples x indices of peak minima
        't_onset' : 2D array of N samples x indices of peak onsets
        't_max' : 2D array of N samples x indices of peak maxima
        't_offset' : 2D array of N samples x indices of peak offsets
        'sleep' : 2D array of N samples x sleep durations
        'rest' : 2D array of N samples x rest durations

    """
    dct, dcts = {}, []
    for x_ in tqdm(x):
        dcts.append(_timestamps(x_, window, level, nmin, 
            atol, rtol, prec, loop=True, durations=True))
    for key in ["t_start", "t_min", "t_onset", "t_max", "t_offset", "sleep", "rest"]:
        dct[key] = np.stack([peakts.pad_timestamps(d[key]) for d in dcts])
    return dct


# Functions with sanity checks for input parameters and data
timestamps_and_scores = checks.sanity_check(_timestamps_and_scores, intervals=True)
timestamps_biobank = checks.sanity_check(_timestamps_biobank, intervals=True)
timestamps = checks.sanity_check(_timestamps, intervals=True)






