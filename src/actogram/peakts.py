#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
from tqdm import tqdm
from . import checks

"""
Copyright © 2025 Actogram LTD (https://actogram.ai)

This module (peak time stamps) contains functions to extract circadian 
rhythm timestamps based on ACTIVITY DENSITY PEAK DETECTION.
"""


def pad(x, window, loop=False, npad=2):
    """
    Expand data to match integer number of window repeats

    Parameters
    ----------
    x : ndarray
        Array of equispaced time series data
    window : int
        Window size
    loop : bool, default False
        Flag to loop time series (intended to use for biobanks)
    npad : int, default 2
        Number of windows to pad in both directions

    Returns
    -------
    ndarray
        1D array of expanded values, prepended by two-window length

    """
    x_ = x.flatten()
    n = int(window)
    n1 = npad * n
    n2 = n1 + np.sign(x.size % n) * (n - x.size % n)
    x1 = x_[-n1:] if loop else np.zeros((n1))
    x2 = x_[:n2] if loop else np.zeros((n2))
    x_ = np.concatenate([x1, x_, x2])
    return x_


def _slice_values(x, window):
    """
    Utility function to represent input data as array of 
    window-length slices with stride = window / 2

    Parameters
    ----------
    x : ndarray
        Array of equispaced time series data
    window : int
        Window size

    Returns
    -------
    ndarray
        Same as numpy.lib.stride_tricks.as_strided with stride  = window / 2

    """
    n = window * (x.size // window)
    x_ = x.flatten()
    if x_.size > window:
        x_ = np.stack([x_[:n], np.roll(x_, -(window//2))[:n]])
        x_ = np.split(x_, x_.shape[1]//window, 1)
        x_ = np.array(x_).reshape(-1, window)
        x_ = x_[:-1] if x.size % window == 0 else x_
    return x_


def _remove_nonlocal_peaks(x, idx, window):
    """
    Utility function to remove not true max values 
    which occur at the edges of window-length slices

    Parameters
    ----------
    x : ndarray
        Array of equispaced time series data
    idx : ndarray
        Indices of peaks
    window : int,
        Window size

    Returns
    -------
    ndarray
        Indices of true local peaks

    """
    def is_max(val, arr):
        return val == arr.max() and val > arr.min()
    idxs = []
    x_ = x.flatten()
    n = len(x_)
    for i in np.unique(idx):
        if len(idxs) > 0 and idxs[-1] > i - window:
            continue
        i0 = max(0, i - window // 2)
        i1 = min(n, i + window // 2)
        if (i > 0) & (i < n) & \
           is_max(x_[i], x_[i0:i+1]) & is_max(x_[i], x_[i:i1]):
            idxs.append(i)
    idxs = np.array(idxs).astype(int)
    return idxs


def smoother(x, window, epsilon=1e-5):
    """
    Smooth data using running average by bell-shaped Hann window

    Parameters
    ----------
    x : ndarray
        Array of equispaced time series data
    window : int,
        Window size
    epsilon : float, default 1e-5
        Set values below epsilon to zero

    Returns
    -------
    ndarray
        Smoothed time series data (of same size as input)

    """
    hann = np.hanning(window)
    hann = hann / np.sum(hann)
    w = np.correlate(x.flatten(), hann, mode='same')
    w[np.abs(w) < epsilon] = 0
    return w.reshape(x.shape)


def find_plateaus(x,  window, smooth=False, epsilon=1e-5):
    """
    Finds plateaus; Min distance between plateaus = 2 / 3 of window

    Notes
    -----
    - Needs smoothed data as input; if not - use flag smooth=True
    - NaN or Inf values NOT allowed

    Parameters
    ----------
    x : ndarray
        Array of smoothed equispaced time series data
    window : int,
        Window size
    smooth : bool, default False
        If True - apply Hann window averaging smooth
    epsilon : float, default 1e-5
        Set values below epsilon to zero

    Returns
    -------
    ndarray
        2D array of start-end indices of plateaus

    """
    n = len(x)
    # Find intervals of constant value
    diff = np.diff(x)
    diff = (diff == 0).astype(int)
    diff = np.diff(np.pad(diff, (1,1)))
    i = np.arange(len(diff))
    i0, i1 = i[diff == 1], i[diff == -1]
    idxs = np.stack([i0,i1]).T
    # Mask to keep only local max value plateaus
    mask = np.ones((len(idxs))).astype(bool)
    for i, (i0, i1) in enumerate(idxs):
        j0 = max(0, i0 - window // 3)
        j1 = min(n, i1 + window // 3)
        xmax = x[j0:j1].max()
        if xmax < epsilon or x[i0] < xmax:
           mask[i] = False
    idxs = idxs[mask]
    return idxs


def _find_peaks_and_plateaus(x, window, smooth=False):
    """
    Utility function to find peaks and plateaus; 
    Min distance between peaks and plateau endges = 2 / 3 of window

    Notes
    -----
    - Needs smoothed data as input; if not - use flag smooth=True
    - NaN or Inf values NOT allowed

    Parameters
    ----------
    x : ndarray
        Array of smoothed equispaced time series data
    window : int
        Window size; 2 / 3 of window will be applied
    smooth : bool, default False
        If True - apply Hann window averaging smooth

    Returns
    -------
    idx : ndarray
        1D array of peak indices
    idxs : ndarray
        2D array of start-end indices of plateaus

    """
    n = 2 * int(window) // 3
    # Smooth using window
    x_ = smoother(x, window) if smooth else x
    # Find peaks using 2 / 3 of window
    xs = _slice_values(x_, window=n)
    idx = np.argmax(xs, axis=1)
    idx = idx + np.arange(len(xs)) * (n // 2)
    idx = _remove_nonlocal_peaks(x_, idx, window=n)
    # Find plateaus using 2 / 3 of window; applied inside find_plateaus()
    idxs = find_plateaus(x_, window)
    # Update peaks with plateaus' midpoints
    idxm = np.sum(idxs, axis=1) // 2
    idx = np.sort(np.concatenate([idxm, idx]))
    # Short plateaus were detected as peaks - remove these
    mask = np.ones((len(idx))).astype(bool)
    for k, i in enumerate(idx):
        diff = np.abs(idx - i)
        if k > 0 and diff[:k].min() < window // 3:
            mask[k] = False
    idx = idx[mask]
    return idx, idxs


def find_peaks(x, window, smooth=False):
    """
    Finds peaks; Min distance between peaks = 2 / 3 of window

    Notes
    -----
    - Needs smoothed data as input; if not - use flag smooth=True
    - NaN or Inf values NOT allowed

    Parameters
    ----------
    x : ndarray
        Array of smoothed equispaced time series data
    window : int
        Window size
    smooth : bool, default False
        If True - apply Hann window averaging smooth

    Returns
    -------
    idx : ndarray
        1D array of peak indices

    """
    idx, idxs = _find_peaks_and_plateaus(x, window, smooth)
    return idx


def _normalize_scores(x, t_max, t_plateau, epsilon=1e-5):
    """
    Utility function to normalize smoothed array of data 
    between each pair of peaks to range 0 - 1 
    
    Parameters
    ----------
    x : ndarray
        Array of equispaced time series data
    t_max : ndarray
        1D array of peak indices
    t_plateau : ndarray
        2D array of start-end indices of plateaus
    epsilon : float, default 1e-5
        Set values below epsilon to zero

    Returns
    -------
    scores : ndarray
        Normalized 0 - 1 range scores for each data point

    """
    t = t_plateau.flatten()
    t = np.concatenate([t, t_max])
    t = np.unique(t).astype(int)
    scores = np.zeros_like(x)
    if len(t):
        scores = np.pad(x, (1,1), constant_values=(x[t[0]], x[t[-1]])) + 1
        i = np.pad(t + 1, (1,1), constant_values=(0, x.size + 1))
        d = np.interp(np.arange(len(scores)), i, scores[i])
        scores = np.min(np.stack([scores, d]), axis=0)
        scores = scores / d
        idx = np.stack([i[:-1], i[1:]]).T
        for i0, i1 in idx:
            if (1 - np.min(scores[i0:i1])) > epsilon:
                scores[i0:i1] -= scores[i0:i1].min()
            if scores[i0] > 0:
                scores[i0:i1] /= scores[i0]
        scores = scores[1:-1]
    return scores


def _find_minima(scores, t_max, window):
    """
    Utility function to find minima between each pair of peaks

    Parameters
    ----------
    scores : ndarray
        Normalized 0 - 1 range scores for each data point
    t_max : ndarray
        1D array of peak indices
    window : int
        Window size

    Returns
    -------
    t_min : ndarray
        Indices of peak minima

    """
    n = len(scores)
    x = np.copy(scores)
    # Extend t_max
    t1 = 0 if t_max[0] > window else t_max[0]
    t2 = n if t_max[-1] < n - window else t_max[-1]
    t_ext = np.concatenate([(t1,), t_max, (t2,)])
    t_ext = np.unique(t_ext.astype(int))
    # Find last minimum
    i0, i1 = t_ext[-2], t_ext[-1]
    t0 = i0 + np.argmin(x[i0:i1]) + 1
    # Find other minima in reversed arrays
    x = x[::-1]
    t_ext = n - t_ext[::-1]
    idx = zip(t_ext[:-1], t_ext[1:])
    t_min = np.array([i0 + np.argmin(x[i0:i1]) for (i0, i1) in idx])
    t_min = n - t_min[::-1]
    # Replace last minimum if last max index == n (first max in reversed array)
    if t_ext[0] == 0:
        t_min[-1] = t0
    return t_min


def _empty_arrays():
    """
    Utility function to init empty timestamp arrays
    """
    t_start = np.zeros((0), dtype=int)
    t_min = np.zeros((0), dtype=int)
    t_onset = np.zeros((0), dtype=int)
    t_max = np.zeros((0), dtype=int)
    t_offset = np.zeros((0), dtype=int)
    return t_start, t_min, t_onset, t_max, t_offset


def _add_tmin(t_min, t_onset, t_max, t_offset, n, window, loop):
    """
    Utility function to add missing t_min timestamps
    """
    if loop:
        i0, i1 = t_max[0], t_max[0] + n
        mask = (t_min > i0) & (t_min < i1)
        if np.sum(mask):
            t0 = t_min[mask][-1] - n
            t_min = np.concatenate([(t0,), t_min])
        else:
            t_onset = t_onset[1:]
            t_max = t_max[1:]
            t_offset = t_offset[1:]
    else:
        t0 = t_onset[0] - window // 6
        t_min = np.concatenate([(t0,), t_min])
    return t_min, t_onset, t_max, t_offset


def _add_tstart(t_min, t_offset, n, window, loop):
    """
    Utility function to add t_start timestamps
    """
    t_start = np.zeros((0), dtype=int)
    if len(t_offset) == 0:
        return t_start
    t_start = np.roll(t_offset, 1)
    # Find first t_start
    t0 = t_min[0] - window // 6
    if loop and t0 > t_offset[0]:
        mask = t_offset < t_min[0] + n
        if np.sum(mask):
            t0 = t_offset[mask][-1] - n
        else:
            t0 = t_offset[0] - window
    # replace first t_start
    t_start[0] = t0
    t_start = t_start.astype(int)
    return t_start


def _select_timestamps(x, t_min, t_onset, t_max, t_offset, window, loop=False, npad=2):
    """
    Utility function to optimize overlap of peaks with the original array

    Parameters
    ----------
    x : ndarray
        Array of equispaced time series data    
    t_min : ndarray
        Indices of peak minima
    t_onset : ndarray
        Indices of peak onsets
    t_max : ndarray
        Indices of peak maxima
    t_offset : ndarray
        Indices of peak offsets
    window : int
        Window size
    loop : bool, default False
        Flag to loop time series (intended to use for biobanks)
    npad : int, default 2
        Number of windows to pad in both directions

    Returns
    -------
    ndarrays
        Indices of onsets, maxima, onsets, minima

    """
    # Length of original array
    n = len(x)
    # If no peaks found, return empty arrays
    if len(t_min) == 0 or len(t_max) == 0:
        t_start, t_min, t_onset, t_max, t_offset = _empty_arrays()
        return t_start, t_min, t_onset, t_max, t_offset
    # Add missing t_min before peak t_max
    if t_min[0] > t_max[0]:
        t_min, t_onset, t_max, t_offset = _add_tmin(
            t_min, t_onset, t_max, t_offset, n, window, loop)
    if len(t_min) == 0 or len(t_max) == 0:
        t_start, t_min, t_onset, t_max, t_offset = _empty_arrays()
        return t_start, t_min, t_onset, t_max, t_offset
    # Add start points before t_min
    t_start = _add_tstart(t_min, t_offset, n, window, loop)
    # Drop extra troughs that do not belong to a selected day
    if len(t_min) > len(t_start) and (len(t_start) == 0 or t_min[0] < t_start[0]):
        t_min = t_min[1:]
    if len(t_min) > len(t_offset) and (len(t_offset) == 0 or t_min[-1] > t_offset[-1]):
        t_min = t_min[:-1]
    # If loop, select days with best overlap with calendar days
    if loop:
        delta = window // 24
        t0, t1, overlap = npad * window, npad * window + n, 0
        mask = np.zeros((len(t_start))).astype(bool)
        for t in t_start[t_start < 2 * npad * window]:
            mask_ = (t_start >= t) & (t_offset <= t + n + delta)
            if not np.any(mask_):
                continue
            overlap_ = min(t1, t_offset[mask_][-1]) - max(t0, t_start[mask_][0])
            if overlap_ > overlap:
                overlap = overlap_
                mask = (t_start >= t) & (t_start < t + n - delta)
        n_align = min(len(t_start), len(t_min), len(t_onset), len(t_max),
                      len(t_offset), len(mask))
        t_start = t_start[:n_align]
        t_min = t_min[:n_align]
        t_onset = t_onset[:n_align]
        t_max = t_max[:n_align]
        t_offset = t_offset[:n_align]
        mask = mask[:n_align]
        t_start = t_start[mask]
        t_min = t_min[mask]
        t_onset = t_onset[mask]
        t_max = t_max[mask]
        t_offset = t_offset[mask]
    return t_start, t_min, t_onset, t_max, t_offset


def _shift_timestamps(dct, window, npad=2):
    """
    Utility function to shift timestamps to original (non-padded) data sample

    Parameters
    ----------
    dct : dict
        Dictionary of t_start, t_min, t_onset, t_max, t_offset
        and optionally sleep and rest interval indices
    window : int
        Window size
    npad : int, default 2
        Number of windows to pad in both directions

    Returns
    -------
    dct : dict
        Dictionary of t_start, t_min, t_onset, t_max, t_offset
        and optionally sleep and rest interval indices

    """
    n = npad * window
    for key in ["t_start", "t_min", "t_onset", "t_max", "t_offset", "sleep", "rest"]:
        if key in dct:
            dct[key] = (dct[key] - n).astype(int)
    return dct


def pad_timestamps(x, nmax=10):
    """
    Pad array to fixed length with NaN
    Useful to store indices of various length for each sample of a biobank

    Parameters
    ----------
    x : ndarray
        Array of equispaced time series data
    nmax : int, default=10
        Max num of indices to store

    Returns
    -------
    ndarray
        Padded array of indices

    """
    x_ = np.round(x.astype(float), 2)
    x_ = np.pad(x_, (0,nmax), constant_values=np.nan)[:nmax]
    return x_


def _downsample(x, window=1440, prec=False):
    """
    Utility function to downsample array to speed up calculations
    Shrink is applied if precise == False and values/hour h > 1
    """
    h = window // 24
    x_, win = np.copy(x), window 
    if not prec and h > 1:
        win = window // h
        x_ = x.reshape(-1,h)
        x_ = np.mean(x_, axis=1)
    return x_, win


def _upsample(t_min, t_onset, t_max, t_offset, scores, x, window=1440):
    """
    Utility function to upsample arrays back if shrink was applied
    """
    h = window // 24
    if len(scores) < len(x):
        scores = np.repeat(scores, h)
        t_min = t_min * h - h // 2 # '-' because min defined for reversed array
        t_max = t_max * h + h // 2
        t_onset = t_onset * h + h // 2
        t_offset = t_offset * h + h // 2
    return t_min, t_onset, t_max, t_offset, scores


def _find_timestamps_and_scores(x, window=1440, level=0.5, prec=False, loop=False, npad=2):
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
    prec : bool, dfault False
        Flag to find precise timestamps (default False sets precision to 1 hour)
    loop : bool, default False
        Flag to loop time series (intended to use for biobanks)
    npad : int, default 2
        Number of windows to pad in both directions

    Returns
    -------
    dict
        't_start' : 1D array of indices of peak starts
        't_min' : 1D array of indices of peak minima
        't_onset' : 1D array of indices of peak onsets
        't_max' : 1D array of indices of peak maxima
        't_offset' : 1D array of indices of peak offsets
        'scores' : Scores of time point falling between onset and offset (close to max)

    """
    # Initialize empty arrays
    t_start, t_min, t_onset, t_max, t_offset = _empty_arrays()
    # Downsample to speedup if not required precise timestamps
    x_ = pad(x, window, loop, npad)
    x_, win = _downsample(x_, window, prec)
    w = smoother(x_, win)
    # Detect peaks t_max and plateaus
    t_max, t_plateau = _find_peaks_and_plateaus(w, win)
    scores = _normalize_scores(w, t_max, t_plateau)
    # Detect t_min, t_onset, t_offset
    if len(t_max):
        t_min = _find_minima(scores, t_max, win)
        t_onset = np.zeros((len(t_max))).astype(int)
        t_offset = np.zeros((len(t_max))).astype(int)
        t = np.arange(len(w))
        for i, tx in enumerate(t_max):
            t0 = t[(t < tx) & (scores < level)]
            t_onset[i] = max(t0) if len(t0) else tx
            t1 = t[(t > tx) & (scores < level)]
            t_offset[i] = min(t1) if len(t1) else tx
    # Upsample back if downsample was applied
    t_min, t_onset, t_max, t_offset, scores = _upsample(t_min, 
        t_onset, t_max, t_offset, scores, x, window)
    # Select timestamps by optimal overlap with original data array
    t_start, t_min, t_onset, t_max, t_offset = _select_timestamps(x, 
        t_min, t_onset, t_max, t_offset, window, loop, npad)
    dct = {"t_start": t_start, "t_min": t_min, "t_onset": t_onset, 
           "t_max": t_max, "t_offset": t_offset, "scores": scores}
    return dct


def _timestamps_and_scores(x, window=1440, level=0.5, prec=False, loop=False):
    """
    Finds peak timestamps for array of time series data

    Notes
    -----
    - Input "x" as binarized (1 - active / 0 - inactive) steps, bpm, sleep
    - Use loop=True only for short samples (<= 7 days, typically from biobanks)

    Parameters
    ----------
    x : ndarray
        1D array of equispaced binarized integer time series data
    window : int, default 1440
        Window size, positive int
    level : float, default 0.5
        Cutoff level for peak onset and offset between max and min in range 0.0 - 1.0
    prec : bool, dfault False
        Flag to find precise timestamps (default False sets precision to 1 hour)
    loop : bool, default False
        Flag to loop time series (intended to use for biobanks)

    Returns
    -------
    dict
        't_start' : 1D array of indices of peak starts
        't_min' : 1D array of indices of peak minima
        't_onset' : 1D array of indices of peak onsets
        't_max' : 1D array of indices of peak maxima
        't_offset' : 1D array of indices of peak offsets
        'scores' : Scores of time point falling between onset and offset (close to max)

    """
    npad = 2
    dct = _find_timestamps_and_scores(x, window, level, prec, loop, npad)
    dct = _shift_timestamps(dct, window, npad)
    return dct


def _timestamps(x, window=1440, level=0.5, prec=False, loop=False):
    """
    Finds peak timestamps for array of time series data

    Notes
    -----
    - Input "x" as binarized (1 - active / 0 - inactive) steps, bpm, sleep
    - Use loop=True only for short samples (<= 7 days, typically from biobanks)

    Parameters
    ----------
    x : ndarray
        1D array of equispaced binarized integer time series data
    window : int, default 1440
        Window size, positive int
    level : float, default 0.5
        Cutoff level for peak onset and offset between max and min in range 0.0 - 1.0
    prec : bool, dfault False
        Flag to find precise timestamps (default False sets precision to 1 hour)
    loop : bool, default False
        Flag to loop time series (intended to use for biobanks)

    Returns
    -------
    dict
        't_start' : 1D array of indices of peak starts
        't_min' : 1D array of indices of peak minima
        't_onset' : 1D array of indices of peak onsets
        't_max' : 1D array of indices of peak maxima
        't_offset' : 1D array of indices of peak offsets

    """
    dct = _timestamps_and_scores(x, window, level, prec, loop)
    dct = {k: v for k, v in dct.items() if k != "scores"}
    return dct


def _timestamps_biobank(x, window=1440, level=0.5, prec=False):
    """
    Finds peak timestamps for arrays of time series data for biobank;

    Notes
    -----
    - Input "x" as binarized (1 - active / 0 - inactive) steps, bpm, sleep
    - Loop = True to keep edge start / end days of short samples (<= 7 days, typically from biobanks)

    Parameters
    ----------
    x : ndarray
        2D array of samples x equispaced binarized integer time series data
    window : int, default 1440
        Window size, positive int
    level : float, default 0.5
        Cutoff level for peak onset and offset between max and min in range 0.0 - 1.0
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

    """
    dct, dcts = {}, []
    for x_ in tqdm(x):
        dcts.append(_timestamps(x_, window, level, prec, loop=True))
    for key in ["t_start", "t_min", "t_onset", "t_max", "t_offset"]:
        dct[key] = np.stack([pad_timestamps(d[key]) for d in dcts])
    return dct


# Functions with sanity checks for input parameters and data
timestamps_and_scores = checks.sanity_check(_timestamps_and_scores)
timestamps_biobank = checks.sanity_check(_timestamps_biobank)
timestamps = checks.sanity_check(_timestamps)







