#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import pylab as plt

"""
Copyright © 2025 Actogram LTD (https://actogram.ai)

This module contains utility functions to visualize
analysis of circadian rhythm parameters.
"""


def plot_week(x, dct, rate=1, title=None, show_pad=True, loop=False):
    """
    Plot timestamps detected by actogram for a one week sample

    Parameters
    ----------
    x : ndarray
        1D array of equispaced one-week activity time series data
    dct : dict
        Output of timestamps_and_scores() or timestamps() from peakts
        or intervalts
    rate : int, default 1
        Sampling rate in range 1 - 60:
        1 - value / min; 1440 values per day
        60 - value / hour; 24 values per day
    title : str or None, default None
        Title for matplotlib image
    show_pad : bool, default True
        Flag to show or hide padding used by timestamps functions
    loop : bool, default False
        If True, wrap the series into the plot padding (biobank samples)

    """
    # Pad
    pad = 2 * 1440 // rate
    week = x.size // rate
    n = len(dct["scores"]) if "scores" in  dct else 2 * pad + week
    x_ = np.zeros((n))
    if isinstance(x, list) or isinstance(x, np.ndarray):
        m = len(x)
        x_[pad:pad+m] = np.asarray(x).astype(float)
        if loop and m > pad:
            x_[:pad] = x[-pad:]
            x_[m+pad:] = x[:pad]
    # dummy arrays
    t = np.arange(n)  # time
    z = np.zeros((n)) # zeros
    # xticks
    nday = int(rate * n / 1440) + 1
    xticks = [
        np.arange(nday) * 1440 // rate,
        np.arange(nday, dtype=int) - 2,
    ]
    color = {
        "t_start": "red",  
        "t_min": "blue",
        "t_onset": "green",  
        "t_max": "cyan",
        "t_offset": "red", 
        "sleep": "magenta",    
        "rest": "yellow",
        "activity": "grey",
        "scores": "grey",
    }
    
    label_dct = {
        "t_onset": "Wake up",      "t_max": "Max activity",
        "t_offset": "Go to sleep", "t_min": "Min activity",
    }
    
    show = len(plt.get_fignums()) == 0
    if show:
        fig = plt.figure(figsize=(24,6), facecolor="w")
    plt.title(title, fontsize=24)
    for key in ["sleep", "rest"]:
        if key in dct and dct[key].ndim == 2:
            label = key.capitalize()
            idx = dct[key] + pad
            h = np.zeros((n))
            for j0, j1 in idx:
                h[j0:j1] = 0.5
            plt.fill_between(t, z, h, color=color[key], label=label, alpha=0.55)

    # Plot activity
    plt.fill_between(t, z, x_, color=color["activity"], alpha=0.2)
    plt.plot(t, x_, color=color["activity"], lw=2, alpha=0.5)
    # Plot scores
    s = dct["scores"] if "scores" in dct else np.zeros((n))
    plt.plot(t, s, color=color["scores"], lw=8)
    # PLot timestamps
    zorder = max([ch.zorder for ch in plt.gca().get_children()]) + 1
    for key in ["t_start", "t_min", "t_onset", "t_max", "t_offset"]:
        ts = np.copy(dct[key])
        ts = ts[np.isfinite(ts)].astype(int)+ pad
        ts = ts[(ts >= 0) & (ts < n)]
        if key in label_dct:
            plt.scatter(ts, s[ts], s=500, color=color[key], 
                        label=label_dct[key], zorder=zorder)
        else:
            plt.scatter(ts, s[ts], s=500, color=color[key], zorder=zorder)
    plt.legend(loc="upper right", ncol=2, fontsize=24)
    plt.xticks(xticks[0], xticks[1], fontsize=24)
    plt.yticks([])
    plt.ylabel("Activity", fontsize=24)
    plt.xlabel("Day", fontsize=24)
    plt.ylim(-0.1, 2)
    plt.xlim(pad, n - pad)
    if show_pad:
        plt.xlim(0, n)
    plt.tight_layout()
    if show:
        plt.show()
    return




