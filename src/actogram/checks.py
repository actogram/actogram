#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import functools
import warnings

"""
Copyright © 2025 Actogram LTD (https://actogram.ai)

This module contains decorator functions for
sanity checks of input parameters and data.
"""


def sanity_check(func, intervals=False, *args, **kwargs):
    """
    Sanity checks decorator for input parameters and data

    Parameters
    ----------
    func : python function object
        Timestamps calculating function
    intervals : bool, default False
        Flag to indicate if wrapper is used in intervalts module
    *args
        Positional arguments
    **kwargs
        Keyword arguments

    Returns
    -------
    python function object
        Decorated function

    """
    @functools.wraps(func)
    def wrapper_decorator(x, *args, **kwargs):
        key = ["window", "level", "prec", "loop"]
        if intervals:
            key = ["window", "level", "nmin", "atol", 
                   "rtol", "prec", "loop", "durations"]
        # Convert positional arguments to keyword arguments
        for i, arg in enumerate(args):
            kwargs[key[i]] = arg
        # for k, v in kwargs.items():
        #     print(k, v)

        # Check smoothing "window"
        if  "window" in kwargs:
            window = kwargs["window"]
            if not isinstance(window, int):
                try:
                    window = int(window)
                except:
                    warnings.warn(f"WARNING: 'window' should be positive int, but '{window}' was provided. "
                                   "Setting to default value window = 1440.", stacklevel=2)
                    window = 1440
            if window < 1:
                warnings.warn(f"WARNING: 'window' should be positive int, but '{window}' was provided. "
                               "Setting to default value window = 1440.", stacklevel=2)
                window = 1440
            if window < 24 or window > 1440:
                warnings.warn(f"WARNING: Optimal 'window' is one day (24 for hourly sampling, 1440 for minutely sampling). "
                               "Make sure the provided window = {window} is the intended use.", stacklevel=2)
            kwargs["window"] = window

        # Check: x is 1-dimensional array or list
        if not isinstance(x, list) and not isinstance(x, np.ndarray):
            warnings.warn(f"WARNING: 'x' should be array or list, but '{type(x)}' was provided. "
                           "Setting to x = np.zeros((7 * window)).", stacklevel=2)
            x = np.zeros((7 * window), dtype=int)
        x = np.asarray(x)
        if x.ndim > 2:
            warnings.warn(f"WARNING: 'x' should have 1 dimension, but x of dimension '{x.ndim}' was provided. "
                           "Make sure this is the intended use.", stacklevel=2)
        # Check x is or can be converted to int
        if not np.issubdtype(x.dtype, float) and not np.issubdtype(x.dtype, int):
            try:
                xdtype = x.dtype
                x = x.astype(float)
                mask = np.isfinite(x)
                if np.sum(mask) < x.size:
                    warnings.warn(f"WARNING: 'x' should have finite values, but NaN or Inf were provided. "
                                   "Setting them to zeros.", stacklevel=2)
                    x[~mask] = 0
                warnings.warn(f"WARNING: 'x' should be binary int, but '{xdtype}' was provided. "
                               "Setting to x = x.astype(int).", stacklevel=2)
                x = x.astype(int)
            except:
                warnings.warn(f"WARNING: 'x' data type should be binary int, but '{x.dtype}' was provided. "
                               "Setting to x = np.zeros((7 * window)).", stacklevel=2)
                x = np.zeros((7 * window), dtype=int)
        # Check x is binary int            
        xvalues = np.unique(x)
        if len(xvalues) and xvalues.min() < 0:
            warnings.warn(f"WARNING: 'x' should be binary int, but '{xvalues.min()}' values were provided. "
                           "Binarizing to x = (x > 0).astype(int).", stacklevel=2)
            x = (x > 0).astype(int)
        elif len(xvalues) and xvalues.max() > 1:
            warnings.warn(f"WARNING: 'x' should be binary int, but '{xvalues.max()}' values were provided. "
                           "Binarizing to x = (x > 0).astype(int).", stacklevel=2)
            x = (x > 0).astype(int)

        # Check score cutoff "level"
        if  "level" in kwargs:
            level = kwargs["level"]
            if not isinstance(level, float):
                try:
                    level = float(level)
                except:
                    warnings.warn(f"WARNING: 'level' should be float in range 0.0 - 1.0, but '{level}' was provided. "
                                   "Setting to default value level = 0.5.", stacklevel=2)
                    level = 0.5
            if level < 0.0 or level > 1.0:
                    warnings.warn(f"WARNING: 'level' should be float in range 0.0 - 1.0, but '{level}' was provided. "
                                   "Setting to default value level = 0.5.", stacklevel=2)
                    level = 0.5
            kwargs["level"] = level

        # Check interval length "nmin"
        if  "nmin" in kwargs:
            nmin = kwargs["nmin"]
            if not isinstance(nmin, int):
                try:
                    nmin = int(nmin)
                except:
                    warnings.warn(f"WARNING: 'nmin' should be positive int, but '{nmin}' was provided. "
                                   "Setting to default value nmin = 30.", stacklevel=2)
                    nmin = 30
            if nmin < 1:
                    warnings.warn(f"WARNING: 'nmin' should be positive int, but '{nmin}' was provided. "
                                   "Setting to default value nmin = 30.", stacklevel=2)
                    nmin = 30
            kwargs["nmin"] = nmin

        # Check tolerance for active intervals "atol"
        if  "atol" in kwargs:
            atol = kwargs["atol"]
            if not isinstance(atol, int):
                try:
                    atol = int(atol)
                except:
                    warnings.warn(f"WARNING: 'atol' should be non-negative int, but '{atol}' was provided. "
                                   "Setting to default value atol = 5.", stacklevel=2)
                    atol = 5
            if atol < 0:
                    warnings.warn(f"WARNING: 'atol' should be non-negative int, but '{atol}' was provided. "
                                   "Setting to default value atol = 5.", stacklevel=2)
                    atol = 5
            kwargs["atol"] = atol

        # Check tolerance for rest intervals "rtol"
        if  "rtol" in kwargs:
            rtol = kwargs["rtol"]
            if not isinstance(rtol, int):
                try:
                    rtol = int(rtol)
                except:
                    warnings.warn(f"WARNING: 'rtol' should be non-negative int, but '{rtol}' was provided. "
                                   "Setting to default value rtol = 1.", stacklevel=2)
                    rtol = 1
            if rtol < 0:
                    warnings.warn(f"WARNING: 'rtol' should be non-negative int, but '{rtol}' was provided. "
                                   "Setting to default value rtol = 1.", stacklevel=2)
                    rtol = 1
            kwargs["rtol"] = rtol

        # Check precision flag "prec"
        if  "prec" in kwargs:
            prec = kwargs["prec"]
            if not isinstance(prec, bool):
                warnings.warn(f"WARNING: 'prec' should be boolean flag, but '{prec}' was provided. "
                               "Setting to default value prec = False.", stacklevel=2)
                prec = False
            kwargs["prec"] = prec

        # Check flag "loop"
        if  "loop" in kwargs:
            loop = kwargs["loop"]
            if not isinstance(loop, bool):
                warnings.warn(f"WARNING: 'loop' should be boolean flag, but '{loop}' was provided. "
                               "Setting to default value loop = False.", stacklevel=2)
                loop = False
            kwargs["loop"] = loop

        # Check flag to convert rest/sleep interval indices to "durations"
        if  "durations" in kwargs:
            durations = kwargs["durations"]
            if not isinstance(durations, bool):
                warnings.warn(f"WARNING: 'durations' should be boolean flag, but '{durations}' was provided. "
                               "Setting to default value durations = False.", stacklevel=2)
                durations = False
            kwargs["durations"] = durations

        return func(x, **kwargs)
    return wrapper_decorator

