#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import dask
from loguru import logger
import dask.array as da
import xarray as xr

def _auto_chunk_time(da_in):
    """
    Ensure time dimension is in a single chunk for resampling/groupby operations.
    Other dimensions are auto-chunked for memory efficiency.
    """
    chunk_dict = {}
    for dim, size in da_in.sizes.items():
        if dim == "time":
            chunk_dict[dim] = -1  # all time in one chunk for resample/groupby
        else:
            chunk_dict[dim] = "auto"
    return da_in.chunk(chunk_dict)


def _compute_monthly_means(da):
    """
    Compute monthly means grouped by calendar month (1..12).
    Return an xarray DataArray with a 'month' dimension (1..12).
    This is kept lazy (no .compute()) so caller can decide when to evaluate.
    """
    da = _auto_chunk_time(da)

    # Group by calendar month (1..12) — keeps 'month' as a dimension
    m = da.groupby("time.month").mean(dim="time")

    if "depth" in m.dims:
        m = m.mean(dim="depth")  # lazy depth averaging

    # Ensure months 1..12 exist and are in order
    all_months = xr.DataArray(np.arange(1, 13), dims="month", name="month")
    m = m.reindex(month=all_months)  # missing months will be NaN

    return m  # xarray DataArray with 'month' dim


def _compute_monthly_std(da):
    """
    Fully Dask-optimized monthly std.
    Returns an xarray DataArray indexed by 'month' (1..12), lazy.
    """
    da = _auto_chunk_time(da)

    g = da.groupby("time.month").std(dim="time")

    if "depth" in g.dims:
        g = g.mean(dim="depth")  # lazy depth averaging

    # Ensure all months exist
    all_months = xr.DataArray(np.arange(1, 13), dims="month", name="month")
    g = g.reindex(month=all_months)

    # Rechunk month to a single chunk for interpolation (safe)
    g = g.chunk({"month": -1})

    # Interpolate missing months lazily
    g = g.interpolate_na(dim="month", method="linear", max_gap=None)

    # 90° cyclic shift interpolation to handle cyclic boundary
    rolled = g.roll(month=3, roll_coords=False)
    rolled = rolled.interpolate_na(dim="month", method="linear", max_gap=None)
    g = rolled.roll(month=-3, roll_coords=False)

    # Return xarray DataArray (still lazy)
    return g


def _harmonic_regression(ts):
    """
    Perform the 4-cycle harmonic regression used by OOI QARTOD.
    """
    f = 1 / 12
    N = len(ts)
    t = np.arange(N)

    mask = ~np.isnan(ts)
    ts_fit = ts[mask]
    t_fit = t[mask]
    n = len(ts_fit)

    if n < 4:  # not enough data
        return None, None, None

    X = np.column_stack([
        np.ones(n),
        np.sin(2*np.pi*f*t_fit), np.cos(2*np.pi*f*t_fit),
        np.sin(4*np.pi*f*t_fit), np.cos(4*np.pi*f*t_fit),
        np.sin(6*np.pi*f*t_fit), np.cos(6*np.pi*f*t_fit),
        np.sin(8*np.pi*f*t_fit), np.cos(8*np.pi*f*t_fit),
    ])

    beta, resid, rank, s = np.linalg.lstsq(X, ts_fit, rcond=None)

    if ts_fit.size == 0:
        r2 = 0.0
    else:
        # resid is an array (sum of squared residuals). ensure scalar
        rss = float(resid) if np.asarray(resid).size > 0 else 0.0
        tss = np.sum((ts_fit - ts_fit.mean())**2)
        r2 = 0.0 if tss == 0 else 1.0 - (rss / tss)

    return beta, r2, N


def _compute_monthly_fit(monthly_means):
    """
    Compute the monthly fitted climatology curve from monthly means (pandas Series or xarray DataArray).

    If harmonic regression fails or r2 < 0.15, return the raw monthly means as a pandas Series indexed 1..12.
    """
    # Accept either pandas Series or xarray DataArray
    if isinstance(monthly_means, xr.DataArray):
        # convert to pandas Series with month index 1..12
        monthly_means = pd.Series(monthly_means.values.flatten(), index=np.arange(1, 13))

    ts = monthly_means.values
    beta, r2, N = _harmonic_regression(ts)

    if beta is None or r2 is None or r2 < 0.15:
        # fallback to raw monthly means (ensure index 1..12)
        return monthly_means.groupby(monthly_means.index).mean().reindex(np.arange(1, 13)), r2

    # Construct fitted monthly climatology (use first 12 points)
    f = 1 / 12
    t = np.arange(N)

    fitted = (
        beta[0]
        + beta[1]*np.sin(2*np.pi*f*t)
        + beta[2]*np.cos(2*np.pi*f*t)
        + beta[3]*np.sin(4*np.pi*f*t)
        + beta[4]*np.cos(4*np.pi*f*t)
    )

    # Convert to monthly climatology by averaging fitted values by calendar month
    idx = monthly_means.index
    fitted_monthly = pd.Series(fitted[:len(idx)], index=idx).groupby(idx).mean()
    return fitted_monthly.reindex(np.arange(1, 13)), r2


def process_climatology(ds, param, sensor_range, **kwargs):
    
    logger.info(f"[CLIM] processing climatology for {param}")
    
    site   = kwargs.get('site')
    node   = kwargs.get('node')
    sensor = kwargs.get('sensor')
    stream = kwargs.get('stream')

    results = {}
    da = ds[param]

    # Lazy masking: does not load data
    da = da.where(
        (da > sensor_range[0]) &
        (da < sensor_range[1]) &
        (~np.isnan(da))
    )
        
    monthly_mean = _compute_monthly_means(da)
    monthly_std  = _compute_monthly_std(da)

    # compute harmonic fit + r2 ----
    fitted_monthly, r2 = _compute_monthly_fit(monthly_mean)

    # Construct note based on r2
    if r2 is None or r2 < 0.15:
        note = f"Using raw monthly means (low variance explained: r2={0.0 if r2 is None else r2:.3f})"
    else:
        note = f"Harmonic regression variance explained: r2={r2:.3f}"
    # ----------------------------------------

    # Number of std deviations 
    n_std = 3

    # Compute numeric monthly mean/std (small arrays)
    mm = monthly_mean.compute().values
    ms = monthly_std.compute().values
  
    upper = mm + n_std * ms
    lower = mm - n_std * ms

    if len(mm) != 12:
        logger.warning(f"[CLIM] monthly_mean wrong size ({len(mm)}), padding to 12")
        mm = np.resize(mm, 12)

    if len(ms) != 12:
        logger.warning(f"[CLIM] monthly_std wrong size ({len(ms)}), padding to 12")
        ms = np.resize(ms, 12)

    results = {
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "notes": note,      
    }

    return results
