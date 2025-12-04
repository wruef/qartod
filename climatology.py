#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import dask
from loguru import logger
import dask.array as da
import xarray as xr

def _auto_chunk_time(da):
    """
    Ensure time dimension is in a single chunk for resampling/groupby operations.
    Other dimensions are auto-chunked for memory efficiency.
    """
    chunk_dict = {}
    for dim, size in da.sizes.items():
        if dim == "time":
            chunk_dict[dim] = -1  # all time in one chunk for resample/groupby
        else:
            chunk_dict[dim] = "auto"
    return da.chunk(chunk_dict)


def _compute_monthly_means(da):
    """
    Fully Dask-optimized monthly mean.
    Depth dims are averaged lazily.
    """
    da = _auto_chunk_time(da)
    m = da.resample(time="1M").mean(dim="time")

    if "depth" in m.dims:
        m = m.mean(dim="depth")  # lazy depth averaging

    return m


def _compute_monthly_std(da):
    """
    Fully Dask-optimized monthly std.
    Returns a pandas Series indexed 1..12.
    """
    da = _auto_chunk_time(da)
    g = da.groupby("time.month").std(dim="time")

    if "depth" in g.dims:
        g = g.mean(dim="depth")  # lazy depth averaging

    # Ensure all months exist
    all_months = xr.DataArray(np.arange(1, 13), dims="month", name="month")
    g = g.reindex(month=all_months)

    # Rechunk month to a single chunk for interpolation
    g = g.chunk({"month": -1})

    # Interpolate missing months lazily
    g = g.interpolate_na(dim="month", method="linear", fill_value="extrapolate")

    # 90° cyclic shift interpolation
    rolled = g.roll(month=3, roll_coords=False)
    rolled = rolled.interpolate_na(dim="month", method="linear", fill_value="extrapolate")
    g = rolled.roll(month=-3, roll_coords=False)

    # Compute final 12-element series
    return g.compute().to_series()


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
        r2 = 0
    else:
        r2 = 1 - (resid / np.sum((ts_fit - ts_fit.mean())**2))

    return beta, r2, N


def _compute_monthly_fit(monthly_means):
    """
    Compute the monthly fitted climatology curve from monthly means (pandas Series).

    If harmonic regression fails or r2 < 0.15, return the raw monthly means.
    """
    ts = monthly_means.values
    beta, r2, N = _harmonic_regression(ts)

    if beta is None or r2 < 0.15:
        # fallback to raw monthly means
        return monthly_means.groupby(monthly_means.index.month).mean()

    # Construct fitted monthly climatology
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
    idx = monthly_means.index.month
    fitted_monthly = pd.Series(fitted, index=idx).groupby(idx).mean()

    # Make sure it's 1..12
    return fitted_monthly.reindex(np.arange(1, 13))

# Assume _compute_monthly_means and _compute_monthly_std are imported from previous Dask-safe version

def process_climatology(data, param, limits, site=None, node=None, sensor=None, stream=None):
    """
    Compute climatology QARTOD test for a parameter in a Dask-safe way.

    Parameters:
        data (xarray.Dataset): input dataset
        param (str): parameter to test
        limits (dict): limits dictionary
        site, node, sensor, stream: optional metadata

    Returns:
        dict: climatology test results
    """
    results = {}

    # Select the variable
    da = data[param]

    # Compute monthly mean and std lazily
    monthly_mean = _compute_monthly_means(da)
    monthly_std = _compute_monthly_std(da)

    # Compute climatology thresholds (example: mean ± N*std)
    # Adjust based on your QARTOD definition
    n_std = limits.get("std_multiplier", 3)

    # Create lazy masks
    upper_limit = monthly_mean + n_std * monthly_std
    lower_limit = monthly_mean - n_std * monthly_std

    # Align to full time series
    monthly_mean_aligned = monthly_mean.reindex(time=da.time, method="nearest")
    monthly_std_aligned = monthly_std.reindex(da.time, method="nearest")
    upper_limit_aligned = upper_limit.reindex(da.time, method="nearest")
    lower_limit_aligned = lower_limit.reindex(da.time, method="nearest")

    # Compute flags lazily
    flags = xr.where((da < lower_limit_aligned) | (da > upper_limit_aligned), 1, 0)

    # Compute final results (small arrays only)
    results["flags"] = flags.compute()  # this triggers computation
    results["monthly_mean"] = monthly_mean.compute()
    results["monthly_std"] = monthly_std

    return results
