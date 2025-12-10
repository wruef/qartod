#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import dask.array as da_np
from dask.array import stats as da_stats
from dask.diagnostics import ProgressBar

def process_gross_range(ds, param, sensor_range, **kwargs):
    """
    Memory-safe gross range calculation using Dask-backed xarray.
    - Avoids SciPy (loads full array)
    - Avoids xarray.quantile (requires single chunk)
    - Uses streaming Dask operations only
    - Compatible with older Dask (uses dask.array.percentile)
    """

    site   = kwargs.get('site')
    node   = kwargs.get('node')
    sensor = kwargs.get('sensor')
    stream = kwargs.get('stream')

    results = {}

    # ----------------------------------------------------------------------
    # Ensure dataset is chunked (CRITICAL)
    # ----------------------------------------------------------------------
    if not any(getattr(v.data, "chunks", None) for v in ds.data_vars.values()):
        if "time" in ds.dims:
            ds = ds.chunk({"time": 50000})
        else:
            ds = ds.chunk({dim: 50000 for dim in ds.dims})

    da = ds[param]

    # Lazy masking: does not load data
    da = da.where(
        (da > sensor_range[0]) &
        (da < sensor_range[1]) &
        (~np.isnan(da))
    )
    
    # Underlying dask array
    darr = da.data 
    
    # ------------------------------------------------------------------
    # NORMALITY CHECK (streaming)
    # ------------------------------------------------------------------
    with ProgressBar():
        skew = da_stats.skew(darr, axis=0).compute()
        kurt = da_stats.kurtosis(darr, axis=0).compute()

    is_normal = (abs(skew) < 1.0) and (2 < kurt < 5)

    # ------------------------------------------------------------------
    # USER RANGE COMPUTATION
    # ------------------------------------------------------------------
    if not is_normal:
        print("   Non-normal distribution: using percentile-based limits")

        # NOTE: Older Dask uses percentile, NOT quantile
        # 0.0015 quantile == 0.15 percentile
        # 0.9985 quantile == 99.85 percentile
        with ProgressBar():
            lower = da_np.percentile(darr, 0.15).compute()
            upper = da_np.percentile(darr, 99.85).compute()

        notes = (
            "Distribution is non-normal; user range based on 0.15 and "
            "99.85 percentiles (99.7% coverage)."
        )

    else:
        print("   Normal-ish distribution: using mean  3")

        with ProgressBar():
            mu = da.mean().compute()
            sd = da.std().compute()

        lower = mu - 3 * sd
        upper = mu + 3 * sd

        notes = "User range based on mean 3 standard deviations."

    # ------------------------------------------------------------------
    # ENFORCE VENDOR SENSOR LIMITS
    # ------------------------------------------------------------------
    if lower < sensor_range[0]:
        lower = sensor_range[0]

    if upper > sensor_range[1]:
        upper = sensor_range[1]

    results = {
        "lower": lower,
        "upper": upper,
        "notes": notes
    }

    return results


