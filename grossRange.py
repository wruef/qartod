#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
import sys
import decimate
from scipy.stats import normaltest
import dask
from dask.diagnostics import ProgressBar
from loguru import logger

import dask.array as da

from ast import literal_eval
from datetime import datetime, timezone
import dateutil.parser as parser
import json
import os
import pytz
import s3fs
import xarray as xr


def format_gross_range(parameter, sensor_range, user_range, site, node, sensor, stream, notes):
    """
    Creates a dictionary object that can later be saved to a CSV formatted
    file for use in the Gross Range lookup tables.

    :param parameter: parameter name of the variable for the calculated user range
    :param sensor_range: default sensor, or fail range, usually referenced
        from the vendor documentation
    :param user_range: user range, or sensor range, calculated from the data
    :param site: Site designator, extracted from the first part of the reference
        designator
    :param node: Node designator, extracted from the second part of the reference
        designator
    :param sensor: Sensor designator, extracted from the third and fourth part of
        the reference designator
    :param stream: Stream name that contains the data of interest
    :param notes: Notes or comments about how the Gross Range values were
        obtained
    :return qc_dict: dictionary with the sensor and user gross range values
        added in the formatting expected by the QC lookup tables
    """
    # create the dictionary
    qc_dict = {
        'subsite': site,
        'node': node,
        'sensor': sensor,
        'stream': stream,
        'parameters': {
            'inp': parameter
        },
        'qcConfig': {
             'qartod': {
                 'gross_range_test': {
                     'suspect_span': user_range,
                     'fail_span': sensor_range
                 }
             }
         },
        'source': '',
        'notes': notes
    }
    logger.info(f"qc_dict: {qc_dict}")
    return qc_dict



def process_gross_range(ds, parameters, sensor_range, **kwargs):
    """
    Memory-safe gross range calculation using Dask-backed xarray.
    - Avoids SciPy (loads full array)
    - Avoids xarray.quantile (requires single chunk)
    - Uses streaming Dask operations only
    - Compatible with older Dask (uses dask.array.percentile)
    """
    import numpy as np
    import pandas as pd
    import dask.array as da_np
    from dask.array import stats as da_stats
    from dask.diagnostics import ProgressBar

    site   = kwargs.get('site')
    node   = kwargs.get('node')
    sensor = kwargs.get('sensor')
    stream = kwargs.get('stream')
    fixed_lower = kwargs.get('fixed_lower')
    fixed_upper = kwargs.get('fixed_upper')

    gross_range = []
    sensor_range = np.atleast_2d(sensor_range).tolist()

    # ----------------------------------------------------------------------
    # Ensure dataset is chunked (CRITICAL)
    # ----------------------------------------------------------------------
    if not any(getattr(v.data, "chunks", None) for v in ds.data_vars.values()):
        if "time" in ds.dims:
            ds = ds.chunk({"time": 50000})
        else:
            ds = ds.chunk({dim: 50000 for dim in ds.dims})

    # ----------------------------------------------------------------------
    # Loop through each parameter
    # ----------------------------------------------------------------------
    for idx, param in enumerate(parameters):
        if param not in ds:
            continue

        print(f"Processing {param} ...")

        da = ds[param]

        # Lazy masking: does not load data
        da = da.where(
            (da > sensor_range[idx][0]) &
            (da < sensor_range[idx][1]) &
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

            notes = "User range based on mean  3 standard deviations."

        # ------------------------------------------------------------------
        # ENFORCE VENDOR SENSOR LIMITS
        # ------------------------------------------------------------------
        if fixed_lower or lower < sensor_range[idx][0]:
            lower = sensor_range[idx][0]

        if fixed_upper or upper > sensor_range[idx][1]:
            upper = sensor_range[idx][1]

        # ------------------------------------------------------------------
        # OUTPUT ROW FORMAT
        # ------------------------------------------------------------------
        user_range = [round(float(lower), 5), round(float(upper), 5)]

        qc_dict = format_gross_range(
            param,
            sensor_range[idx],
            user_range,
            site,
            node,
            sensor,
            stream,
            notes,
        )

        df = pd.Series(qc_dict).to_frame().T
        gross_range.append(df)

    return pd.concat(gross_range, ignore_index=True, sort=False)

