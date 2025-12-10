#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
import sys
import decimate
from loguru import logger
from datetime import datetime
import dateutil.parser as parser
import json
import os
import pytz
import s3fs
import xarray as xr


def identify_blocks(flags, time_step, padding=0):
    """
    Use a boolean array of quality flags to find and create blocks of data
    bound by the starting and ending dates and times of consecutive flagged
    points. Points are defined as consecutive if they occur within a certain
    time range of the preceding flagged point (default is 8 hours). This helps
    to limit cases of noisy data where the flagging is inconsistent.

    There must be a minimum time range of flagged points (defined as covering
    more than 24 hours as the default) in order to create a block. Consecutive
    blocks must be more than the minimum time window apart, or they are
    combined into a single block.

    :param flags: a boolean array of data points flagged as failing a QC
        assessment
    :param time_step: a two-value list of the minimum time range to use in
        combining flagged points into a group, and the minimum range of a
        group of flagged points to determine if a block should be created.
    :param padding: add padding (in hours) to identified blocks
    :return: List of starting and ending dates and times defining
        a block of flagged data.
    """
    # find blocks of consecutive points that span a time range greater than time_step[0]
    padding = np.timedelta64(int(padding), 'h')
    diff = 0
    flg = False
    dates = []
    start = None
    stop = None
    for i in range(flags.size):
        # index through the boolean array until we find a flagged data point
        if flags.values[i] and not flg:
            diff = 0  # reset the difference estimate
            flg = True  # set the conditional to indicate we have found a bad data point
            start = flags.time.values[i]  # set the start time for the data point

        # if we have identified a starting point and the next value is flagged, set the
        # stop time and calculate the time difference
        if flags.values[i] and flg:
            stop = flags.time.values[i]
            diff = ((stop - start) / 1e9 / 60 / 60).astype(int)  # convert from nanoseconds to hours

        # if we have identified a starting point and now find a data point that is not flagged,
        # check to see if either we are at the end of the record or the next set of data points
        # (based on a time window) are flagged. we don't want one good point resetting the block
        # if we have a cluster of good/bad points.
        if not flags.values[i] and flg:
            # check to see if we are at the end of the record
            if i == flags.size - 1:
                stop = flags.time.values[i]
                diff = ((stop - start) / 1e9 / 60 / 60).astype(int)  # convert from nanoseconds to hours
                dates.append([start, stop, diff])
                continue

            # look forward time_step[0] hours
            m = (flags.time.values > flags.time.values[i]) & (
                        flags.time.values <= flags.time.values[i] + np.timedelta64(time_step[0], 'h'))

            # if there are bad points within the time window, keep adding them
            if np.any(flags.values[m]):
                stop = flags.time.values[i]
                diff = ((stop - start) / 1e9 / 60 / 60).astype(int)  # convert from nanoseconds to hours
            else:
                # otherwise close out the block
                flg = False
                if diff > time_step[0]:
                    dates.append([start, stop, diff])

    # now check the blocks to see if we have any consecutive blocks (less than time_step[1] apart)
    blocks = []
    if dates:
        # first, did we find any blocks of data?
        start = dates[0][0]
        stop = dates[0][1]
        if len(dates) == 1:
            # if there was only one block...
            blocks.append([start - padding, stop + padding])
        else:
            # ...otherwise
            for i in range(1, len(dates)):
                diff = ((dates[i][0] - dates[i - 1][1]) / 1e9 / 60 / 60).astype(int)
                # test to see if the difference between blocks is greater than time_step[1]
                if diff > time_step[1]:
                    # create a block
                    stop = dates[i - 1][1]
                    blocks.append([start - padding, stop + padding])
                    # update the start time for the next set
                    start = dates[i][0]

                # test if we are at the end of the blocks, if so use the last point
                if i == len(dates) - 1:
                    stop = dates[i][1]
                    blocks.append([start - padding, stop + padding])

    return blocks


def create_annotations(site, node, sensor, blocks):
    """
    Use the identified blocks of data marked as "fail" to create initial HITL
    annotations for the data. Additional HITL work will be required to review
    the data and the initial annotation flags to create a final HITL set of
    annotations that can be posted to the database.

    :param site: Site designator, extracted from the first part of the
        reference designator
    :param node: Node designator, extracted from the second part of the
        reference designator
    :param sensor: Sensor designator, extracted from the third and fourth part
        of the reference designator
    :param blocks: Consecutive blocks of bad data determined via the
        identify_blocks function
    :return: Dictionary of the initial annotations for further review
    """
    # default text to use for the HITL annotation
    fail_text = ('Based on a HITL review and automated quality assessments of the data, the data highlighted '
                 'during this time period is considered inaccurate and users should avoid using the data as '
                 'part of any analysis.')

    # create the initial annotation dictionary structure
    output = {'id': [], 'subsite': [], 'node': [], 'sensor': [], 'method': [], 'stream': [], 'parameters': [],
              'beginDT': [], 'beginDate': [], 'endDT': [], 'endDate': [], 'exclusionFlag': [], 'qcFlag': [],
              'source': [], 'annotation': []}

    for block in blocks:
        start = ((block[0] - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')).astype(np.int64) * 1000
        stop = ((block[1] - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')).astype(np.int64) * 1000
        output['id'].append('')
        output['subsite'].append(site)
        output['node'].append(node)
        output['sensor'].append(sensor)
        output['method'].append(None)
        output['stream'].append(None)
        output['parameters'].append([])
        output['beginDT'].append(start)
        output['beginDate'].append(np.datetime_as_string(block[0], unit='s'))
        output['endDT'].append(stop)
        output['endDate'].append(np.datetime_as_string(block[1], unit='s'))
        output['exclusionFlag'].append('False')
        output['qcFlag'].append('fail')
        output['source'].append('replace.me@whatever.com')
        output['annotation'].append(fail_text)

    return output



def parse_qc(ds):
    """
    Extract the QC test results from the different variables in the data set,
    and create a new variable with the QC test results set to match the logic
    used in QARTOD testing. Instead of setting the results to an integer
    representation of a bitmask, use the pass = 1, not_evaluated = 2,
    suspect_or_of_high_interest = 3, fail = 4 and missing = 9 flag values from
    QARTOD.

    This code was inspired by an example notebook developed by the OOI Data
    Team for the 2018 Data Workshops. The original example, by Friedrich Knuth,
    and additional information on the original OOI QC algorithms can be found
    at:

    https://oceanobservatories.org/knowledgebase/interpreting-qc-variables-and-results/

    :param ds: dataset with *_qc_executed and *_qc_results variables
    :return ds: dataset with the *_qc_executed and *_qc_results variables
        reworked to create a new *_qc_summary variable with the results
        of the QC checks decoded into a QARTOD style flag value.
    """
    # create a list of the variables that have had QC tests applied
    variables = [x.split('_qc_results')[0] for x in ds.variables if 'qc_results' in x]

    # for each variable with qc tests applied
    for var in variables:
        # set the qc_results and qc_executed variable names and the new qc_flags variable name
        qc_result = var + '_qc_results'
        qc_executed = var + '_qc_executed'
        qc_summary = var + '_qc_summary_flag'

        # create the initial qc_flags array
        flags = np.tile(np.array([0, 0, 0, 0, 0, 0, 0, 0]), (len(ds.time), 1))
        # the list of tests run, and their bit positions are:
        #    0: dataqc_globalrangetest
        #    1: dataqc_localrangetest
        #    2: dataqc_spiketest
        #    3: dataqc_polytrendtest
        #    4: dataqc_stuckvaluetest
        #    5: dataqc_gradienttest
        #    6: undefined
        #    7: dataqc_propagateflags

        # use the qc_executed variable to determine which tests were run, and set up a bit mask to pull out the results
        executed = np.bitwise_or.reduce(ds[qc_executed].values.astype('uint8'))
        executed_bits = np.unpackbits(executed.astype('uint8'))

        # for each test executed, reset the qc_flags for pass == 1, suspect == 3, or fail == 4
        for index, value in enumerate(executed_bits[::-1]):
            if value:
                if index in [2, 3, 4, 5, 6, 7]:
                    flag = 3
                else:
                    # only mark the global range test as fail, all the other tests are problematic
                    flag = 4
                mask = 2 ** index
                m = (ds[qc_result].values.astype('uint8') & mask) > 0
                flags[m, index] = 1   # True == pass
                flags[~m, index] = flag  # False == suspect/fail

        # add the qc_flags to the dataset, rolling up the results into a single value
        ds[qc_summary] = ('time', flags.max(axis=1, initial=1).astype(np.int32))

        # set up the attributes for the new variable
        ds[qc_summary].attrs = dict({
            'long_name': '%s QC Summary Flag' % ds[var].attrs['long_name'],
            'standard_name': 'aggregate_quality_flag',
            'comment': ('Converts the QC Results values from a bitmap to a QARTOD style summary flag, where '
                        'the values are 1 == pass, 2 == not evaluated, 3 == suspect or of high interest, '
                        '4 == fail, and 9 == missing. The QC tests, as applied by OOI, only yield pass or '
                        'fail values.'),
            'flag_values': np.array([1, 2, 3, 4, 9]).astype(np.int32),
            'flag_meanings': 'pass not_evaluated suspect_or_of_high_interest fail missing'
        })

    return ds


def add_annotation_qc_flags(ds, annotations, pidDict):
    """
    Add the annotation qc flags to a dataset as a data variable. From the
    annotations, add the QARTOD flags to the dataset for each relevant data
    variable in the annotations.

    :param ds: Xarray dataset object containing the OOI data for a given
        reference designator-method-stream
    :param annotations: Pandas dataframe object which contains the annotations
        to add to the dataset

    :return ds: The input xarray dataset with the annotation qc flags added as a
        named variable to the dataset.
    """
    # First, add a local function to convert times
    def convert_time(ms):
        if ms is None:
            return None
        else:
            return datetime.utcfromtimestamp(ms/1000)

    # First, check the type of the annotations to determine if needed to put into a dataframe
    if type(annotations) is list or type(annotations) is dict:
        annotations = pd.DataFrame(annotations)

    # Convert the flags to QARTOD flags
    codes = {
        None: 0,
        'pass': 1,
        'not_evaluated': 2,
        'suspect': 3,
        'fail': 4,
        'not_operational': 0,
        'not_available': 0,
        'pending_ingest': 0
    }
    annotations['qcFlag'] = annotations['qcFlag'].map(codes).astype('category')

    # Filter only for annotations which apply to the dataset
    stream = ds.attrs["stream"]
    stream_mask = annotations["stream"].apply(lambda x: True if x == stream or x is None else False)
    annotations = annotations[stream_mask]

    # Explode the annotations so each parameter is hit for each
    # annotation
    annotations = annotations.explode(column="parameters")

    ###
    ### Lookup parameter name in parameter/pid dictionary
    ###
    stream_annos = {}
    for pid in annotations["parameters"].unique():
        if np.isnan(pid):
            param_name = "rollup"
        else:
            pid_key = 'PD' + str(pid)
            pid_info = pidDict.get(pid_key)
            param_name = pid_info['netcdf_name'] if pid_info and 'netcdf_name' in pid_info else "unknown_param"
        stream_annos.update({param_name: pid})

    # Next, get the flags associated with each parameter or all parameters
    flags_dict = {}
    for key in stream_annos.keys():
        # Get the pid and associated name
        pid_name = key
        pid = pd.to_numeric(stream_annos.get(key), errors='coerce')

        # Get the annotations associated with the pid
        if np.isnan(pid):
            pid_annos = annotations[annotations["parameters"].isna()]
        else:
            pid_annos = annotations[annotations["parameters"] == pid]

        pid_annos = pid_annos.sort_values(by="qcFlag")

        # Create an array of flags to begin setting the qc-values
        pid_flags = pd.Series(np.zeros(ds.time.values.shape), index=ds.time.values)

        # For each index, set the qcFlag for each respective time period
        for ind in pid_annos.index:
            beginDT = pid_annos["beginDT"].loc[ind]
            endDT = pid_annos["endDT"].loc[ind]
            qcFlag = pid_annos["qcFlag"].loc[ind]
            # Convert the time to actual date times
            beginDT = convert_time(beginDT)
            if endDT is None or np.isnan(endDT):
                endDT = datetime.now()
            else:
                endDT = convert_time(endDT)
            # Set the qcFlags for the given time range
            pid_flags[(pid_flags.index > beginDT) & (pid_flags.index < endDT)] = qcFlag

        # Save the results
        flags_dict.update({pid_name: pid_flags})

    # Add the flag results to the dataset for key in flags_dict
    for key in flags_dict.keys():
        # Generate a variable name
        var_name = "_".join((key.lower(), "annotations", "qc", "results"))

        # Now add to the dataset
        flags = xr.DataArray(flags_dict.get(key), dims="time")
        ds[var_name] = flags

    return ds


def decimateData(xs,decimationThreshold):
    xs = xs.where(~np.isnan(xs['time']), drop=True)
    # decimate data
    dec_data_df = decimate.downsample(xs, decimationThreshold)
    # turn dataframe into dataset
    dec_data = xr.Dataset.from_dataframe(dec_data_df, sparse=False)
    dec_data = dec_data.swap_dims({'index': 'time'})
    dec_data = dec_data.reset_coords()
    dec_data.attrs = xs.attrs

    return dec_data

def filterData(data, node, site, sensor, param, cut_off, annotations, pidDict):
    index = 1

    annotations = annotations.drop(columns=['@class'])
    annotations['beginDate'] = pd.to_datetime(annotations.beginDT, unit='ms').dt.strftime('%Y-%m-%dT%H:%M:%S')
    annotations['endDate'] = pd.to_datetime(annotations.endDT, unit='ms').dt.strftime('%Y-%m-%dT%H:%M:%S')

    # create an annotation-based quality flag
    data = add_annotation_qc_flags(data, annotations, pidDict)

    # clean-up the data, NaN-ing values that were marked as fail in the QC checks and/or identified as a block
    # of failed data, and then removing all records where the rollup annotation (every parameter fails) was
    # set to fail.
    qcVar_summary_string = param + '_qc_summary_flag'
    if qcVar_summary_string in data.variables:
        m = data[qcVar_summary_string] == 4
        data[param][m] = np.nan
    qcVar_results_string = param + '_qc_results'
    if qcVar_results_string in data.variables:
        m = data[qcVar_results_string] == 4
        data[param][m] = np.nan

    if 'rollup_annotations_qc_results' in data.variables:
        data = data.where(data.rollup_annotations_qc_results < 4)
 
    annotations_flag_string = param + '_annotations_qc_results'
    if annotations_flag_string in data.variables:
        data = data.where(data[annotations_flag_string] < 3)

    # if a cut_off date was used, limit data to all data collected up to the cut_off date.
    # otherwise, set the limit to the range of the downloaded data.
    if cut_off:
        cut = parser.parse(cut_off)
        cut = cut.astimezone(pytz.utc)
        end_date = cut.strftime('%Y-%m-%dT%H:%M:%S')
        src_date = cut.strftime('%Y-%m-%d')
    else:
        cut = parser.parse(data.time_coverage_end)
        cut = cut.astimezone(pytz.utc)
        end_date = cut.strftime('%Y-%m-%dT%H:%M:%S')
        src_date = cut.strftime('%Y-%m-%d')

    data = data.sel(time=slice('2014-01-01T00:00:00', end_date))

    return data

def get_s3_kwargs():
    aws_key = os.environ.get("AWS_KEY")
    aws_secret = os.environ.get("AWS_SECRET")

    s3_kwargs = {'key': aws_key, 'secret': aws_secret}
    return s3_kwargs


def inputs(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    # initialize argument parser
    inputParser = argparse.ArgumentParser(
        description="""Download and process instrument data to generate QARTOD lookup tables""")

    # assign input arguments.
    inputParser.add_argument("-rd", "--refDes", dest="refDes", type=str, required=True)
    inputParser.add_argument("-co", "--cut_off", dest="cut_off", type=str, required=False)
    inputParser.add_argument("-d", "--decThreshold", dest="decThreshold", type=str, required=True)
    inputParser.add_argument("-v", "--userVars", dest="userVars", type=str, required=True)

    # parse the input arguments and create a parser object
    args = inputParser.parse_args(argv)
 
    return args


def loadAnnotations(site):
    logger.info(f"loading annotations for {site}")
    anno = {}
    fs = s3fs.S3FileSystem(**get_s3_kwargs())
    INPUT_BUCKET = 'ooi-data/'
    annoFile = INPUT_BUCKET + 'annotations/' + site + '.json'
    if fs.exists(annoFile):
        anno_store = fs.open(annoFile)
        anno = json.load(anno_store)
        anno = pd.DataFrame(anno)
    else:
        print(f"error retrieving annotation history for {site}")

    return anno


def loadData(zarrDir):
    fs = s3fs.S3FileSystem(**get_s3_kwargs())
    zarr_store = fs.get_mapper('ooi-data/' + zarrDir)
    ds = xr.open_zarr(zarr_store, consolidated=True)

    return ds


def loadPID():
    pidRawFile = 'https://raw.githubusercontent.com/oceanobservatories/preload-database/refs/heads/master/csv/ParameterDefs.csv'
    pidDict = pd.read_csv(pidRawFile,usecols=['netcdf_name','id']).set_index('id').T.to_dict()

    return pidDict


def processData(data,param):
    print('in the processData loop: ', param)

    return data