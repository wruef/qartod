#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ast import literal_eval
import numpy as np
import pandas as pd
from loguru import logger
import traceback

import qartodProcessing as qp
import grossRange as gr
import climatology as ct
import export as ex


def runQartod(test, data, param, limits, site, node, sensor, stream):
    """
    Runs code to compile tables for the specified QARTOD test on the provided dataset for a given parameter.
    """
    logger.debug(f"running qartod for {test}, {param}")

    qartodResults = None

    try:
        if 'gross_range' in test:
            qartodResults = gr.process_gross_range(
                data, param, limits, site=site, node=node, sensor=sensor, stream=stream
            )

        elif 'climatology' in test:
            qartodResults = ct.process_climatology(
                data, param, limits, site=site, node=node, sensor=sensor, stream=stream
            )

        else:
            logger.warning(f"unsupported_qartod_test {test}")
            return None

    except Exception as e:
        logger.error(f"exception_running_test {test}, {param}, {e}")
        traceback.print_exc()

    return qartodResults


def run_binned_processing_for_param(data, param, pressParam, bins, qartodTests_dict, qartod_tests,
                                    site, node, sensor, stream, cut_off, annotations, pidDict, decThreshold):
    """
    Helper: process a parameter's profile tests for binned and integrated options.
    Returns a dict indexed by test name -> { 'int':..., 'binned': {pressBin: result, ...} }
    """
    results_for_param = {}

    for test in qartodTests_dict[param]['tests']:
        results_for_param[test] = {}
        # integrated processing
        if 'integrated' in qartod_tests[test]['profileCalc']:
            # Evaluate using a copy/process local to this param (do not mutate 'data')
            data_param = data
            # decimate if necessary (use the overall data length; safe heuristic)
            if (len(data_param['time']) > decThreshold) and (decThreshold > 0):
                data_param = qp.decimateData(data_param, decThreshold)
            data_param = qp.processData(data_param, param)
            data_param = qp.filterData(data_param, node, site, sensor, param, cut_off, annotations, pidDict)
            try:
                results_for_param[test]['int'] = runQartod(
                    test, data_param, param, qartodTests_dict[param]['limits'],
                    site, node, sensor, stream
                )
            except Exception as e:
                logger.error(f"integrated_runQartod_failed: {test}, {param}, {e}")
                traceback.print_exc()
                #####results_for_param[test]['int'] = {"climatology_table": {"month": [], "mean": [], "std": []}, "flags": []}

        # binned processing
        if 'binned' in qartod_tests[test]['profileCalc']:
            results_for_param[test]['binned'] = {}
            for pressBin in bins:
                logger.info(f"processing_press_bin {pressBin}")
                # Build mask using the full-pressure array but select in a non-mutating way
                mask = (data[pressParam] > pressBin[0]) & (data[pressParam] < pressBin[1])
                data_bin = data.where(mask, drop=False)
                data_bin = data_bin.dropna(dim="time", how="all")

                # default empty placeholder
                qartodRow = None

                try:
                    # Check whether there is any non-null pressure in the bin
                    has_data = False
                    try:
                        has_data = data_bin[pressParam].notnull().any().compute()
                    except Exception:
                        # If compute fails, fall back to checking without compute (best-effort)
                        has_data = bool(data_bin[pressParam].notnull().any())

                    if has_data:
                        # decimate based on data_bin length (not whole dataset)
                        if (len(data_bin['time']) > decThreshold) and (decThreshold > 0):
                            data_bin = qp.decimateData(data_bin, decThreshold)

                        # process and filter the bin-specific dataset
                        data_bin = qp.processData(data_bin, param)
                        data_bin = qp.filterData(data_bin, node, site, sensor, param, cut_off, annotations, pidDict)

                        # run QARTOD on the bin
                        qartodRow = runQartod(
                            test, data_bin, param,
                            qartodTests_dict[param]['limits'],
                            site, node, sensor, stream
                        )
                    else:
                        logger.info(f"no_data_for_press_bin: {pressBin}")
                        #####qartodRow = {"climatology_table": {"month": [], "mean": [], "std": []}, "flags": []}
                except Exception as e:
                    logger.error(f"failed_runQartod_for_bin {test}, {pressBin}, {e}")
                    traceback.print_exc()
                    #####qartodRow = {"climatology_table": {"month": [], "mean": [], "std": []}, "flags": []}

                # assign per-bin result inside the loop (was previously outside)
                results_for_param[test]['binned'][pressBin] = qartodRow

    return results_for_param


def runQartod_driver_main():
    """Main driver function (refactored main body)."""
    # parse inputs
    args = qp.inputs()
    refDes = args.refDes
    cut_off = args.cut_off
    decThreshold = int(args.decThreshold)
    userVars = args.userVars

    # load parameter dictionaries
    param_dict = (pd.read_csv('parameterMap.csv', converters={"variables": literal_eval, "limits": literal_eval})
                  ).set_index('dataParameter').T.to_dict()

    sites_dict = (pd.read_csv('siteParameters.csv', converters={"variables": literal_eval})
                  ).set_index('refDes').T.to_dict('series')

    qartod_tests = (pd.read_csv('qartodTests.csv', converters={"output": literal_eval, "parameters": literal_eval,
                                                              "profileCalc": literal_eval})
                    ).set_index('qartodTest').T.to_dict()

    platform = sites_dict[refDes]['platformType']
    pidDict = qp.loadPID()

    # define sub-variables
    site, node, port, instrument, method, stream = sites_dict[refDes]['zarrFile'].split('-')
    sensor = port + '-' + instrument

    # load data
    data = qp.loadData(sites_dict[refDes]['zarrFile'])
    allVars = list(data.keys())

    # load annotations
    annotations = qp.loadAnnotations(refDes)

    if 'all' in userVars:
        dataVars = sites_dict[refDes]['variables']
    else:
        dataVars = [userVars]

    paramList = []
    qartodTests_dict = {}
    for qcVar in dataVars:
        qartodTests_dict[qcVar] = {}
        qcParamList = [i for i in param_dict if qcVar in param_dict[i]['variables']]
        if not qcParamList:
            logger.warning(f"variable_not_found_in_param_dict: {qcVar}")
            continue
        qcParam = qcParamList[0]
        qartodTests_dict[qcVar]['tests'] = {t for t in qartod_tests if qcParam in qartod_tests[t]['parameters']}
        qartodTests_dict[qcVar]['limits'] = param_dict[qcParam]['limits']
        for p in param_dict[qcParam]['variables']:
            paramList.append(p)

    # profiler pressure param detection
    if 'profiler' in platform:
        if 'int_ctd_pressure' in data:
            paramList.append('int_ctd_pressure')
        elif 'sea_water_pressure' in data:
            paramList.append('sea_water_pressure')

    dropList = [item for item in allVars if item not in paramList]
    data = data.drop_vars(dropList)
    qartodDict = {}

    if 'fixed' in platform:
        pressParam = None
        # copy of data used per-param to avoid in-place mutation across params
        if ((len(data['time']) > decThreshold) and (decThreshold > 0)):
            data = qp.decimateData(data, decThreshold)
        for param in dataVars:
            qartodDict[param] = {}
            # operate on a local copy for each parameter
            data_param = qp.processData(data, param)
            data_param = qp.filterData(data_param, node, site, sensor, param, cut_off, annotations, pidDict)
            for test in qartodTests_dict[param]['tests']:
                qartodDict[param][test] = {}
                try:
                    qartodDict[param][test][platform] = runQartod(
                        test, data_param, param, qartodTests_dict[param]['limits'], site, node, sensor, stream
                    )
                except Exception as e:
                    logger.error(f"runQartod_failed_fixed_platform {test}, {param}, {e}")
                    traceback.print_exc()
                    qartodDict[param][test][platform] = {"climatology_table": {"month": [], "mean": [], "std": []}, "flags": []}

    elif 'profiler' in platform:
        # determine pressure parameter and bins
        if 'sea_water_pressure' in data:
            pressParam = 'sea_water_pressure'
        elif 'int_ctd_pressure' in data:
            pressParam = 'int_ctd_pressure'
        else:
            logger.error(f"no_pressure_parameter_found_unable_to_bin {refDes}")
            raise RuntimeError('No pressure parameter found; unable to bin data!')

        if 'SF0' in node:
            shallow_upper = np.arange(6, 8, 1)
            shallow_lower = np.arange(10, 15, 5)
            binList = np.concatenate((shallow_upper, shallow_lower), axis=0).tolist()
        elif 'DP0' in node:
            maxDepth = {'DP01A': 2900, 'DP01B': 600, 'DP03A': 2600}
            binList = np.arange(200, maxDepth[node], 5).tolist()
        else:
            # default binning fallback
            binList = np.arange(0, 200, 10).tolist()

        bins = []
        for i in range(0, len(binList) - 1):
            bins.append((binList[i], binList[i + 1]))

        # iterate parameters and produce binned/integrated results
        for param in dataVars:
            qartodDict[param] = {}
            results_for_param = run_binned_processing_for_param(
                data, param, pressParam, bins, qartodTests_dict, qartod_tests,
                site, node, sensor, stream, cut_off, annotations, pidDict, decThreshold
            )
            qartodDict[param] = results_for_param

    # Export tables once fully built
    
    print(qartodDict)
    ex.exportTables(qartodDict, qartodTests_dict, site, node, sensor, stream, platform, pressParam)


if __name__ == '__main__':
    runQartod_driver_main()

