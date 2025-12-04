#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ast import literal_eval
import numpy as np
import pandas as pd
from loguru import logger

import qartodProcessing as qp 
import grossRange as gr
import climatology as ct

def runQartod(test,data,param,limits,site,node,sensor,stream):
    """
    Runs code to compile tables for the specified QARTOD test on the provided dataset for a given parameter.

    Parameters:
        test (str): The name of the QARTOD test to compile (e.g., 'gross_range', 'climatology').
        data (xarray.Dataset): The dataset containing the data to be tested.
        param (str): The parameter to test within the dataset.
        limits (dict): Dictionary of limits or thresholds for the test.
        site (str): Site identifier.
        node (str): Node identifier.
        sensor (str): Sensor identifier.
        stream (str): Data stream identifier.

    Returns:
        qartodResults: The results of the QARTOD test, format depends on the test type.
    """

    logger.info(f"running {test} for {param}")

    qartodResults = None

    if 'gross_range' in test:
        qartodResults = gr.process_gross_range(data, [param], limits, site=site,
                                        node=node, sensor=sensor, stream=stream)

    elif 'climatology' in test:
        qartodResults = ct.process_climatology(data, param, limits, site=site,
                                                    node=node, sensor=sensor, stream=stream)
      
    else:
        logger.warning(f"Unsupported QARTOD test: {test}")
        
    return qartodResults


def main(argv=None):
    # setup input arguments
    args = qp.inputs(argv)
    refDes = args.refDes
    cut_off = args.cut_off
    decThreshold = int(args.decThreshold)
    userVars = args.userVars

    # load parameter dictionaries
    param_dict = (pd.read_csv('parameterMap.csv',converters={"variables": literal_eval,"limits": literal_eval})).set_index('dataParameter').T.to_dict()

    sites_dict = (pd.read_csv('siteParameters.csv',converters={"variables": literal_eval})).set_index('refDes').T.to_dict('series')

    qartod_tests = (pd.read_csv('qartodTests.csv', converters={"output": literal_eval,"parameters": literal_eval,"profileCalc": literal_eval})).set_index('qartodTest').T.to_dict()

    platform = sites_dict[refDes]['platformType']

    pidDict = qp.loadPID();

    # define sub-variables
    site, node, port, instrument,method,stream = sites_dict[refDes]['zarrFile'].split('-')
    sensor = port + '-' + instrument

    # load data
    data = qp.loadData(sites_dict[refDes]['zarrFile'])
    allVars = list(data.keys())

    # load annotations
    annotations = qp.loadAnnotations(refDes)
 
    if 'all' in userVars:
        dataVars=sites_dict[refDes]['variables']
    else:
        dataVars = [userVars]
    
    paramList = []
    qartodTests_dict = {}
    for qcVar in dataVars:
        qartodTests_dict[qcVar] = {}
        qcParamList = [i for i in param_dict if qcVar in param_dict[i]['variables']]
        if not qcParamList:
            logger.warning(f"Variable '{qcVar}' not found in param_dict. Skipping.")
            continue
        qcParam = qcParamList[0]
        qartodTests_dict[qcVar]['tests'] = {t for t in qartod_tests if qcParam in qartod_tests[t]['parameters']}
        qartodTests_dict[qcVar]['limits'] = param_dict[qcParam]['limits']
        for p in param_dict[qcParam]['variables']:
            paramList.append(p)
     
    if 'profiler' in platform:
        if 'int_ctd_pressure' in data:
            paramList.append('int_ctd_pressure')
        elif 'sea_water_pressure' in data:
            paramList.append('sea_water_pressure')

    dropList = [item for item in allVars if item not in paramList]
    data = data.drop_vars(dropList)
    qartodDict = {}
    
    if 'fixed' in platform:
        if ( (len(data['time']) > decThreshold) and (decThreshold > 0) ):
            data = qp.decimateData(data, decThreshold)
        for param in dataVars:
            data = qp.processData(data,param)
            data = qp.filterData(data, node, site, sensor, param, cut_off, annotations, pidDict)
            qartodDict[param] = {}
            for test in qartodTests_dict[param]['tests']:
                qartodDict[param][test] = {}
                qartodDict[param][test][platform] = runQartod(test,data,param,qartodTests_dict[param]['limits'],site,node,sensor,stream)

    elif 'profiler' in platform:
        if 'sea_water_pressure' in data:
            pressParam = 'sea_water_pressure'
        elif 'int_ctd_pressure' in data:   
            pressParam = 'int_ctd_pressure'
        else:
            logger.error('No pressure parameter found; unable to bin data!')
        if 'SF0' in node:
            #shallow_upper = np.arange(6,105,1)  
            #shallow_lower = np.arange(105,200,5)
            shallow_upper = np.arange(6,8,1)  
            shallow_lower = np.arange(10,15,5)
            binList = np.concatenate((shallow_upper,shallow_lower), axis=0).tolist()
        elif 'DP0' in node:
            maxDepth = {'DP01A': 2900, 'DP01B': 600, 'DP03A': 2600}
            binList = np.arange(200,maxDepth[node], 5).tolist()
        bins = []
        for i in range(0,len(binList)-1):
            bins.append((binList[i], binList[i+1]))

        for param in dataVars:
            qartodDict[param] = {}
            for test in qartodTests_dict[param]['tests']:
                qartodDict[param][test] = {}
                if 'integrated' in qartod_tests[test]['profileCalc']:
                    if ( (len(data['time']) > decThreshold) and (decThreshold > 0) ): 
                        data = qp.decimateData(data, decThreshold)
                    data = qp.processData(data,param)
                    data = qp.filterData(data, node, site, sensor, param, cut_off, annotations, pidDict)
                    qartodDict[param][test]['int'] = runQartod(test,data,param,qartodTests_dict[param]['limits'],site,node,sensor,stream)
                
                if 'binned' in qartod_tests[test]['profileCalc']:
                    qartodDict[param][test]['binned'] = {}
                    for pressBin in bins:
                        print('pressBin: ', pressBin)
    
                        # Lazy selection of bin; no compute() yet
                        mask = (data[pressParam] > pressBin[0]) & (data[pressParam] < pressBin[1])
                        data_bin = data.where(mask, drop=False)
                        data_bin = data_bin.dropna(dim="time", how="all")


                        # Only compute a small boolean to check if there’s data
                        if data_bin[pressParam].notnull().any().compute():
                            if (len(data['time']) > decThreshold) and (decThreshold > 0):
                                data_bin = qp.decimateData(data_bin, decThreshold)

                            data_bin = qp.processData(data_bin, param)  # ensure processData is Dask-aware
                            data_bin = qp.filterData(data_bin, node, site, sensor, param, cut_off, annotations, pidDict)

                            try:
                                qartodRow = runQartod(
                                    test, data_bin, param,
                                    qartodTests_dict[param]['limits'],
                                    site, node, sensor, stream
                                )
                                print(qartodRow)
                            except Exception as e:
                                print('failed runQartod for ', test, e)
                                qartodRow = 'unable to calculate for pressure bin'
                        else:
                            print('no data available for bin: ', pressBin)
                            qartodRow = 'unable to calculate for pressure bin'

    qartodDict[param][test]['binned'][pressBin] = qartodRow




                    
                       
    qp.exportTables(qartodDict,site,node,sensor,qartod_tests)    
    

if __name__ == '__main__':
    main()
