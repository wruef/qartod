#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import pandas as pd
import os

def build_climatology_object(clim,testKey):
    if 'binned' in testKey:
        rows = []
        for depth_bin, values in clim['binned'].items():
            monthRanges = [f"[{values['lower'][i]:.2f}, {values['upper'][i]:.2f}]" for i in range(12)]
            # Prepend the depth bin
            first = f"[{depth_bin[0]}, {depth_bin[1]}]"
            row = [first] + monthRanges

            rows.append(row)
    else:
        first = "[0, 0]"
        monthRanges = [f"[{clim[testKey]['lower'][i]:.2f}, {clim[testKey]['upper'][i]:.2f}]" for i in range(12)]
        rows = [[first] + monthRanges]
        
    df_clim = pd.DataFrame(rows)
    header = [""] + [f"[{i}, {i}]" for i in range(1, 13)]
    df_out = pd.concat([pd.DataFrame([header]), df_clim], ignore_index=True)
         
    return df_out


def build_gross_range_object(gr,testKey):  
    if 'binned' in testKey:
        rows = []
        for depth_bin, values in gr['binned'].items():
            gr_range = [f"[{values['lower'][0]:.2f}, {values['upper'][0]:.2f}]"]
             # Prepend the depth bin
            first = f"[{depth_bin[0]}, {depth_bin[1]}]"
            row = [first] + gr_range
            rows.append(row)
    else:
        first = "[0, 0]"
        gr_range = [f"[{gr[testKey]['lower'][0]:.2f}, {gr[testKey]['upper'][0]:.2f}]"]
        rows = [[first] + gr_range]

    df_gr = pd.DataFrame(rows)
    df_out = pd.concat([pd.DataFrame([["", "gross_range"]]), df_gr], ignore_index=True)
    
    return df_out



def exportTables(qartodDict, qartodTests_dict, site, node, sensor, stream, platform, pressParam):
    folderPath = os.path.join(os.path.expanduser('~'), 'qartod_staging')
    os.makedirs(folderPath, exist_ok=True)
    for param in qartodDict:
        param_dict = qartodDict[param]
        limits = qartodTests_dict[param]['limits']
        if 'climatology' in param_dict:
            # export tables for integrated and binned
            for testKey in param_dict['climatology'].keys():   
                if testKey not in ['int', 'binned', 'fixed']:
                    raise ValueError(f"Unexpected key in climatology dict: {testKey}")
               
                df_clim = build_climatology_object(param_dict['climatology'],testKey)
                outfile = os.path.join(folderPath, f"{site}-{node}-{sensor}-{param}.climatology_table.csv.{testKey}")
                df_clim.to_csv(outfile, index=False,header=False)
                print(f"Exported: {outfile}")
            # export lookup dictionary
            if 'profiler' in platform:
                zinp = pressParam
                notes = "Variance not reported for binned profiler climatology"
            else:
                zinp = None
                notes = param_dict['climatology'][testKey]['notes']
                
            qc_dict = {
                'subsite': site,
                'node': node,
                'sensor': sensor,
                'stream': stream,
                'parameters': {'inp': param, 'tinp': 'time', 'zinp': zinp},
                'notes': notes
            } 
            climatologyTable = f"climatology_tables/{site}-{node}-{sensor}-{param}.climatology_table.csv"
            
            dictRow = [
                qc_dict['subsite'],
                qc_dict['node'],
                qc_dict['sensor'],
                qc_dict['stream'],
                qc_dict['parameters'],
                f"{climatologyTable}",
                "",
                qc_dict['notes'],
                ]

            # --- Write CSV file ---
            outfile = os.path.join(folderPath, f"{site}-{node}-{sensor}-{param}-climatology_test_values.csv")
            with open(outfile, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                "subsite","node","sensor","stream","parameters","climatologyTable","source","notes"
                ])
                writer.writerow(dictRow)
            print(f"Exported: {outfile}")
                
                
        if 'gross_range' in param_dict:
            # export tables for integrated and binned
            for testKey in param_dict['gross_range'].keys():   
                if testKey not in ['int', 'binned', 'fixed']:
                    raise ValueError(f"Unexpected key in gross_range dict: {testKey}")
                df_gr = build_gross_range_object(param_dict['gross_range'],testKey)
                
                outfile = os.path.join(folderPath, f"{site}-{node}-{sensor}-{param}.gross_range_table.csv.{testKey}")
                df_gr.to_csv(outfile, index=False, header=False)
                print(f"Exported: {outfile}")
            # export lookup dictionary
            qc_dict = {
                'subsite': site,
                'node': node,
                'sensor': sensor,
                'stream': stream,
                'parameters': {'inp': param},
                'gross_range_suspect': [
                    float(f"{param_dict['gross_range'][testKey]['lower'][0]:.2f}"),
                    float(f"{param_dict['gross_range'][testKey]['upper'][0]:.2f}")
                ],
                'gross_range_fail': limits,
                'notes': param_dict['gross_range'][testKey]['notes']
            } 
            qcConfig = {
                "qartod": {
                    "gross_range_test": {
                    "suspect_span": qc_dict['gross_range_suspect'],
                    "fail_span": qc_dict['gross_range_fail']
                    }
                }   
            }   
            dictRow = [
                qc_dict['subsite'],
                qc_dict['node'],
                qc_dict['sensor'],
                qc_dict['stream'],
                qc_dict['parameters'],
                f"{qcConfig}",
                "",
                qc_dict['notes'],
                ]

            # --- Write CSV file ---
            outfile = os.path.join(folderPath, f"{site}-{node}-{sensor}-{param}-gross_range_test_values.csv")
            with open(outfile, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                "subsite","node","sensor","stream","parameters","qcConfig","source","notes"
                ])
                writer.writerow(dictRow)
            print(f"Exported: {outfile}")
    return
