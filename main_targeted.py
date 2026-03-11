# -*- coding: utf-8 -*-
"""
@author: liutong
Modified for GitHub release.
"""

import os
import re
import sys
import argparse
import subprocess
import multiprocessing
import concurrent.futures
import numpy as np
import pandas as pd

from utils.common.plot_peaks import plot_peaks_multithread

from utils.targeted.peak_refinder import PeakRefinder
from utils.targeted.inconsistency_correction import detect_anomalies_by_group
from utils.targeted.peak_matrix_utils import (
        preprocess_data, 
        update_peak_information, 
        post_process_peak_matrix
    )

# ==========================================
# Global configuration and dynamic path constants
# ==========================================
# Dynamically get the current python environment and project directory
PYTHON_EXEC = sys.executable 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UTILS_EXTRACTION = os.path.join(BASE_DIR, "utils", "targeted", "extraction_peak.py")
UTILS_CLUSTERING = os.path.join(BASE_DIR, "utils", "targeted", "clustering_algorithm.py")

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Peak mapping workflow')
    parser.add_argument('--indir', type=str, default='indir', help='Data folder.')
    parser.add_argument('--threads', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
    parser.add_argument('--ignore_group', type=str, default="false", help='Directory for saving results')
    parser.add_argument('--type', type=str, default="rp", help='type')
    return parser.parse_args()

def run_targeted_pipeline(args=None):
    if args is None:
        args = get_args()
    indir = args.indir 

    # Preprocessing: Wiff conversion
    if not os.path.exists(os.path.join(indir, "mzML")):
        print("Start converting raw ms file into mzML")
        # Ensure Docker is installed and running in the host environment
        command = f'docker run --rm -e WINEDEBUG=-all -v "{indir}:/data" chambm/pwiz-skyline-i-agree-to-the-vendor-licenses:latest wine msconvert /data/*.wiff --mzML --outdir mzML'
        subprocess.run(command, shell=True, check=True)
    else:
        print("Read files from mzML (already converted)!")

    # Standardized Subprocess: use list arguments to avoid path and space parsing errors
    if not os.path.exists(os.path.join(indir, 'temp', 'result.csv')):
        subprocess.run([PYTHON_EXEC, UTILS_EXTRACTION, "--indir", str(indir), "--threads", str(args.threads)], check=True)
    
    subprocess.run([PYTHON_EXEC, UTILS_CLUSTERING, "--indir", str(indir), "--threads", str(args.threads), "--type", str(args.type)], check=True)

    # Load the dataset
    dataset_path = os.path.join(indir, 'temp', 'dataset_all_samples.pkl')
    dataset = pd.read_pickle(dataset_path)
    
    Refind = PeakRefinder()
    peak_table_filter = pd.DataFrame()
    peak_final = pd.DataFrame()
    peak_information_final = {}
    peak_table_filter_remark = pd.DataFrame()

    for model, model_1 in zip(["pos", "neg"], ["P", "N"]):
        target_csv_path = os.path.join(args.indir, 'temp', f'peak_table_{model}_rt.csv')
        
        if os.path.exists(target_csv_path):
            peak_information = pd.read_csv(os.path.join(args.indir, 'temp', f'peak_list_{model}.txt'), sep="\t")
            peak_matrix = pd.read_csv(target_csv_path, sep=",")
            peak_group = pd.read_csv(os.path.join(args.indir, 'sample_info.csv'), sep=",")

            sample_list = peak_group["sample_name"].tolist()
            if sample_list[0].endswith(("_N", "_P", "_HN")):
                sample_name = sample_list
            else:
                sample_name = [f"{i}_{model_1}" for i in sample_list]

            all_ions_df = pd.read_excel(os.path.join(args.indir, 'ALL_ions.xlsx'))
            all_ions_df.rename(columns={'RT (min)': 'rt_theory', 'mw ID': 'mwID'}, inplace=True)
            all_ions = pd.Series(all_ions_df.rt_theory.values, index=all_ions_df.mwID).to_dict()

            peak_information, peak_matrix, dataset_dataframe = preprocess_data(peak_information, peak_matrix, model_1, dataset)
            
            # Multi-threaded parallel computation section
            if len(peak_group) >= 2:
                peak_info_df_transposed = pd.DataFrame(peak_information).T
                peak_info_df_transposed.columns = [
                    'rt', 'rtmin', 'rtmax', 'area', 'sn', 'sn_2', 'sn_3', 'sn_5', 'peak_class',
                    'baseline', 'height', 'points', 'min_rt', 'max_rt', 'lr_diff', 'sampleID', 'mw ID'
                ]
                
                grouped_df = dataset_dataframe.groupby('quan')
                sub_dfs = [group for _, group in grouped_df]
                
                manager = multiprocessing.Manager()
                refind_result_shared = manager.list()
                chunk_size = int(np.ceil(len(sub_dfs) / (args.threads * 10)))
                chunks = [sub_dfs[i:i+chunk_size] for i in range(0, len(sub_dfs), chunk_size)]
                
                refind_result = []
                with concurrent.futures.ProcessPoolExecutor(max_workers=args.threads) as executor:
                    futures = [executor.submit(Refind.refind_peak_later_period, batch, peak_matrix, peak_info_df_transposed) for batch in chunks]
                    for future in futures:
                        refind_result.extend(future.result())

                peak_information, peak_matrix = update_peak_information(peak_information, refind_result, peak_matrix)
                peak_information, peak_matrix = detect_anomalies_by_group(args.threads, peak_information, Refind, peak_matrix, dataset_dataframe, 1)
                peak_information, peak_matrix = detect_anomalies_by_group(args.threads, peak_information, Refind, peak_matrix, dataset_dataframe, 2)

            # Matrix post-processing and result archiving
            peak_matrix, peak_information = post_process_peak_matrix(peak_matrix, peak_information, all_ions, args.indir, model, args.type)

            peak_matrix.to_csv(os.path.join(args.indir, f'peak_table_{model}_filter.csv'), index=True, na_rep='NA')
            peak_matrix_save = peak_matrix.copy()
            peak_matrix_save.columns = [re.split(r'_N|_P', col)[0] for col in peak_matrix_save.columns]
            peak_table_filter = pd.concat([peak_table_filter, peak_matrix_save], axis=0)

            peak_information_final.update(peak_information)
            peak_information_dataframe = pd.DataFrame(peak_information).T
            peak_information_dataframe.columns = [
                'rt', 'rtmin', 'rtmax', 'area', 'sn', 'sn_2', 'sn_3', 'sn_5', 
                'peak_class', 'baseline', 'height', 'points', 'min_rt', 'max_rt', 'lr_diff', 'sampleID', 'mw ID'
            ]

            peak_final = pd.concat([peak_final, peak_information_dataframe], axis=0)
    
    # Summarize and write to file
    if not peak_final.empty:
        peak_final.to_csv(os.path.join(args.indir, 'temp', 'peak_final.csv'), index=True)
    
    dataset_dataframe = pd.DataFrame([[d[0][0], d[0][1], d[1], d[2]] for d in dataset], 
                                     columns=['SampleID', 'quan', 'rt', "int"])
    
    all_ions_mode = pd.read_excel(os.path.join(args.indir, 'ALL_ions.xlsx'))
    all_ions_mode.rename(columns={'Ion mode': 'ion_mode', 'mw ID': 'quan'}, inplace=True)
    replace_dict = {'Positive': 'P', 'positive': 'P', 'Negative': 'N', 'negative': 'N', 'Hilic': 'N', 'hilic': 'N'}
    all_ions_mode['ion_mode'] = all_ions_mode['ion_mode'].replace(replace_dict)
    
    plot_peaks_multithread(args.threads, peak_table_filter, dataset_dataframe, args.indir, peak_information_final, all_ions, all_ions_mode)
    
    if args.indir.split('/')[-1] not in ["HN", "PN"]:
        if "rt" in peak_table_filter.columns:
            del peak_table_filter["rt"]
            
    peak_table_filter.to_csv(os.path.join(args.indir, 'peak_table_filter.csv'), index=True, na_rep='NA')
    peak_table_filter.to_csv(os.path.join(args.indir, f"{os.path.basename(os.path.normpath(args.indir))}.txt"), sep="\t", index=True, na_rep='NA')
    
    peak_table_filter_remark['Index'] = range(1, len(peak_table_filter) + 1)
    peak_table_filter_remark["Name"] = peak_table_filter.index
    peak_table_filter_remark["Note"] = "-"
    peak_table_filter_remark.to_csv(os.path.join(args.indir, 'remark.txt'), sep="\t", index=False)

    for file_name in ['peak_list_pos.txt', 'peak_list_neg.txt']:
        temp_file = os.path.join(args.indir, 'temp', file_name)
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == '__main__':
    # Fix for Windows multiprocessing, good practice for cross-platform support
    multiprocessing.freeze_support()
    run_targeted_pipeline()