# -*- coding: utf-8 -*-
"""
@author: liutong
Modified for GitHub release: Modularized Untargeted pipeline, removed hardcoded paths.
"""

import os
import re
import sys
import time
import argparse
import subprocess
import numpy as np
import pandas as pd
import multiprocessing
import concurrent.futures

from utils.common.plot_peaks import plot_peaks_multithread
from utils.untargeted.get_feature import get_features
from utils.untargeted.peak_refinder import PeakRefinder
from utils.untargeted.inconsistency_correction import detect_anomalies_by_group

PYTHON_EXEC = sys.executable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPT_EXTRACTION = os.path.join(BASE_DIR, "utils", "untargeted", "extraction_peak.py")
SCRIPT_CLUSTERING = os.path.join(BASE_DIR, "utils", "untargeted", "clustering_algorithm.py")

DIR_MZML = "mzML"
DIR_TEMP = "temp"

FILE_RESULT = os.path.join(DIR_TEMP, "result.csv")
FILE_DATASET_PKL = os.path.join(DIR_TEMP, "dataset_all_samples.pkl")
FILE_PEAK_LIST = os.path.join(DIR_TEMP, "peak_list.txt")
FILE_PEAK_TABLE_RT = os.path.join(DIR_TEMP, "peak_table_rt.csv")
FILE_PEAK_TABLE_RT_PEAK = os.path.join(DIR_TEMP, "peak_table_rt_peak.csv")
FILE_PEAK_FINAL = os.path.join(DIR_TEMP, "peak_final.csv")

FILE_SAMPLE_INFO = "sample_info.csv"
FILE_ALL_IONS = "ALL_ions.xlsx"
FILE_PEAK_TABLE_FILTER = "peak_table_filter.csv"

SPLIT_CONDITIONS_REGEX = re.compile(r'_a|_b|_c|_d|_e|_f|_g|_h|_i|_j|_k')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Untargeted Peak mapping workflow')
    parser.add_argument('--indir', type=str, default='indir', help='Data folder.')
    parser.add_argument('--threads', default=16, type=int, metavar='N', help='number of data loading workers (default: 16)')
    parser.add_argument('--ignore_group', type=str, default="false", help='Directory for saving results')
    parser.add_argument('--ppm', type=int, default=10, help='PPM tolerance for extraction peak')
    parser.add_argument('--all_ions', type=str, default="self", choices=["cal", "self"], help='Type of all ions processing')
    parser.add_argument('--polarity', default='positive', choices=["positive", "negative"], help='Ionization polarity')
    parser.add_argument('--minWidth', type=float, default=5.0, help='Minimum peak width')
    parser.add_argument('--maxWidth', type=float, default=50.0, help='Maximum peak width')
    parser.add_argument('--s2n', type=float, default=5.0, help='Signal-to-noise ratio threshold')
    parser.add_argument('--noise', type=float, default=100.0, help='Noise level threshold')
    parser.add_argument('--mzDiff', type=float, default=0.015, help='m/z difference for peak grouping')
    parser.add_argument('--prefilter', type=float, default=3.0, help='Pre-filtering intensity threshold')
    return parser.parse_args()


def combine(peak_information):
    """Dataframe to dict conversion for peak information."""
    dict_data = {}
    for row in peak_information.itertuples(index=False):
        key = str(row.Sample_Name) + str(row.Component_Name)
        value = [
            row.Retention_Time, row.rtmin, row.rtmax, row.Area, row.sn, 
            row.sn_2, row.sn_3, row.sn_5, row.peak_class, row.baseline, 
            row.Height, row.points, row.min_rt, row.max_rt, row.lr_diff, 
            str(row.Sample_Name), str(row.Component_Name)
        ]
        dict_data[key] = value
    return dict_data

def split_and_get_first_element(value):
    split_result = SPLIT_CONDITIONS_REGEX.split(value)
    return split_result[0] if split_result else value

def find_nearest_index(lst, target):
    return min(range(len(lst)), key=lambda i: abs(lst[i] - target))

def find_closest_index(lst, target):
    closest_index, _ = min(enumerate(lst), key=lambda x: (abs(x[1] - target), x[0]))
    return closest_index

def preprocess_data(peak_information, peak_matrix, dataset):
    peak_information = peak_information.rename(columns={
        'Sample Name': 'Sample_Name', 'Component Name': 'Component_Name', 
        'Retention Time': 'Retention_Time', 'Signal/Noise': "sn", 
        'Signal/Noise_2': "sn_2", 'Signal/Noise_3': "sn_3", 'Signal/Noise_5': "sn_5"
    })
    
    peak_matrix = peak_matrix.drop_duplicates(subset='mw ID', keep='first')
    peak_matrix.set_index(peak_matrix.columns[0], inplace=True)
    
    dataset_1 = [(i[0][0], i[0][1], i[1], i[2]) for i in dataset]
    dataset_dataframe = pd.DataFrame(dataset_1, columns=['SampleID', 'quan', 'rt', "int"])
    dataset_dataframe = dataset_dataframe[dataset_dataframe['quan'].isin(peak_information["Component_Name"])]
    
    peak_info_dict = combine(peak_information)
    
    cols_to_check = peak_matrix.columns[1:]
    for col_name in cols_to_check:
        for row_index in peak_matrix.index:
            key = str(col_name) + str(row_index)
            if key not in peak_info_dict and not pd.isna(peak_matrix.at[row_index, col_name]):
                peak_matrix.at[row_index, col_name] = np.nan

    return peak_info_dict, peak_matrix, dataset_dataframe

def refind_peaks_parallel(threads, Refind, peak_matrix, dataset_dataframe, peak_group, all_ions, peak_information):
    peak_matrix_refind = peak_matrix.copy()
    peak_matrix_refind.drop(["rt"], axis=1, inplace=True, errors='ignore')
    
    sample_number = peak_group.groupby(['sample_group']).size().reset_index(name='group_count')
    sample_number.columns = ['Category', 'group_count']
    mapping = dict(zip(peak_group['sample_name'], peak_group['sample_group']))
    
    df_reset = peak_matrix_refind.reset_index()
    df_stacked = df_reset.melt(id_vars=['mw ID'], value_vars=df_reset.columns, value_name='ColumnValue')
    df_stacked.columns = ['mw ID', 'sample_name', 'area']
    df_stacked['sample_name'] = df_stacked['sample_name'].str.slice(0, -2)
    df_stacked['Category'] = df_stacked['sample_name'].map(mapping)
    df_stacked["group"] = df_stacked["mw ID"] + df_stacked["Category"]
    
    grouped = df_stacked.groupby(['mw ID', "Category"])['area'].apply(lambda x: x.notnull().sum()).reset_index(name='non_na_count')
    grouped = pd.merge(grouped, sample_number, on=["Category"], how="inner")
    
    filtered_df = grouped[grouped['non_na_count'] % grouped['group_count'] != 0]
    filtered_df = filtered_df[["mw ID", "Category"]]
    filtered_df.columns = ['quan', 'Category']

    peak_information_dataframe = pd.DataFrame(peak_information).transpose()
    peak_information_dataframe.columns = [
        'rt', 'rtmin', 'rtmax', 'area', 'sn', 'sn_2', 'sn_3', 'sn_5', 'peak_class',
        'baseline', 'height', 'points', 'min_rt', 'max_rt', 'lr_diff', 'sampleID', 'mw ID'
    ]

    grouped_df = dataset_dataframe.groupby('quan')
    sub_dfs = [group for name, group in grouped_df]

    num_cores = threads
    chunk_size = int(np.ceil(len(sub_dfs) / (num_cores * 10))) if sub_dfs else 1
    chunks = [sub_dfs[i:i+chunk_size] for i in range(0, len(sub_dfs), chunk_size)]

    refind_result = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(Refind.refind_peak_later_period, batch, peak_matrix, peak_information_dataframe, all_ions) for batch in chunks]
        for future in futures:
            refind_result.extend(future.result())

    return refind_result

def update_peak_information(peak_information, refind_result, peak_matrix):
    refind_result_dict = {}
    for peak_ion in refind_result:
        key = str(peak_ion[0]) + str(peak_ion[1])
        value = [
            peak_ion[2], peak_ion[3], peak_ion[4], peak_ion[5], peak_ion[6], peak_ion[7], 
            peak_ion[8], peak_ion[9], peak_ion[10], peak_ion[11], peak_ion[12], peak_ion[13], 
            peak_ion[14], peak_ion[15], peak_ion[16], str(peak_ion[0]), str(peak_ion[1])
        ]
        refind_result_dict[key] = value
    
    peak_information.update(refind_result_dict)
    
    for column_name in peak_matrix.columns:
        for index in peak_matrix.index:
            key = str(column_name) + str(index)
            if key in refind_result_dict:
                peak_matrix.at[index, column_name] = refind_result_dict[key][3]
    
    return peak_information, peak_matrix

def post_process_peak_matrix(peak_matrix, peak_information, all_ions, indir):
    for col_name in peak_matrix.columns[1:]:
        for row_index in peak_matrix.index:
            peak_name = str(col_name) + str(row_index)
            if peak_name in peak_information and pd.isna(peak_matrix.at[row_index, col_name]):
                peak_matrix.at[row_index, col_name] = peak_information[peak_name][3]

    peak_information_dataframe = pd.DataFrame(peak_information).transpose()
    peak_information_dataframe.columns = [
        'rt', 'rtmin', 'rtmax', 'area', 'sn', 'sn_2', 'sn_3', 'sn_5', 
        'peak_class', 'baseline', 'height', 'points', 'min_rt', 'max_rt', 'lr_diff', 'sampleID', 'mw ID'
    ]
    peak_information_dataframe = peak_information_dataframe[peak_information_dataframe['height'] > 0]

    peak_information = {k: v for k, v in peak_information.items() if v[16] in peak_matrix.index.tolist()}
    
    output_rt_peak_csv = os.path.join(indir, FILE_PEAK_TABLE_RT_PEAK)
    peak_matrix.to_csv(output_rt_peak_csv, index=True, na_rep='NA')
    
    mwID_list = peak_matrix.index.tolist()
    peak_information = {k: v for k, v in peak_information.items() if v[16] in mwID_list}

    return peak_matrix, peak_information


def run_untargeted_pipeline(args=None):
    if args is None:
        args = parse_arguments()
        
    indir = args.indir 
    start = time.time()

    # wiff to mzML
    mzml_dir = os.path.join(indir, DIR_MZML)
    if not os.path.exists(mzml_dir):
        print("[*] Start converting raw ms file into mzML")
        command = f'docker run --rm -e WINEDEBUG=-all -v "{indir}:/data" chambm/pwiz-skyline-i-agree-to-the-vendor-licenses:latest wine msconvert /data/*.wiff --mzML --outdir mzML'
        subprocess.run(command, shell=True, check=True)
    else:
        print("[*] Read files from mzML (already converted)!")

    # Execute additional script if all_ions is "cal"
    all_ions_path = os.path.join(args.indir, FILE_ALL_IONS)
    if not os.path.exists(all_ions_path) or args.all_ions == "cal":
        print(f"[*] Triggering feature extraction (XCMS) for: {args.indir}")
        get_features(
            datadir=os.path.join(args.indir, "mzML"),
            polarity=args.polarity,
            ms1ppm=args.ppm,       
            minWidth=args.minWidth,
            maxWidth=args.maxWidth,
            s2n=args.s2n,
            noise=args.noise,
            mzDiff=args.mzDiff,
            prefilter=args.prefilter
        )
        print(f"[*] Feature extraction completed. {FILE_ALL_IONS} generated.")
    else:
        print(f"[*] Found existing {FILE_ALL_IONS}. Skipping feature extraction.")

    result_csv_path = os.path.join(indir, FILE_RESULT)
    if not os.path.exists(result_csv_path):
        subprocess.run([
            PYTHON_EXEC, SCRIPT_EXTRACTION, 
            "--indir", str(indir), 
            "--threads", str(args.threads), 
            "--ppm", str(args.ppm)
        ], check=True)
        
    subprocess.run([
        PYTHON_EXEC, SCRIPT_CLUSTERING, 
        "--indir", str(args.indir), 
        "--threads", str(args.threads)
    ], check=True)

    # Load datasets
    dataset = pd.read_pickle(os.path.join(indir, FILE_DATASET_PKL))
    peak_result = pd.read_csv(result_csv_path)
    
    Refind = PeakRefinder()
    peak_table_filter = pd.DataFrame()
    peak_final = pd.DataFrame()
    peak_information_final = {}
    peak_table_filter_remark = pd.DataFrame()

    peak_table_rt_path = os.path.join(indir, FILE_PEAK_TABLE_RT)
    if os.path.exists(peak_table_rt_path):
        peak_information = pd.read_csv(os.path.join(indir, FILE_PEAK_LIST), sep="\t")
        peak_matrix = pd.read_csv(peak_table_rt_path, sep=",")
        peak_group = pd.read_csv(os.path.join(indir, FILE_SAMPLE_INFO), sep=",")

        all_ions = pd.read_excel(os.path.join(indir, FILE_ALL_IONS))
        all_ions.rename(columns={'RT (min)': 'rt_theory', 'mw ID': 'mwID'}, inplace=True)
        all_ions = pd.Series(all_ions.rt_theory.values, index=all_ions.mwID).to_dict()
        result_information = pd.read_csv(result_csv_path, sep=",")

        peak_information, peak_matrix, dataset_dataframe = preprocess_data(peak_information, peak_matrix, dataset)
        
        refind_result = refind_peaks_parallel(args.threads, Refind, peak_matrix, dataset_dataframe, peak_group, all_ions, peak_information)
        peak_information, peak_matrix = update_peak_information(peak_information, refind_result, peak_matrix)
        
        peak_information, peak_matrix = detect_anomalies_by_group(args.threads, peak_information, Refind, peak_matrix, dataset_dataframe)
        peak_information, peak_matrix = detect_anomalies_by_group(args.threads, peak_information, Refind, peak_matrix, dataset_dataframe)

        peak_matrix, peak_information = post_process_peak_matrix(peak_matrix, peak_information, all_ions, args.indir)
        
        peak_table_filter_path = os.path.join(indir, FILE_PEAK_TABLE_FILTER)
        peak_matrix.to_csv(peak_table_filter_path, index=True, na_rep='NA')

        peak_information_final.update(peak_information)
        peak_information_dataframe = pd.DataFrame(peak_information).transpose()
        peak_information_dataframe.columns = [
            'rt', 'rtmin', 'rtmax', 'area', 'sn', 'sn_2', 'sn_3', 'sn_5', 
            'peak_class', 'baseline', 'height', 'points', 'min_rt', 'max_rt', 'lr_diff', 'sampleID', 'mw ID'
        ]
    
    # Save final peak information
    if not peak_information_dataframe.empty:
        peak_final_path = os.path.join(indir, FILE_PEAK_FINAL)
        peak_information_dataframe.to_csv(peak_final_path, index=True)
    
    # Remap dataset format
    dataset_1 = [(i[0][0], i[0][1], i[1], i[2]) for i in list(dataset)]
    dataset_dataframe = pd.DataFrame(dataset_1, columns=['SampleID', 'quan', 'rt', "int"])
    
    end = time.time()
    print(f"[*] Untargeted Pipeline Finished! Runtime: {end - start:.3f} seconds")
    
    # Executing multi-thread plotting
    plot_peaks_multithread(args.threads, peak_matrix, dataset_dataframe, args.indir, peak_information_final, all_ions)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    run_untargeted_pipeline()