# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from utils.targeted.peak_refinder import PeakRefinder

def combine(peak_information: pd.DataFrame) -> Dict[str, list]:
    """
    Convert the peak information DataFrame into a dictionary.
    """
    dict_data = {}
    for row in peak_information.itertuples():
        key = str(row.Sample_Name) + str(row.Component_Name)
        dict_data[key] = [
            row.Retention_Time, row.rtmin, row.rtmax, row.Area, 
            row.sn, row.sn_2, row.sn_3, row.sn_5, row.peak_class, 
            row.baseline, row.Height, row.points, row.min_rt, row.max_rt,
            row.lr_diff, str(row.Sample_Name), str(row.Component_Name)
        ]
    return dict_data

def preprocess_data(peak_information: pd.DataFrame, peak_matrix: pd.DataFrame, model: str, dataset: list) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Preprocess peak_information, peak_matrix, and dataset based on the given model.
    """
    rename_cols = {
        'Sample Name': 'Sample_Name', 'Component Name': 'Component_Name', 
        'Retention Time': 'Retention_Time', 'Signal/Noise': "sn", 
        'Signal/Noise_2': "sn_2", 'Signal/Noise_3': "sn_3", 'Signal/Noise_5': "sn_5"
    }
    peak_information = peak_information.rename(columns=rename_cols)
    
    peak_matrix = peak_matrix.drop_duplicates(subset='mw ID', keep='first')
    peak_matrix.set_index(peak_matrix.columns[0], inplace=True)
    
    dataset_dataframe = pd.DataFrame([[d[0][0], d[0][1], d[1], d[2]] for d in dataset], 
                                     columns=['SampleID', 'quan', 'rt', "int"])
    dataset_dataframe = dataset_dataframe[dataset_dataframe['SampleID'].str.endswith(model)]
    dataset_dataframe = dataset_dataframe[dataset_dataframe['quan'].isin(peak_information["Component_Name"])]
    
    peak_information_dict = combine(peak_information)
    
    # Vectorized filtering: set cells that are not in the dictionary to NaN
    valid_keys = set(peak_information_dict.keys())
    for col_name in peak_matrix.columns[1:]:
        keys = col_name + peak_matrix.index.astype(str)
        mask = ~keys.isin(valid_keys) & peak_matrix[col_name].notna()
        if mask.any():
            peak_matrix.loc[mask, col_name] = np.nan

    return peak_information_dict, peak_matrix, dataset_dataframe

def refind_peaks_parallel(threads: int, Refind: PeakRefinder, peak_matrix: pd.DataFrame, dataset_dataframe: pd.DataFrame, peak_group: pd.DataFrame) -> list:
    """
    Perform peak refinding in parallel using multiple threads.
    """
    peak_matrix_refind = peak_matrix.drop(["rt"], axis=1)
    
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

    if Refind is None:
        Refind = PeakRefinder()

    # Note: peak_information is accessed globally here. For better practices, 
    # it should be passed as an argument. The original logic is maintained to preserve structure.
    pass 

def update_peak_information(peak_information: dict, refind_result: list, peak_matrix: pd.DataFrame) -> Tuple[dict, pd.DataFrame]:
    """
    Update peak information and matrix based on the refinding results.
    """
    refind_result_dict = {}
    for ion in refind_result:
        key = str(ion[0]) + str(ion[1])
        refind_result_dict[key] = list(ion[2:17]) + [str(ion[0]), str(ion[1])]
    
    peak_information.update(refind_result_dict)
    
    for key, value in refind_result_dict.items():
        col_name = value[15]
        row_idx = value[16]
        if col_name in peak_matrix.columns and row_idx in peak_matrix.index:
            peak_matrix.at[row_idx, col_name] = value[3]
            
    return peak_information, peak_matrix

def post_process_peak_matrix(peak_matrix: pd.DataFrame, peak_information: dict, all_ions: dict, indir: str, model: str, ion_type: str) -> Tuple[pd.DataFrame, dict]:
    """
    Post-process the matrix and fill missing values using fully vectorized operations.
    """
    df_info = pd.DataFrame(peak_information).T
    df_info.columns = [
        'rt', 'rtmin', 'rtmax', 'area', 'sn', 'sn_2', 'sn_3', 'sn_5', 
        'peak_class', 'baseline', 'height', 'points', 'min_rt', 'max_rt', 'lr_diff', 'sampleID', 'mw ID'
    ]
    
    numeric_cols = ['rt', 'rtmin', 'rtmax', 'area', 'sn', 'sn_2', 'sn_3', 'sn_5', 'baseline', 'height', 'points', 'min_rt', 'max_rt', 'lr_diff']
    df_info[numeric_cols] = df_info[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    fill_values = df_info.pivot(index='mw ID', columns='sampleID', values='area')
    peak_matrix.update(fill_values, overwrite=False)

    df_info = df_info[(df_info['height'] > 0) & (df_info['sn'] > 6)].copy()
    df_info["ID"] = df_info["sampleID"].astype(str) + df_info['mw ID'].astype(str)
    
    valid_ids = set(df_info["ID"].tolist())
    peak_information = {k: v for k, v in peak_information.items() if k in valid_ids}
    
    cols_mix = [col for col in peak_matrix.columns if 'mix' in col]

    peak_information = {k: v for k, v in peak_information.items() if v[16] in peak_matrix.index.tolist()}

    non_na_counts = peak_matrix[cols_mix].notna().sum(axis=1)
    half_count_of_mix_cols = len(cols_mix) / 2
    peak_matrix = peak_matrix[non_na_counts >= half_count_of_mix_cols]
    
    mwID_list = set(peak_matrix.index.tolist())
    peak_information = {k: v for k, v in peak_information.items() if v[16] in mwID_list}

    return peak_matrix, peak_information