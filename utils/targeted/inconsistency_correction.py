# -*- coding: utf-8 -*-
"""
Created on Mon Dec  16 09:38:56 2023

@author: liutong
"""

import numpy as np
import pandas as pd
import multiprocessing
import concurrent.futures

from typing import Dict, List, Tuple
from utils.common.data_processing import * 
from utils.targeted.peak_refinder import PeakRefinder


def update_peak_information(peak_information: dict, refind_result: list, peak_matrix: pd.DataFrame) -> Tuple[dict, pd.DataFrame]:
    """
    Update peak information based on the refinding results and peak matrix data.

    Args:
        peak_information (dict): Dictionary containing reference peak information.
        refind_result (list): List of refinding results for target peaks.
        peak_matrix (pd.DataFrame): DataFrame containing peak matrix data to be updated.

    Returns:
        Tuple[dict, pd.DataFrame]: Updated peak information dictionary and peak matrix DataFrame.
    """
    # Create a dictionary to store refind results using string concatenated keys
    refind_result_dict = {}
    for peak_ion in refind_result:
        key = str(peak_ion[0]) + str(peak_ion[1])
        value = [
            peak_ion[2], peak_ion[3], peak_ion[4], peak_ion[5], peak_ion[6], peak_ion[7], 
            peak_ion[8], peak_ion[9], peak_ion[10], peak_ion[11], peak_ion[12], peak_ion[13], 
            peak_ion[14], peak_ion[15], peak_ion[16], str(peak_ion[0]), str(peak_ion[1])
        ]
        refind_result_dict[key] = value
    
    # Update peak information with refind results
    peak_information.update(refind_result_dict)
    
    # Pre-map string representations of columns and indices to ensure exact matching 
    # without running an O(N*M) nested loop over the entire matrix.
    str_cols = {str(c): c for c in peak_matrix.columns}
    str_idxs = {str(i): i for i in peak_matrix.index}
    
    # Directly update specific cells in peak_matrix (O(K) complexity instead of O(N*M))
    for key, value in refind_result_dict.items():
        col_str = value[15]
        idx_str = value[16]
        if col_str in str_cols and idx_str in str_idxs:
            actual_col = str_cols[col_str]
            actual_idx = str_idxs[idx_str]
            # value[3] corresponds to peak_ion[5]
            peak_matrix.at[actual_idx, actual_col] = value[3]
            
    return peak_information, peak_matrix


def refind_peaks_parallel_single(threads: int, result_df: pd.DataFrame, Refind: PeakRefinder, 
                                 peak_matrix: pd.DataFrame, dataset_dataframe: pd.DataFrame, 
                                 peak_information: dict, num_epoch: int) -> list:
    """
    Perform parallel peak refinding using multiple threads and batch processing.

    Args:
        threads (int): Number of threads/workers to use for parallel processing.
        result_df (pd.DataFrame): DataFrame containing detected anomalies.
        Refind (PeakRefinder): PeakRefinder instance. If None, a new one is instantiated.
        peak_matrix (pd.DataFrame): DataFrame containing peak matrix data.
        dataset_dataframe (pd.DataFrame): DataFrame containing the raw dataset.
        peak_information (dict): Dictionary containing current peak information.
        num_epoch (int): Parameter specifying the number of epochs for refinding.

    Returns:
        list: A list of newly refound peaks.
    """
    if Refind is None:
        Refind = PeakRefinder()
        
    # Merge dataset with the anomalies result
    dataset_dataframe = pd.merge(dataset_dataframe, result_df, on=["quan", "SampleID"], how="inner")
    sub_dfs = dataset_dataframe.values.tolist()

    if not sub_dfs:
        return []

    # Calculate optimal chunk size for the executor
    chunk_size = int(np.ceil(len(sub_dfs) / (threads * 10)))
    chunks = [sub_dfs[i:i + chunk_size] for i in range(0, len(sub_dfs), chunk_size)]

    refind_result = []
    # Execute the refinding process concurrently
    with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(Refind.refind_peak_later_period_signal, batch, peak_matrix, peak_information, num_epoch) for batch in chunks]
        for future in futures:
            refind_result.extend(future.result())

    return refind_result


def detect_anomalies_by_group(threads: int, peak_information: dict, Refind: PeakRefinder, 
                              peak_matrix: pd.DataFrame, dataset_dataframe: pd.DataFrame, num_epoch: int) -> Tuple[dict, pd.DataFrame]:
    """
    Detect statistical anomalies in peak retention times across groups and execute parallel refinding.
    Utilizes Pandas vectorized operations for significant performance improvement.

    Args:
        threads (int): Number of concurrent workers.
        peak_information (dict): Dictionary of peak information.
        Refind (PeakRefinder): Instance for refinding peaks.
        peak_matrix (pd.DataFrame): Existing peak matrix.
        dataset_dataframe (pd.DataFrame): Dataset configuration.
        num_epoch (int): Epoch parameters for the refinding function.

    Returns:
        Tuple[dict, pd.DataFrame]: The updated peak_information and peak_matrix.
    """
    # Convert peak_information dictionary to DataFrame
    df_info = pd.DataFrame(peak_information).T
    df_info.columns = [
        'rt', 'rtmin', 'rtmax', 'area', 'sn', 'sn_2', 'sn_3', 'sn_5', 'peak_class',
        'baseline', 'height', 'points', 'min_rt', 'max_rt', 'lr_diff', 'sampleID', 'mw ID'
    ]
    
    # Ensure numeric types for calculation
    for col in ['rtmin', 'rtmax', 'rt']:
        df_info[col] = pd.to_numeric(df_info[col], errors='coerce')
        
    df_info["weight"] = df_info["rtmax"] - df_info["rtmin"]
    
    # Vectorized computation of medians (0.5 quantile) grouped by "mw ID"
    medians = df_info.groupby("mw ID")[["rtmin", "rtmax", "weight"]].transform('median')
    
    # Calculate anomaly boundaries based on medians
    upper_bound_rtmin = medians["rtmin"] + 0.05
    lower_bound_rtmin = medians["rtmin"] - 0.05
    upper_bound_rtmax = medians["rtmax"] + 0.05
    lower_bound_rtmax = medians["rtmax"] - 0.05
    
    # Build Boolean mask for anomaly detection
    mask = (
        (df_info["rtmin"] > upper_bound_rtmin) | 
        (df_info["rtmin"] < lower_bound_rtmin) | 
        (df_info["rtmax"] > upper_bound_rtmax) | 
        (df_info["rtmax"] < lower_bound_rtmax)
    )
    
    # Filter anomalous rows
    anomalies = df_info[mask]
    
    # Construct the result DataFrame directly from anomalous segments
    result_df = pd.DataFrame({
        "quan": anomalies["mw ID"],
        "SampleID": anomalies["sampleID"],
        "rtmin": medians.loc[mask, "rtmin"],
        "rtmax": medians.loc[mask, "rtmax"],
        "weight": medians.loc[mask, "weight"],
        "peak_class": anomalies["peak_class"],
        "rtmin_error": (anomalies["rtmin"] - medians.loc[mask, "rtmin"]).abs(),
        "rtmax_error": (anomalies["rtmax"] - medians.loc[mask, "rtmax"]).abs(),
        "rt": anomalies["rt"]
    }).reset_index(drop=True)

    # Perform parallel refinding and deduplicate the result using sets
    refind_result = refind_peaks_parallel_single(
        threads, result_df, Refind, peak_matrix, dataset_dataframe, peak_information, num_epoch
    )
    
    # Deduplicate keeping list structure
    refind_result = list(set(tuple(row) for row in refind_result))
    
    # Update information and matrix
    peak_information, peak_matrix = update_peak_information(peak_information, refind_result, peak_matrix)

    return peak_information, peak_matrix