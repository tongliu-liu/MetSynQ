# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 09:38:56 2023

@author: liutong
"""

import numpy as np
import pandas as pd
import multiprocessing
import concurrent.futures

from utils.common.data_processing import * 
from utils.untargeted.peak_refinder import PeakRefinder


def update_peak_information(peak_information, refind_result, peak_matrix):
    """
    Update peak information based on the refind results and peak matrix data.

    Args:
    - peak_information (dict): Dictionary containing peak information
    - refind_result (list): List of refind results for peaks
    - peak_matrix (pd.DataFrame): DataFrame containing peak matrix data

    Returns:
    - peak_information (dict): Updated peak information after incorporating refind results
    """

    # Create a dictionary to store refind results
    refind_result_dict = {}

    # Populate refind_result_dict with refind results
    for peak_ion in refind_result:
        key = str(peak_ion[0]) + str(peak_ion[1])
        value = [
            peak_ion[2], peak_ion[3], peak_ion[4], peak_ion[5], peak_ion[6], 
            peak_ion[7], peak_ion[8], peak_ion[9], peak_ion[10], peak_ion[11], 
            peak_ion[12], peak_ion[13], peak_ion[14], peak_ion[15], peak_ion[16], 
            str(peak_ion[0]), str(peak_ion[1])
        ]
        refind_result_dict[key] = value
    
    # Update peak information with refind results
    peak_information.update(refind_result_dict)
    
    for index, row in peak_matrix.iterrows():
        for column_name in peak_matrix.columns:
            key = str(column_name) + str(index)
            if key in refind_result_dict:
                peak_matrix.at[index, column_name] = refind_result_dict[key][3]
    
    return peak_information, peak_matrix


def refind_peaks_parallel_single(threads, result_df, Refind, peak_matrix, dataset_dataframe, peak_information):
    """
    Parallel processing to refind peaks using multiple threads.

    Args:
    - threads (int): Number of threads to use for parallel processing
    - result_df (pd.DataFrame): DataFrame containing detection results
    - Refind (PeakRefinder): PeakRefinder object for peak refinding
    - peak_matrix (pd.DataFrame): DataFrame containing peak matrix data
    - dataset_dataframe (pd.DataFrame): DataFrame containing dataset information
    - peak_information (dict): Dictionary containing peak information

    Returns:
    - refind_result (list): List of refound peaks
    """
    # Create PeakRefinder object if not provided (Note: overrides passed argument as per original logic)
    Refind = PeakRefinder()
    
    dataset_dataframe = pd.merge(dataset_dataframe, result_df, on=["quan", "SampleID"], how="inner")
    sub_dfs = dataset_dataframe.values.tolist()

    # Define number of cores and chunk size for parallel processing
    num_cores = threads
    chunk_size = int(np.ceil(len(sub_dfs) / (num_cores * 10)))

    # Split data into chunks for parallel processing
    chunks = [sub_dfs[i:i + chunk_size] for i in range(0, len(sub_dfs), chunk_size)]

    # Perform parallel processing using multiprocessing Pool
    refind_result = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [
            executor.submit(Refind.refind_peak_later_period_signal, batch, peak_matrix, peak_information) 
            for batch in chunks
        ]
        for future in futures:
            refind_result.extend(future.result())

    return refind_result


def detect_anomalies_by_group(threads, peak_information, Refind, peak_matrix, dataset_dataframe):
    """
    Group the DataFrame by a specified column, detect anomalies in another specified column within each group,
    and return the group name and the value of another column for rows with anomalies.

    Parameters:
    threads (int): Number of threads to use.
    peak_information (dict/list): Peak information data.
    Refind (PeakRefinder): Object for refining peaks.
    peak_matrix (pd.DataFrame): DataFrame containing peak matrix data.
    dataset_dataframe (pd.DataFrame): DataFrame containing the dataset.

    Returns:
    tuple: Updated peak_information and peak_matrix.
    """

    # Convert peak_information to a DataFrame and transpose it
    peak_information_dataframe = pd.DataFrame(peak_information)
    peak_information_dataframe = peak_information_dataframe.transpose()
    
    # Rename the columns of the DataFrame
    peak_information_dataframe.columns = [
        'rt', 'rtmin', 'rtmax', 'area', 'sn', 'sn_2', 'sn_3', 'sn_5', 'peak_class',
        'baseline', 'height', 'points', 'min_rt', 'max_rt', 'lr_diff', 'sampleID', 'mw ID'
    ]
    
    # Calculate the weight as the difference between rtmax and rtmin
    peak_information_dataframe["weight"] = peak_information_dataframe["rtmax"] - peak_information_dataframe["rtmin"]
    
    # Reset index and drop the old index column
    peak_information_dataframe.reset_index(inplace=True)
    peak_information_dataframe = peak_information_dataframe.drop(columns=['index'])
    results = []

    # Group the DataFrame by "mw ID"
    grouped = peak_information_dataframe.groupby("mw ID")

    for group_name, group in grouped:

        # Calculate the mean value of the filtered rows
        mean_value_rtmin = group["rtmin"].quantile(0.3)
        mean_value_weight = group["weight"].quantile(0.5)
        mean_value_rtmax = group["rtmax"].quantile(0.7)

        # Define the lower and upper bounds for detecting anomalies
        lower_bound_rtmin = mean_value_rtmin - 0.05
        upper_bound_rtmin = mean_value_rtmin + 0.05
        lower_bound_rtmax = mean_value_rtmax - 0.05
        upper_bound_rtmax = mean_value_rtmax + 0.05
        
        # Identify rows with weight values outside the defined bounds 
        # (Retained exactly as original logic, even if unused directly in the loop below)
        anomalies = group[
            (group["rtmin"] > upper_bound_rtmin) |
            (group["rtmin"] < lower_bound_rtmin) | 
            (group["rtmax"] > upper_bound_rtmax) | 
            (group["rtmax"] < lower_bound_rtmax)
        ]
        
        # Append the group name, sample ID, and mean weight to the results list for each anomaly
        for _, row in group.iterrows():
            results.append([
                group_name, row.sampleID, mean_value_rtmin, mean_value_rtmax, 
                mean_value_weight, row.peak_class, abs(row["rtmin"] - mean_value_rtmin), 
                abs(row["rtmax"] - mean_value_rtmax), row["rt"]
            ])

    # Convert the results list to a DataFrame and rename the columns
    result_df = pd.DataFrame(results)
    result_df.columns = [
        "quan", "SampleID", "rtmin", "rtmax", "weight", 
        "peak_class", "rtmin_error", "rtmax_error", "rt"
    ]
    
    # Call the refind_peaks_parallel_single function with the specified parameters
    refind_result = refind_peaks_parallel_single(threads, result_df, Refind, peak_matrix, dataset_dataframe, peak_information)
    
    # Remove duplicates by converting to a set of tuples and back to a list
    refind_result = list(set(tuple(row) for row in refind_result))
    
    # Update peak information and matrix
    peak_information, peak_matrix = update_peak_information(peak_information, refind_result, peak_matrix)

    return peak_information, peak_matrix