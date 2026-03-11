# -*- coding: utf-8 -*-
"""
@author: liutong
Modified for GitHub release: Standardized formatting, removed unused imports, 
and parameterized hardcoded multiprocessing thread counts.
"""

import bisect
import traceback
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
import pymzml
from natsort import natsorted


# ==========================================
# Utility Functions
# ==========================================
def get_files(path, suffix):
    """
    Get all files with the specified suffix in a directory.
    
    Args:
        path (str/Path): Directory path containing data files.
        suffix (str): File extension (e.g., 'mzML').
    
    Returns:
        list: List of file paths sorted naturally.
    """
    try:
        p = Path(path).resolve()
        if not p.is_dir():
            raise NotADirectoryError(f"'{path}' is not a valid directory.")
        paths = [f for f in p.rglob(f"*.{suffix}")]
        return natsorted(paths)
    except Exception as e:
        error_type = type(e).__name__
        error_traceback = traceback.format_exc()
        print(f"[Error] get_files failed - Type: {error_type}, Message: {e}")
        print(f"[Traceback]:\n{error_traceback}")
        return []


def get_closest(mzmean, mz, pos):
    """
    Find the index of the closest m/z value to the target.
    
    Args:
        mzmean (np.ndarray): Array of m/z values.
        mz (float): Target m/z value.
        pos (int): Initial search position obtained via searchsorted.
    
    Returns:
        int: Index of the closest m/z value.
    """
    if pos == len(mzmean):
        res = pos - 1
    elif pos == 0:
        res = pos
    else:
        # Determine whether the left or right neighbor is closer
        res = pos if (mzmean[pos] - mz) < (mz - mzmean[pos - 1]) else pos - 1
    return res


def calc_coordinate(info, intensity, rt, k, windows_size=2):
    """
    Calculate the Retention Time (RT) window coordinates and extract the 
    corresponding intensity and RT arrays using fast binary search.
    
    Args:
        info (np.ndarray): Array containing target feature information.
        intensity (np.ndarray): Full intensity array.
        rt (np.ndarray): Full retention time array.
        k (int): Current feature index.
        windows_size (float, optional): RT window size. Defaults to 2.
        
    Returns:
        tuple: (calc_intensity, calc_rt) sliced arrays within the RT window.
    """
    t_rt = info[k][2]
    
    # Define boundary limits to prevent out-of-bounds errors
    lrt = t_rt - windows_size / 2 if t_rt - windows_size / 2 > 0 else 0
    rrt = t_rt + windows_size / 2 if t_rt + windows_size / 2 < rt[-1] else rt[-1]
    
    # Use bisect for O(log N) fast searching
    lindex = bisect.bisect_left(rt, lrt)
    rindex = bisect.bisect_right(rt, rrt)
    
    calc_intensity, calc_rt = [], []
    if rindex - lindex >= 0:
        calc_intensity = intensity[lindex:rindex]
        calc_rt = rt[lindex:rindex]
        
    return calc_intensity, calc_rt


# ==========================================
# Core Extraction Logic
# ==========================================
def extract_eic(path, df_info, ppm, rt_window=1.2):
    """
    Extract Extracted Ion Chromatogram (EIC) data from an mzML file.
    
    Args:
        path (Path): Path to the mzML file.
        df_info (pd.DataFrame): DataFrame containing target feature information.
        ppm (float): Mass tolerance in parts per million.
        rt_window (float, optional): Retention time window size. Defaults to 1.2.
    
    Returns:
        dict: Contains sample_name, compound_names, and compound_data arrays.
    """
    _ppm = int(ppm) * 1e-6
    flag = 0
    df_info_arr = df_info.values
    
    with pymzml.run.Reader(str(path)) as run:
        # Pre-allocate matrix for performance
        matrix = np.zeros(((len(df_info_arr)) + 1, run.get_spectrum_count()))
        
        for i, spec in enumerate(run):
            if spec.ms_level == 1:
                _mzs = spec.mz
                _intensities = spec.i
                
                # Standardize scan time to minutes
                if spec.scan_time[1] == 'second':
                    matrix[0, i - flag] = spec.scan_time[0] / 60
                else:
                    matrix[0, i - flag] = spec.scan_time[0]
                    
                # Extract intensities for each target m/z
                for index in range(len(df_info_arr)):
                    f_mz = df_info_arr[index][1]
                    indices = np.searchsorted(_mzs, f_mz)
                    closest = get_closest(_mzs, f_mz, indices)
                    
                    if abs(_mzs[closest] - f_mz) < f_mz * _ppm:
                        matrix[index + 1, i - flag] = _intensities[closest]
                    else:
                        matrix[index + 1, i - flag] = 0
            else:
                flag += 1
                
        # Trim unused pre-allocated space
        matrix = matrix[:, :len(matrix[0]) - flag]
        
        rt_values = matrix[0]
        compounds_count = len(matrix) - 1
        table_rt_min = df_info_arr[:, 2].min()
        table_rt_max = df_info_arr[:, 2].max()
        
        # Validate RT range compatibility
        assert table_rt_min > rt_values.min() and table_rt_max < rt_values.max(), \
            f"Feature table is incompatible with EIC. Min acceptable RT: {rt_values.min()}, Max: {rt_values.max()}"
            
        compound_data = []
        sample_name = path.stem
        
        for index in range(compounds_count):
            intensity_values = matrix[index + 1]
            calc_intensity, calc_rt = calc_coordinate(df_info_arr, intensity_values, rt_values, index)
            
            compound_data.append({
                'rt': calc_rt,
                'intensity': calc_intensity
            })
        
        compound_names = df_info_arr[:, 0]

        return {
            'sample_name': sample_name,
            'compound_names': compound_names,
            'compound_data': compound_data
        }


def read_feature_list(data):
    """
    Read and format target feature list from the raw dataframe.
    """
    try:
        data['mz'] = data['mz'].astype(float)
        data['RT'] = data['RT'].astype(float)
        features_info = data[['Compound Name', 'mz', 'RT']].dropna()
        return features_info
    except Exception as e:
        error_type = type(e).__name__
        print(f"[Error] Failed to read feature list - Type: {error_type}, Message: {e}")
        print(traceback.format_exc())
        return None


def roi_to_dataframe(roi_list):
    """
    Convert ROI (Region of Interest) dict list to a flat 2D list format.
    
    Returns:
        list: [[Sample, Compound, RT list, Intensity list], ...]
    """
    data_list = []
    for roi_data in roi_list:
        sample_name = roi_data['sample_name']
        compound_names = roi_data['compound_names']
        compound_data = roi_data['compound_data']
        
        for i, compound in enumerate(compound_names):
            data_list.append([
                sample_name,
                compound,
                compound_data[i]['rt'].tolist(), 
                compound_data[i]['intensity'].tolist()
            ])
    return data_list


def process_single_file(args):
    """
    Worker function to process a single mzML file in multiprocessing.
    """
    path, features, ppm = args
    try:
        roi_data = extract_eic(path, features, ppm)
        print(f"[*] Successfully processed: {roi_data['sample_name']}")
        return roi_data
    except Exception as e:
        error_type = type(e).__name__
        print(f"[Error] Processing {path.name} failed - Type: {error_type}, Message: {e}")
        print(traceback.format_exc())
        return None


# ==========================================
# Main Execution Pipeline
# ==========================================
def extract_roi(source_path, feature_data, ppm=10, threads=16):
    """
    Main function to extract Region of Interest (ROI) data from multiple mzML files.
    
    Args:
        source_path (str): Directory containing mzML files.
        feature_data (pd.DataFrame): Dataframe containing the feature list.
        ppm (float): Mass tolerance in ppm (default: 10).
        threads (int): Number of CPU threads for multiprocessing (default: 16).
    
    Returns:
        list: Extracted ROI data matrix.
    """
    # 1. Fetch all mzML files
    paths = get_files(source_path, 'mzML')
    if not paths:
        print("[!] No mzML files found in the specified directory.")
        return None
        
    # 2. Read target features
    features = read_feature_list(feature_data)
    if features is None:
        return None
    
    # 3. Prepare tasks for multiprocessing
    process_args = [(path, features, ppm) for path in paths]
    
    # Parameterized process count instead of hardcoded
    n_processes = threads if threads else multiprocessing.cpu_count() - 1
    print(f"[*] Starting parallel ROI extraction using {n_processes} CPU cores...")
    
    with multiprocessing.Pool(processes=n_processes) as pool:
        roi_list = pool.map(process_single_file, process_args)
    
    # 4. Filter out any failed extractions
    roi_list = [roi for roi in roi_list if roi is not None]
    if not roi_list:
        print("[!] ROI extraction yielded no valid results.")
        return None
        
    # 5. Restructure data
    data_list = roi_to_dataframe(roi_list)
    
    # Show metadata summary
    print("\n[*] Data points per sample (Preview):")
    for i, row in enumerate(data_list[:5]):
        sample_name, compound, rt_list, intensity_list = row
        print(f"    - Sample: {sample_name}, Compound: {compound}, Points: {len(rt_list)}")
    
    return data_list