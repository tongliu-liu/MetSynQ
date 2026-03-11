# -*- coding: utf-8 -*-
"""
@author: liutong
Modified for GitHub release: Reduced deep nesting (Arrow Anti-Pattern) via early 
returns and helper functions. Strictly preserved all mathematical and validation logic.
"""

import traceback
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any

from scipy.ndimage import gaussian_filter1d
from utils.common.data_processing import *


class PeakRefinder:
    """
    A class for finding peaks that the initial program might have missed.
    It compensates for the weak peak recognition vulnerabilities of image segmentation algorithms.
    """

    def __init__(self):
        """Initializes parameter thresholds for the peak re-finding algorithm."""
        self.min_area = 1000
        self.max_peak_signal = 500
        self.min_peak_signal = 500
        self.min_sn = 10
        self.min_peak_baseline_ratio = 2
        self.min_peak_edge_difference = 0.2
        self.max_peak_baseline_difference = 1 / 5
        self.min_weight = 0.3
        self.min_peak_distance = 0.5
        self.max_volatile_peak_signal = 3000
        self.min_sn_2 = 7
        self.volatile_peak_signal = 3500
        self.min_sn_4 = 6
        self.min_weight_volatile_peak = 0.15
        self.max_sn_5_volatile_peak = 4

    def refind_peak_later_period(self, groups: List[pd.DataFrame], peak_matrix: pd.DataFrame, 
                                 peak_information_dataframe: pd.DataFrame) -> List[list]:
        """Extracts peak information for reanalysis to retrieve missed peaks."""
        refind_result = []
        
        # Pre-compute medians to avoid recalculating inside the loop (O(1) lookup)
        rtmin_medians = peak_information_dataframe.groupby("mw ID")["rtmin"].median().to_dict()
        rtmax_medians = peak_information_dataframe.groupby("mw ID")["rtmax"].median().to_dict()

        try:
            for group in groups:
                for row in group.sort_values(['SampleID']).itertuples():
                    # Skip if peak information already exists
                    if not pd.isna(peak_matrix.at[row.quan, row.SampleID]):
                        continue

                    target_rt = float(peak_matrix.at[row.quan, "rt"])
                    rtmin_avge = rtmin_medians.get(row.quan, np.nan)
                    rtmax_avge = rtmax_medians.get(row.quan, np.nan)

                    tmp_result = []
                    # 1. Smoothed processing
                    tmp_result.extend(self._process_row_epochs(row, rtmin_avge, rtmax_avge, target_rt))
                    # 2. Unsmoothed processing
                    tmp_result.extend(self._process_row_unsmoothed(row, rtmin_avge, rtmax_avge, target_rt))

                    if tmp_result:
                        # Select the peak with the largest area (index 5 is 'area')
                        best_tmp = max(tmp_result, key=lambda x: x[5])
                        refind_result.append([
                            row.SampleID, row.quan, best_tmp[2], best_tmp[3], best_tmp[4], best_tmp[5], 
                            best_tmp[6], best_tmp[7], best_tmp[8], best_tmp[9], "D", best_tmp[11], 
                            best_tmp[12], best_tmp[13], best_tmp[14], best_tmp[15], best_tmp[16]
                        ])  

            return refind_result          

        except Exception as e:
            print(f"[Error] Process error refind：{e}")
            traceback.print_exc()
            return []

    def refind_peak_later_period_signal(self, groups: List[list], peak_matrix: pd.DataFrame, 
                                        peak_information: dict, num_epoch: int) -> List[list]:
        """Extracts peak information from chromatographic signal groups."""
        refind_result = []
        
        # Pre-compute metrics outside the loop
        df_info = pd.DataFrame(peak_information).T
        df_info.columns = ['rt', 'rtmin', 'rtmax', 'area', 'sn', 'sn_2', 'sn_3', 'sn_5', 'peak_class',
                           'baseline', 'height', 'points', 'min_rt', 'max_rt', 'lr_diff', 'sampleID', 'mw ID']
        
        df_info['rtmin'] = pd.to_numeric(df_info['rtmin'], errors='coerce')
        df_info['rtmax'] = pd.to_numeric(df_info['rtmax'], errors='coerce')
        rtmin_medians = df_info.groupby("mw ID")["rtmin"].median().to_dict()
        rtmax_medians = df_info.groupby("mw ID")["rtmax"].median().to_dict()

        try:
            for group in groups:
                best_peak = self._process_single_signal_group(group, peak_matrix, rtmin_medians, rtmax_medians, num_epoch)
                if best_peak:
                    refind_result.append(best_peak)
            return refind_result          

        except Exception as e:
            print(f"[Error] Boundary Correction Error: {e}")
            traceback.print_exc()
            return []

    # ==========================================
    # Sub-routines for Process Flow (To Flatten Nesting)
    # ==========================================
    
    def _process_row_epochs(self, row, rtmin_avge, rtmax_avge, target_rt):
        """Processes smoothed signal iteratively (cycle 1 to 3)."""
        res = []
        row_int_list = list(row.int)
        row_rt_list = list(row.rt)
        
        for cycle in range(1, 4):
            smoothed_signal = list(gaussian_filter1d(row.int, sigma=cycle))
            local_maxima = self.find_local_maxima(smoothed_signal)
            top_peaks = self.select_top_peaks(local_maxima, smoothed_signal)

            for peak_index in top_peaks:
                left, right = self.find_peak_boundaries(smoothed_signal, peak_index)
                baseline = self.calculate_baseline(smoothed_signal)
                
                left, right = Expand_boundaries(smoothed_signal, row_rt_list, left, right, 2/3)
                peak_signal = row_int_list[left:right + 1]
                peak_all_rt = row_rt_list[left:right + 1]

                if len(peak_signal) >= 6:
                    rt_seg, _ = segmentation_peak(peak_all_rt, peak_signal)
                    if len(rt_seg) > 1:
                        break
                        
                rt, rtmin, rtmax, weight, height, area, sn_list, _ = self.calculate_peak_info(
                    peak_signal, peak_all_rt, row_int_list, row_rt_list
                )

                if not self.is_valid_peak_later_period(row_int_list, row_rt_list, peak_signal, peak_all_rt, 
                                                       baseline, rt, rtmin, rtmax, weight, height, area, sn_list):
                    continue

                if self._passes_overlap_check(rt, rtmin, rtmax, target_rt, rtmin_avge, rtmax_avge, weight, mode="smoothed"):
                    res.append([row.SampleID, row.quan, rt, rtmin, rtmax, area, sn_list[0], sn_list[1], sn_list[2][0], sn_list[4], 
                                baseline, height, len(peak_all_rt), min(row.rt), max(row.rt), peak_signal[-1] - peak_signal[0]])
        return res

    def _process_row_unsmoothed(self, row, rtmin_avge, rtmax_avge, target_rt):
        """Processes unsmoothed signal."""
        res = []
        signal_arr = np.array(row.int)
        row_int_list = list(row.int)
        row_rt_list = list(row.rt)
        
        local_maxima = self.find_local_maxima(signal_arr)
        top_peaks = self.select_top_peaks(local_maxima, row_int_list)

        for peak_index in top_peaks:
            left, right = self.find_peak_boundaries(row_int_list, peak_index)
            baseline = self.calculate_baseline(row_int_list)
            left, right = self.peak_to_baseline(row_int_list, row_rt_list, baseline, left, right)
            left, right = Expand_boundaries(row_int_list, row_rt_list, left, right, 0.6)
            
            peak_signal = row_int_list[left:right + 1]
            peak_all_rt = row_rt_list[left:right + 1]
            peak_signal, peak_all_rt = looking_boundaries(row_int_list, row_rt_list, peak_signal, peak_all_rt)

            if len(peak_signal) >= 6:
                rt_seg, _ = segmentation_peak(peak_all_rt, peak_signal)
                if len(rt_seg) > 1:
                    break
                    
            rt, rtmin, rtmax, weight, height, area, sn_list, _ = self.calculate_peak_info(
                peak_signal, peak_all_rt, row_int_list, row_rt_list
            )

            if not self.is_valid_peak_later_period(row_int_list, row_rt_list, peak_signal, peak_all_rt, 
                                                   baseline, rt, rtmin, rtmax, weight, height, area, sn_list):
                continue
                
            if self._passes_overlap_check(rt, rtmin, rtmax, target_rt, rtmin_avge, rtmax_avge, weight, mode="unsmoothed"):
                res.append([row.SampleID, row.quan, rt, rtmin, rtmax, area, sn_list[0], sn_list[1], sn_list[2][0], sn_list[4], 
                            baseline, height, len(peak_all_rt), min(row.rt), max(row.rt), peak_signal[-1] - peak_signal[0]])
        return res

    def _process_single_signal_group(self, group, peak_matrix, rtmin_medians, rtmax_medians, num_epoch):
        """Handles signal boundary processing logic for a single group."""
        signal_arr = np.array(group[3])
        signal_rt_arr = np.array(group[2])
        target_rt = float(peak_matrix.at[group[1], "rt"])
        
        local_maxima = self.find_local_maxima(signal_arr)
        top_peaks = self.select_top_peaks(local_maxima, list(signal_arr))

        local_minima = (np.diff(np.sign(np.diff(signal_arr))) > 0).nonzero()[0] + 1
        if len(local_minima) == 0:
            return None
            
        intensities_min = signal_arr[local_minima]
        top_peaks_min_ori = local_minima[np.argsort(intensities_min)].tolist()
        
        rtmin_avge = rtmin_medians.get(group[1], np.nan)
        rtmax_avge = rtmax_medians.get(group[1], np.nan)

        tmp_result = []
        for peak_index in top_peaks:
            rt = signal_rt_arr[peak_index]
            if abs(float(rt) - target_rt) > 0.1:
                continue
                
            left_bound, right_bound = self.find_peak_boundaries(list(signal_arr), peak_index)
            baseline = self.calculate_baseline(list(signal_arr))

            left_bound = self._adjust_left_boundary(signal_arr, left_bound, baseline, group[4])
            right_bound = self._adjust_right_boundary(signal_arr, right_bound, baseline, group[5])

            top_peaks_min = [x for x in top_peaks_min_ori if left_bound <= x <= right_bound]
            left_bound, right_bound = self._refine_boundaries_by_loss(
                signal_arr, signal_rt_arr, left_bound, right_bound, group, peak_index, top_peaks_min
            )

            peak_signal = list(signal_arr)[left_bound:right_bound + 1]
            peak_all_rt = list(signal_rt_arr)[left_bound:right_bound + 1]

            if num_epoch == 1:
                peak_signal, peak_all_rt = self._apply_num_epoch_1(
                    signal_arr, signal_rt_arr, peak_signal, peak_all_rt, rtmin_avge, rtmax_avge
                )

            rt_calc, rtmin, rtmax, weight, height, area, sn_list, max_idx = self.calculate_peak_info(
                peak_signal, peak_all_rt, list(signal_arr), list(signal_rt_arr)
            )

            if self.is_valid_peak_later_period_mofity(list(signal_arr), list(signal_rt_arr), peak_signal, peak_all_rt, 
                                                      baseline, rt_calc, rtmin, rtmax, weight, height, area, sn_list):
                tmp_result.append([group[0], group[1], rt_calc, rtmin, rtmax, area, sn_list[0], sn_list[1], sn_list[2][0], 
                                   sn_list[4], peak_signal[max_idx], baseline, height, len(peak_all_rt), 
                                   np.min(signal_rt_arr), np.max(signal_rt_arr), peak_signal[-1] - peak_signal[0]]) 
        
        if not tmp_result:
            return None
            
        best = min(tmp_result, key=lambda x: (abs(x[3] - group[4]) + abs(x[4] - group[5])))
        return [best[0], best[1], best[2], best[3], best[4], best[5], best[6], best[7],
                best[8], best[9], group[7], best[11], best[12], best[13], best[14], best[15], best[16]]

    # ==========================================
    # Micro-Helpers (To Isolate Complex IF statements)
    # ==========================================
    
    def _passes_overlap_check(self, rt, rtmin, rtmax, target_rt, rtmin_avge, rtmax_avge, weight, mode):
        """Consolidates the highly specific overlap bounding logic."""
        if mode == "smoothed":
            cond1 = abs(float(rt) - target_rt) <= 0.1
            cond2 = (rtmin - rtmin_avge <= 0.1) and (rtmax_avge - rtmax <= 0.1) and (abs(float(rt) - target_rt) <= 0.2)
            if not (cond1 or cond2):
                return False
        else: # unsmoothed
            cond1 = abs(float(rt) - target_rt) <= 0.1
            cond2 = (abs(rtmin - rtmin_avge) <= 0.1) and (abs(rtmax - rtmax_avge) <= 0.1)
            if not (cond1 or cond2):
                return False

        overlap_start = max(rtmin, rtmin_avge)
        overlap_end = min(rtmax, rtmax_avge)
        overlap_length = max(0, overlap_end - overlap_start)
        
        overlap_ratio_1 = overlap_length / max(weight, 0.01)
        overlap_ratio_2 = overlap_length / max(rtmax_avge - rtmin_avge, 0.01)
        
        if overlap_ratio_1 <= 0.5 and overlap_ratio_2 <= 0.5:
            # Check edge boundary violation
            if mode == "smoothed" and (abs(float(rt) - target_rt) > 0.05 and (rt < rtmin_avge - 0.05 or rt > rtmax_avge + 0.05)):
                return False
            if mode == "unsmoothed" and (abs(float(rt) - target_rt) > 0.05 and (rt < rtmin_avge or rt > rtmax_avge)):
                return False
                
        return True

    def _adjust_left_boundary(self, signal_arr, left, baseline, target_limit):
        if signal_arr[left] >= baseline:
            while left > 0 and (signal_arr[left] > baseline or signal_arr[left] > target_limit):
                left -= 1
            while left > 0 and signal_arr[left] > signal_arr[left - 1]:
                left -= 1
        return left

    def _adjust_right_boundary(self, signal_arr, right, baseline, target_limit):
        if signal_arr[right] >= baseline:
            while right < len(signal_arr) - 1 and (signal_arr[right] > baseline or signal_arr[right] < target_limit):
                right += 1
            while right < len(signal_arr) - 1 and signal_arr[right] > signal_arr[right + 1]:
                right += 1
        return right

    def _refine_boundaries_by_loss(self, signal_arr, signal_rt_arr, left, right, group, peak_index, top_peaks_min):
        left_loss, left_list = [], []
        right_loss, right_list = [], []
        
        for peaks in top_peaks_min:
            if abs(signal_rt_arr[left] - group[4]) >= 0.05 and peaks < peak_index:
                h = np.max(signal_arr[peaks:right + 1]) - np.min(signal_arr[peaks:right + 1])
                if h > 0 and (np.max(signal_arr[peaks:right + 1]) - signal_arr[peaks]) / h >= 0.4:
                    left = peaks
                    left_loss.append(abs(signal_rt_arr[left] - group[4]))
                    left_list.append(left)

            if abs(signal_rt_arr[right] - group[5]) >= 0.05 and peaks > peak_index:
                h = np.max(signal_arr[left:peaks + 1]) - np.min(signal_arr[left:peaks + 1])
                if h > 0 and (np.max(signal_arr[left:peaks + 1]) - signal_arr[peaks]) / h >= 0.4:
                    right = peaks
                    right_loss.append(abs(signal_rt_arr[right] - group[5]))
                    right_list.append(right)
        
        if len(left_loss) > 1: left = left_list[np.argmin(left_loss)]
        elif len(left_loss) == 1: left = left_list[0]
            
        if len(right_loss) > 1: right = right_list[np.argmin(right_loss)]
        elif len(right_loss) == 1: right = right_list[0]

        arg_min_g5 = np.argmin(np.abs(signal_rt_arr - group[5]))
        arg_min_g4 = np.argmin(np.abs(signal_rt_arr - group[4]))
        
        if len(top_peaks_min) > 0:
            if right > group[5] > signal_rt_arr[max(top_peaks_min)] and arg_min_g5 > left:
                right = arg_min_g5
            if left < group[4] < signal_rt_arr[min(top_peaks_min)] and arg_min_g4 < right:
                left = arg_min_g4
        else:
            if right > group[5] and arg_min_g5 > left: right = arg_min_g5
            if left < group[4] and arg_min_g4 < right: left = arg_min_g4
                
        return left, right

    def _apply_num_epoch_1(self, signal_arr, signal_rt_arr, peak_signal, peak_all_rt, rtmin_avge, rtmax_avge):
        if abs(peak_all_rt[0] - rtmin_avge) > 0.05:
            closest_idx = np.argmin(np.abs(signal_rt_arr - rtmin_avge))
            original_right_idx = list(signal_rt_arr).index(peak_all_rt[-1])
            if original_right_idx - closest_idx >= 4:
                peak_signal = list(signal_arr)[closest_idx:original_right_idx + 1]
                peak_all_rt = list(signal_rt_arr)[closest_idx:original_right_idx + 1]

        if abs(peak_all_rt[-1] - rtmax_avge) > 0.05:
            closest_idx = np.argmin(np.abs(signal_rt_arr - rtmax_avge))
            original_left_idx = list(signal_rt_arr).index(peak_all_rt[0])
            if closest_idx - original_left_idx >= 4:
                peak_signal = list(signal_arr)[original_left_idx:closest_idx + 1]
                peak_all_rt = list(signal_rt_arr)[original_left_idx:closest_idx + 1]

        left = list(signal_rt_arr).index(peak_all_rt[0])
        right = list(signal_rt_arr).index(peak_all_rt[-1])

        while left > 0 and signal_arr[left] > signal_arr[left - 1]: left -= 1
        while right < len(signal_arr) - 1 and signal_arr[right] > signal_arr[right + 1]: right += 1
        while left > 0 and signal_arr[left] > signal_arr[left + 1]: left += 1
        while right < len(signal_arr) - 1 and signal_arr[right] > signal_arr[right - 1]: right -= 1
        
        return list(signal_arr)[left:right + 1], list(signal_rt_arr)[left:right + 1]

    # ==========================================
    # Core Mathematical & Baseline Validations
    # ==========================================
    
    def find_local_maxima(self, signal: np.ndarray) -> np.ndarray:
        return (np.diff(np.sign(np.diff(np.array(signal)))) < 0).nonzero()[0] + 1

    def select_top_peaks(self, local_maxima: np.ndarray, signal: List[float]) -> List[int]:
        if len(local_maxima) > 0:
            intensities = np.array(signal)[local_maxima]
            top_peaks_indices = np.argsort(intensities)[::-1]
            return local_maxima[top_peaks_indices].tolist()
        return []

    def find_peak_boundaries(self, signal: List[float], peak_index: int) -> Tuple[int, int]:
        left, right = peak_index, peak_index
        while left > 0 and signal[left] > signal[left - 1]: left -= 1
        while right < len(signal) - 1 and signal[right] > signal[right + 1]: right += 1
        return left, right

    def calculate_baseline(self, signal: List[float]) -> float:
        signal_arr = np.array(signal)
        num_to_extract = int(len(signal_arr) * 0.3)
        if num_to_extract < len(signal_arr):
            return float(max(np.partition(signal_arr, num_to_extract)[num_to_extract], 1.0))
        return float(max(np.max(signal_arr), 1.0))

    def peak_to_baseline(self, signal: List[float], rt: List[float], baseline: float, 
                         left_boundary: int, right_boundary: int) -> Tuple[int, int]:
        if right_boundary - left_boundary >= 2:
            if min(signal[left_boundary], signal[right_boundary]) - baseline >= 1/5 * max(signal[left_boundary : right_boundary]):
                while (right_boundary < len(signal) - 1 and signal[right_boundary] - baseline >= 1/5 * max(signal[left_boundary : right_boundary]) 
                       and rt[right_boundary] - rt[right_boundary] <= 0.3):
                    right_boundary += 1
                while right_boundary < len(signal) - 1 and (signal[right_boundary] > signal[right_boundary + 1]):
                    right_boundary += 1
                while (left_boundary > 0 and signal[left_boundary] - baseline >= 1/5 * max(signal[left_boundary : right_boundary])
                       and rt[right_boundary] - rt[right_boundary] <= 0.3):
                    left_boundary -= 1
                while left_boundary > 0 and (signal[left_boundary] > signal[left_boundary - 1]):
                    left_boundary -= 1
        return left_boundary, right_boundary

    def is_valid_peak_later_period(self, signal: List[float], signal_rt: List[float], peak_signal: List[float], 
                                   peak_all_rt: List[float], baseline: float, rt: float, rtmin: float, rtmax: float, 
                                   weight: float, height: float, area: float, sn_list: List[Any]) -> bool:
        sn, sn_2, sn_3, sn_31, sn_32, sn_5 = sn_list[0], sn_list[1], sn_list[2][0], sn_list[2][1], sn_list[2][2], sn_list[4]
        max_peak = max(peak_signal)

        base_cond = (
            area > self.min_area and max(signal) >= self.max_peak_signal and max_peak > self.min_peak_signal
            and (max_peak - peak_signal[0] > self.min_peak_edge_difference * height 
                 and max_peak - peak_signal[-1] > self.min_peak_edge_difference * height)
            and not (sn < 10 and sn_3 < 1.5)
            and not (sn < 6)
        )

        if not base_cond: return False
        if sn_5 >= 1.5: return True
        if weight >= 0.1 and sn_3 >= 2: return True
        if weight >= 0.1 and sn >= 10 and (sn_31 > 1.5 or sn_32 > 1.5): return True
        if sn_31 >= 2 or sn_32 >= 2: return True
        if sn_3 >= 1.5 and sn_2 >= 3: return True
        return False

    def is_valid_peak_later_period_mofity(self, signal: List[float], signal_rt: List[float], peak_signal: List[float], 
                                          peak_all_rt: List[float], baseline: float, rt: float, rtmin: float, rtmax: float, 
                                          weight: float, height: float, area: float, sn_list: List[Any]) -> bool:
        sn, sn_3 = sn_list[0], sn_list[2][0]
        max_peak = max(peak_signal)

        base_cond = (
            area > self.min_area and max(signal) >= self.max_peak_signal and max_peak > self.min_peak_signal
            and (max_peak - peak_signal[0] > 0.2 * height and max_peak - peak_signal[-1] > 0.2 * height)
        )
        return base_cond and sn_3 >= 0.1 and sn > 5
    
    def calculate_peak_info(self, peak_signal: List[float], peak_all_rt: List[float], 
                            signal: List[float], signal_rt: List[float]) -> Tuple:
        max_index = peak_signal.index(max(peak_signal))
        rt = peak_all_rt[max_index] 
        rtmin, rtmax = peak_all_rt[0], peak_all_rt[-1]
        weight = rtmax - rtmin 
        height = max(peak_signal) - min(peak_signal) 
        area = calculate_peak_area(peak_signal, peak_all_rt) 
        sn, baseline, sn_2, sn_3, sn_4, sn_5 = calculate_sn(signal, signal_rt, peak_signal, peak_all_rt) 
        
        return rt, rtmin, rtmax, weight, height, area, [sn, sn_2, sn_3, sn_4, sn_5], max_index