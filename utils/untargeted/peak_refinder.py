# -*- coding: utf-8 -*-
"""
@author: liutong
Modified for GitHub release: Reduced deep nesting, improved readability, 
and maintained strict mathematical logic.
"""

import traceback
import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter1d
from utils.common.data_processing import Expand_boundaries, calculate_peak_area, calculate_sn


class PeakRefinder:
    """
    This is a class for finding peaks that the program has missed.
    This class is used to make up the weak peak recognition vulnerability of image segmentation algorithm.
    """

    def __init__(self):
        """Initialize parameters for peak re-finding algorithm."""
        self.min_area = 500
        self.max_peak_signal = 100
        self.min_peak_signal = 100
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

    def refind_peak_later_period(self, groups, peak_matrix, peak_information_dataframe, all_ions):
        """
        Extracts peak information for reanalysis.
        Retrieves peaks that have not been selected in some samples.
        """
        try:
            refind_result = []
            for group in groups:
                first_group = group.sort_values(['SampleID'])
                
                for row in first_group.itertuples(index=False):
                    # Check if the current sample and quan pair has no peak information
                    if not np.isnan(peak_matrix[row.SampleID][row.quan]):
                        continue

                    # Cache list conversions
                    row_int_arr = np.array(row.int)
                    row_int_list = list(row.int)
                    row_rt_list = list(row.rt)
                    
                    tmp_result = self._process_epochs_for_row(
                        row_int_arr, row_int_list, row_rt_list, row
                    )
                    
                    if not tmp_result:
                        continue

                    # Select the peak with the largest area (closest to theory rt in logic)
                    theory_rt = float(all_ions[row.quan])
                    best = min(tmp_result, key=lambda x: abs(x[2] - theory_rt))
                    
                    refind_result.append([
                        row.SampleID, row.quan, best[2], best[3], best[4], best[5], best[6], 
                        best[7], best[8], best[9], "D", best[10], best[11], best[12], 
                        best[13], best[14], best[15]
                    ])  
                    
            return refind_result

        except Exception as e:
            print(f"[Error] Process error refind：{e}")
            traceback.print_exc()
            return []

    def _process_epochs_for_row(self, row_int_arr, row_int_list, row_rt_list, row):
        """Helper to reduce nesting: processes sigma epochs for a single row."""
        tmp_result = []
        for epoch in range(2, 5):
            smoothed_signal = gaussian_filter1d(row_int_arr, sigma=epoch)
            smoothed_signal_list = list(smoothed_signal)
            
            local_maxima = self.find_local_maxima(smoothed_signal)
            top_peaks = self.select_top_peaks(local_maxima, smoothed_signal_list)

            for peak_index in top_peaks:
                left_bound, right_bound = self.find_peak_boundaries(smoothed_signal, peak_index)
                baseline = self.calculate_baseline(smoothed_signal_list)
                
                left_bound, right_bound = Expand_boundaries(
                    smoothed_signal_list, row_rt_list, left_bound, right_bound, 2/3
                )
                
                peak_signal = row_int_list[left_bound:right_bound + 1]
                peak_all_rt = row_rt_list[left_bound:right_bound + 1]

                rt, rtmin, rtmax, weight, height, area, sn_list, _ = self.calculate_peak_info(
                    peak_signal, peak_all_rt, row_int_list, row_rt_list
                )

                if self.is_valid_peak_later_period(
                    row_int_list, row_rt_list, peak_signal, peak_all_rt, baseline, 
                    rt, rtmin, rtmax, weight, height, area, sn_list
                ):
                    tmp_result.append([
                        row.SampleID, row.quan, rt, rtmin, rtmax, area, sn_list[0], 
                        sn_list[1], sn_list[2][0], sn_list[4], baseline, height, 
                        len(peak_all_rt), min(row_rt_list), max(row_rt_list), 
                        peak_signal[-1] - peak_signal[0]
                    ])
        return tmp_result

    def refind_peak_later_period_signal(self, groups, peak_matrix, peak_information):
        """
        Extracts peak information from a set of chromatographic signal groups.
        """
        try:
            refind_result = []
            
            # PERFORMANCE OPTIMIZATION: Extract DataFrame creation OUTSIDE the loop
            peak_info_df = pd.DataFrame(peak_information).transpose()
            peak_info_df.columns = [
                'rt', 'rtmin', 'rtmax', 'area', 'sn', 'sn_2', 'sn_3', 'sn_5', 'peak_class',
                'baseline', 'height', 'points', 'min_rt', 'max_rt', 'lr_diff', 'sampleID', 'mw ID'
            ]
            
            for group in groups:
                best_peak = self._process_single_signal_group(group, peak_info_df)
                if best_peak:
                    refind_result.append(best_peak)
                    
            return refind_result          

        except Exception as e:
            print(f"[Error] Process error in refind_peak_later_period_signal：{e}")
            traceback.print_exc()
            return []

    def _process_single_signal_group(self, group, peak_info_df):
        """Helper to handle the logic for a single signal group to reduce nesting."""
        single_info = peak_info_df[peak_info_df["mw ID"] == group[1]]
        rt_avge = single_info["rt"].quantile(0.5)
        
        group[4] += (group[10] - rt_avge)  
        group[5] += (group[10] - rt_avge)
        
        group_3_arr = np.array(group[3])
        group_3_list = list(group[3])
        signal_rt = np.array(group[2])
        
        signal = gaussian_filter1d(group_3_arr, sigma=1)
        signal_list = list(signal)
        signal_rt_list = list(signal_rt)
        
        local_maxima = self.find_local_maxima(signal)
        top_peaks = self.select_top_peaks(local_maxima, signal_list)

        local_minima = (np.diff(np.sign(np.diff(signal))) > 0).nonzero()[0] + 1
        if len(local_minima) == 0:
            return None

        intensities_min = [signal[i] for i in local_minima]
        top_peaks_min_ori = [local_minima[i] for i in np.argsort(intensities_min)]
        
        tmp_result = []
        for peak_index in top_peaks:
            rt = signal_rt[peak_index]
            if abs(float(rt) - group[10]) > 0.6:
                continue
                
            left_bound, right_bound = self.find_peak_boundaries(signal_list, peak_index)
            baseline = self.calculate_baseline(signal_list)

            left_bound = self._adjust_left_boundary(signal, left_bound, baseline, group[4])
            right_bound = self._adjust_right_boundary(signal, right_bound, baseline, group[5])

            top_peaks_min = [x for x in top_peaks_min_ori if left_bound <= x <= right_bound]

            left_bound, right_bound = self._refine_boundaries_by_loss(
                signal, signal_rt, left_bound, right_bound, group, peak_index, top_peaks_min
            )

            # Extract peak signal and trim tails if necessary
            peak_signal, peak_all_rt = self._trim_peak_tails(
                group_3_list, signal_rt_list, left_bound, right_bound
            )

            # Calculate and Validate
            rt, rtmin, rtmax, weight, height, area, sn_list, max_idx = self.calculate_peak_info(
                peak_signal, peak_all_rt, group_3_list, signal_rt_list
            )

            if self.is_valid_peak_later_period_mofity(
                signal_list, signal_rt_list, peak_signal, peak_all_rt, baseline, 
                rt, rtmin, rtmax, weight, height, area, sn_list
            ):
                tmp_result.append([
                    group[0], group[1], rt, rtmin, rtmax, area, sn_list[0], sn_list[1], 
                    sn_list[2][0], sn_list[4], peak_signal[max_idx], baseline, height, 
                    len(peak_all_rt), min(signal_rt), max(signal_rt), peak_signal[-1] - peak_signal[0]
                ]) 
        
        if not tmp_result:
            return None
            
        best = min(tmp_result, key=lambda x: abs(x[2] - group[10]))
        return [
            best[0], best[1], best[2], best[3], best[4], best[5], best[6], best[7],
            best[8], best[9], group[7], best[11], best[12], best[13], best[14], 
            best[15], best[16]
        ]

    # -------------- Micro-Helpers for Boundary Adjustment --------------
    def _adjust_left_boundary(self, signal, left, baseline, target_limit):
        if signal[left] >= baseline:
            while left > 0 and (signal[left] > baseline or signal[left] > target_limit):
                left -= 1
            while left > 0 and signal[left] > signal[left - 1]:
                left -= 1
        return left

    def _adjust_right_boundary(self, signal, right, baseline, target_limit):
        if signal[right] >= baseline:
            while right < len(signal) - 1 and (signal[right] > baseline or signal[right] < target_limit):
                right += 1
            while right < len(signal) - 1 and signal[right] > signal[right + 1]:
                right += 1
        return right

    def _refine_boundaries_by_loss(self, signal, signal_rt, left, right, group, peak_index, top_peaks_min):
        left_loss, left_list = [], []
        right_loss, right_list = [], []
        
        for peaks in top_peaks_min:
            if abs(signal_rt[left] - group[4]) >= 0.05 and peaks < peak_index:
                h = max(signal[peaks:right + 1]) - min(signal[peaks:right + 1])
                if h != 0 and (max(signal[peaks:right + 1]) - signal[peaks]) / h >= 0.4:
                    left = peaks
                    left_loss.append(abs(signal_rt[left] - group[4]))
                    left_list.append(left)

            if abs(signal_rt[right] - group[5]) >= 0.05 and peaks > peak_index:
                h = max(signal[left:peaks + 1]) - min(signal[left:peaks + 1])
                if h != 0 and (max(signal[left:peaks + 1]) - signal[peaks]) / h >= 0.4:
                    right = peaks
                    right_loss.append(abs(signal_rt[right] - group[5]))
                    right_list.append(right)
        
        if left_loss:
            left = left_list[left_loss.index(min(left_loss))]
        if right_loss:
            right = right_list[right_loss.index(min(right_loss))]

        arg_min_g5 = np.argmin(np.abs(signal_rt - group[5]))
        arg_min_g4 = np.argmin(np.abs(signal_rt - group[4]))
        
        if len(top_peaks_min) > 0:
            if right > group[5] > signal_rt[max(top_peaks_min)] and arg_min_g5 > left:
                right = arg_min_g5
            if left < group[4] < signal_rt[min(top_peaks_min)] and arg_min_g4 < right:
                left = arg_min_g4
        else:
            if right > group[5] and arg_min_g5 > left:
                right = arg_min_g5
            if left < group[4] and arg_min_g4 < right:
                left = arg_min_g4
                
        return left, right

    def _trim_peak_tails(self, group_3_list, signal_rt_list, left, right):
        peak_signal = group_3_list[left:right + 1]
        peak_all_rt = signal_rt_list[left:right + 1]

        maxvalue_index = peak_signal.index(max(peak_signal))
        
        # Trim left
        left_minvalue = min(peak_signal[: maxvalue_index + 1])
        if peak_signal[0] > left_minvalue:
            idx = peak_signal.index(left_minvalue)
            peak_signal = peak_signal[idx:]
            peak_all_rt = peak_all_rt[idx:]
            
        # Re-calculate max index for right trim
        maxvalue_index = peak_signal.index(max(peak_signal))
        
        # Trim right
        right_minvalue = min(peak_signal[maxvalue_index:])
        if peak_signal[-1] > right_minvalue:
            # We add maxvalue_index because the min was calculated on a slice
            idx = peak_signal[maxvalue_index:].index(right_minvalue) + maxvalue_index
            peak_signal = peak_signal[: idx + 1]
            peak_all_rt = peak_all_rt[: idx + 1]
            
        return peak_signal, peak_all_rt

    # -------------- Core Math and Validation --------------
    def find_local_maxima(self, signal):
        """Find local maxima in the signal."""
        return (np.diff(np.sign(np.diff(np.array(signal)))) < 0).nonzero()[0] + 1

    def select_top_peaks(self, local_maxima, signal):
        """Select the top peaks based on intensities."""
        if len(local_maxima) > 0:
            intensities = [signal[i] for i in local_maxima]
            top_peaks_indices = np.argsort(intensities)[::-1]
            return [local_maxima[i] for i in top_peaks_indices]
        return []

    def find_peak_boundaries(self, signal, peak_index):
        """Find the boundaries of the peak based on gradient."""
        left = right = peak_index
        while left > 0 and signal[left] > signal[left - 1]:
            left -= 1
        while right < len(signal) - 1 and signal[right] > signal[right + 1]:
            right += 1
        return left, right

    def calculate_baseline(self, signal):
        """Calculate the baseline of the signal (30th percentile approximation)."""
        sorted_numbers = sorted(signal)
        num_to_extract = int(len(signal) * 0.3)
        return max(sorted_numbers[num_to_extract], 1)

    def is_valid_peak_later_period(self, signal, signal_rt, peak_signal, peak_all_rt, baseline, rt, 
                                   rtmin, rtmax, weight, height, area, sn_list):
        """Check if the peak is valid."""
        sn, sn_2, sn_3, sn_31, sn_32, sn_5 = sn_list[0], sn_list[1], sn_list[2][0], sn_list[2][1], sn_list[2][2], sn_list[4]
        max_peak, max_signal = max(peak_signal), max(signal)

        base_cond = (
            area > self.min_area 
            and max_signal >= self.max_peak_signal 
            and max_peak > self.min_peak_signal
            and (max_peak - peak_signal[0] > self.min_peak_edge_difference * height)
            and (max_peak - peak_signal[-1] > self.min_peak_edge_difference * height)
            and not (sn < 10 and sn_3 < 1.5)
            and not (sn < 5 or sn_2 < 0)
            and not (sn < 40 and sn_2 < 0.5)
            and not (max_peak / max_signal < 0.05)
        )

        if not base_cond:
            return False
            
        if sn_5 >= 1.:
            return True
        if weight >= 0.1 and sn_3 >= 2:
            return True
        if weight >= 0.1 and sn >= 10 and (sn_31 > 1.5 or sn_32 > 1.5):
            return True
        if sn_31 >= 2 or sn_32 >= 2:
            return True
        if sn_3 >= 1.5 and sn_2 >= 3:
            return True

        return False

    def is_valid_peak_later_period_mofity(self, signal, signal_rt, peak_signal, peak_all_rt, baseline, rt, 
                                   rtmin, rtmax, weight, height, area, sn_list):
        """Check if the modified peak is valid."""
        sn, sn_2, sn_3 = sn_list[0], sn_list[1], sn_list[2][0]
        max_peak, max_signal = max(peak_signal), max(signal)

        base_cond = (
            area > self.min_area 
            and max_signal >= self.max_peak_signal 
            and max_peak > self.min_peak_signal
            and (max_peak - peak_signal[0] > 0.2 * height)
            and (max_peak - peak_signal[-1] > 0.2 * height)
        )

        return base_cond and sn_3 >= 0.1 and sn > 5 and sn_2 > 0

    def calculate_peak_info(self, peak_signal, peak_all_rt, signal, signal_rt):
        """Calculate mathematical properties and SN ratios of the peak."""
        max_index = peak_signal.index(max(peak_signal))
        rt = peak_all_rt[max_index]
        rtmin = peak_all_rt[0]
        rtmax = peak_all_rt[-1]
        weight = rtmax - rtmin
        height = max(peak_signal) - min(peak_signal)
        area = calculate_peak_area(peak_signal, peak_all_rt)
        
        sn, baseline, sn_2, sn_3, sn_4, sn_5 = calculate_sn(
            list(signal), list(signal_rt), peak_signal, peak_all_rt
        )  
        sn_list = [sn, sn_2, sn_3, sn_4, sn_5]

        return rt, rtmin, rtmax, weight, height, area, sn_list, max_index