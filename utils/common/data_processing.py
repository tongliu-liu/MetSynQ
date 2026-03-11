# -*- coding: utf-8 -*-
"""
Created on Fri Apr  14 09:38:56 2023

@author: liutong
"""

import os
import copy
import math
import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from typing import List, Tuple, Union, Optional


def find_discontinuous_points(lst: List[int]) -> List[List[int]]:
    """
    Separate discontinuous points into multiple sublists.

    Divides a list of predicted indices into grouped sublists where consecutive 
    numbers indicate the same peak region.

    Args:
        lst (List[int]): List of predicted index tags.

    Returns:
        List[List[int]]: A list containing sublists of continuous peak regions.
    """
    if not lst:
        return []

    sublists = []
    sublist = [lst[0]]
    
    # Loop to find discontinuous areas
    for i in range(1, len(lst)):
        if lst[i] == lst[i - 1] + 1:
            sublist.append(lst[i])
        else:
            sublists.append(sublist)
            sublist = [lst[i]]
            
    # Append the final sublist
    sublists.append(sublist)

    return sublists


def extract_first_last_elements(lst: List[List[int]]) -> List[Tuple[int, int]]:
    """
    Extract the boundary indices (first and last) of peak signal intervals.

    Args:
        lst (List[List[int]]): List of lists containing peak region indices.

    Returns:
        List[Tuple[int, int]]: List of tuples, each containing the start and end indices of a peak.
    """
    return [(sublist[0], sublist[-1]) for sublist in lst if sublist]


def segmentation_peak(peak_all_rt: list, peak_signal: list) -> Tuple[List[list], List[list]]:
    """
    Segment multimodal peaks using the half-peak width maximum method.
    
    If the boundaries of multiple overlapping peaks coincide, this function splits them
    at the local minimum.

    Args:
        peak_all_rt (list): List of retention times corresponding to the peak region.
        peak_signal (list): Intensity values of the peak region.

    Returns:
        Tuple[List[list], List[list]]: A tuple containing:
            - rt_seg: Segmented retention time lists.
            - int_seg: Segmented intensity lists.
    """
    # Smooth the signal using a Savitzky-Golay filter
    smoothed_signal = savgol_filter(peak_signal, 3, 1)
    
    # Identify local minima indices
    local_minima = (np.diff(np.sign(np.diff(smoothed_signal))) > 0).nonzero()[0] + 1

    rt_seg = []
    int_seg = []
    
    # Split the signal at local minima if magnitude conditions are met
    if len(local_minima) > 0:
        start_index = 0
        for k in local_minima:
            a = smoothed_signal[start_index:k + 1]
            b = smoothed_signal[k:]

            # Conditions to justify peak separation
            if (max(a) - a[-1] > 0.7 * (max(a) - min(a)) and
                max(b) - b[0] > 0.7 * (max(b) - min(b)) and
                max(b) >= 0.5 * max(smoothed_signal) and
                max(a) >= 0.5 * max(smoothed_signal)):
                
                rt_seg.append(list(peak_all_rt[start_index:k + 1]))
                int_seg.append(list(peak_signal[start_index:k + 1]))
                
                start_index = k
                rt_seg.append(list(peak_all_rt[start_index:]))
                int_seg.append(list(peak_signal[start_index:]))
                break
        else:
            # If no minima met the condition, append the remaining part
            rt_seg.append(list(peak_all_rt[start_index:]))
            int_seg.append(list(peak_signal[start_index:]))
    else:
        rt_seg.append(list(peak_all_rt))
        int_seg.append(list(peak_signal))

    return rt_seg, int_seg


def peak_to_baseline(signal: List[float], baseline: float, left_boundary: int, right_boundary: int) -> Tuple[int, int]:
    """
    Expand the base of the peak until it reaches the baseline height or starts rising.

    Args:
        signal (List[float]): The complete signal intensity list.
        baseline (float): The calculated baseline of the signal.
        left_boundary (int): Current left boundary index of the peak.
        right_boundary (int): Current right boundary index of the peak.

    Returns:
        Tuple[int, int]: Adjusted left and right boundary indices.
    """
    # Spread right if the left boundary is lower
    if signal[left_boundary] < signal[right_boundary]:
        while right_boundary < len(signal) - 1 and (
            abs(signal[right_boundary] - baseline) >= (1/3) * max(signal[left_boundary:right_boundary])
        ):
            right_boundary += 1
        while right_boundary < len(signal) - 1 and (signal[right_boundary] >= signal[right_boundary + 1]):
            right_boundary += 1

    # Spread left if the right boundary is lower
    if signal[left_boundary] > signal[right_boundary]:
        while left_boundary > 0 and (
            abs(signal[left_boundary] - baseline) >= (1/3) * max(signal[left_boundary:right_boundary])
        ):
            left_boundary -= 1
        while left_boundary > 0 and (signal[left_boundary] >= signal[left_boundary - 1]):
            left_boundary -= 1

    return left_boundary, right_boundary


def calculate_peak_area(intensity: List[float], time: List[float]) -> float:
    """
    Calculate the area of the peak using the trapezoidal integral rule.
    
    The area below the lowest intensity within the boundary is subtracted.
    Utilizes numpy vectorization for faster execution.

    Args:
        intensity (List[float]): List of intensity data points.
        time (List[float]): List of retention time data points.

    Returns:
        float: Calculated peak area.
    """
    if len(intensity) < 2:
        return 0.0

    pi = np.array(intensity)
    pt = np.array(time) * 60.0  # Convert minute to second

    # Trapezoidal integration
    area = np.sum((pi[:-1] + pi[1:]) * np.abs(pt[1:] - pt[:-1]) / 2.0)
    
    # Subtract the rectangular area below the minimum intensity
    area -= np.min(pi) * np.abs(pt[0] - pt[-1])

    return float(area)


def interpolate_data(intensity: List[float], time: List[float], num_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform cubic interpolation on the peak data to achieve a smooth curve.

    Args:
        intensity (List[float]): List of intensity data.
        time (List[float]): List of retention time data.
        num_points (int): Number of points to interpolate.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Interpolated intensity array and interpolated time array.
    """
    f = interp1d(time, intensity, kind='cubic')
    interpolated_time = np.linspace(time[0], time[-1], num_points)
    interpolated_intensity = f(interpolated_time)

    return interpolated_intensity, interpolated_time


def calculate_fwhm(peak_signal: List[float], signal_rt: List[float]) -> float:
    """
    Calculate the Full Width at Half Maximum (FWHM) of a peak signal.

    Args:
        peak_signal (List[float]): List of peak signal intensities.
        signal_rt (List[float]): Corresponding retention times.

    Returns:
        float: The calculated FWHM value.
    """
    max_index = peak_signal.index(max(peak_signal))
    half_max_height = max(peak_signal) / 2.0

    # Search for the left boundary crossing half maximum
    left_index = max_index
    while left_index > 0 and peak_signal[left_index] > half_max_height:
        left_index -= 1

    # Search for the right boundary crossing half maximum
    right_index = max_index
    while right_index < len(peak_signal) - 1 and peak_signal[right_index] > half_max_height:
        right_index += 1

    fwhm = signal_rt[right_index] - signal_rt[left_index]

    return fwhm


def looking_boundaries(signal: List[float], signal_rt: List[float], peak: List[float], rt: List[float]) -> Tuple[List[float], List[float]]:
    """
    Adjust the boundaries of a peak to appropriate positions by trimming and expanding.

    Args:
        signal (List[float]): The complete signal intensity data.
        signal_rt (List[float]): Corresponding retention times for the full signal.
        peak (List[float]): The identified peak's intensity data.
        rt (List[float]): Retention times corresponding to the peak.

    Returns:
        Tuple[List[float], List[float]]: Adjusted peak intensities and retention times.
    """
    left_value = peak[0]
    right_value = peak[-1]
    max_value = max(peak)

    if left_value != max_value and right_value != max_value:
        max_value_index = peak.index(max_value)
        
        # Trim the left side of the peak
        left_min_value = min(peak[:max_value_index + 1])
        left_min_value_index = peak.index(left_min_value)
        if left_value > left_min_value:
            peak = peak[left_min_value_index:]
            rt = rt[left_min_value_index:]

        # Trim the right side of the peak
        max_value_index = peak.index(max_value)
        right_min_value = min(peak[max_value_index:])
        right_min_value_index = peak.index(right_min_value)
        if right_value > right_min_value:
            peak = peak[:right_min_value_index + 1]
            rt = rt[:right_min_value_index + 1]

        # Expand the boundaries within the full signal
        left_value_signal_index = signal_rt.index(rt[0])
        right_value_signal_index = signal_rt.index(rt[-1])

        current_left_index = left_value_signal_index
        while current_left_index > 2 and signal[current_left_index] > signal[current_left_index - 1]:
            current_left_index -= 1

        current_right_index = right_value_signal_index
        while current_right_index < len(signal) - 2 and signal[current_right_index] > signal[current_right_index + 1]:
            current_right_index += 1
            if current_right_index >= len(signal) - 1:
                break

        peak = signal[current_left_index:current_right_index + 1]
        rt = signal_rt[current_left_index:current_right_index + 1]

    return peak, rt


def Half_peak_expansion(signal: List[float], signal_rt: List[float], peak: List[float], rt: List[float]) -> Tuple[List[float], List[float]]:
    """
    Expand peak boundaries if a half-peak cutoff is detected.

    Args:
        signal (List[float]): Complete signal intensity data.
        signal_rt (List[float]): Corresponding retention times for the signal.
        peak (List[float]): Detected peak intensities.
        rt (List[float]): Retention times for the peak.

    Returns:
        Tuple[List[float], List[float]]: Expanded peak intensities and retention times.
    """
    left_value = rt[0]
    right_value = rt[-1]
    left_value_signal_index = signal_rt.index(left_value)
    right_value_signal_index = signal_rt.index(right_value)

    if left_value == max(peak):
        # Expand missing left half of the peak
        current_left_index = left_value_signal_index
        if signal[current_left_index] >= signal[current_left_index - 1]:
            while current_left_index > 1 and signal[current_left_index] > signal[current_left_index - 1]:
                current_left_index -= 1
        else:
            while current_left_index > 1 and signal[current_left_index] < signal[current_left_index - 1]:
                current_left_index -= 1
            while current_left_index > 1 and signal[current_left_index] > signal[current_left_index - 1]:
                current_left_index -= 1

        peak = signal[current_left_index:right_value_signal_index]
        rt = signal_rt[current_left_index:right_value_signal_index]
    
    elif right_value == max(peak):
        # Expand missing right half of the peak
        current_right_index = right_value_signal_index
        if signal[current_right_index] >= signal[current_right_index + 1]:
            while current_right_index < len(signal) - 1 and signal[current_right_index] > signal[current_right_index + 1]:
                current_right_index += 1
        else:
            while current_right_index < len(signal) - 1 and signal[current_right_index] < signal[current_right_index + 1]:
                current_right_index += 1
            while current_right_index < len(signal) - 1 and signal[current_right_index] > signal[current_right_index + 1]:
                current_right_index += 1

        peak = signal[left_value_signal_index:current_right_index + 1]
        rt = signal_rt[left_value_signal_index:current_right_index + 1]

    return peak, rt


def Expand_boundaries(signal: List[float], signal_rt: List[float], left_boundary: int, right_boundary: int, num: float) -> Tuple[int, int]:
    """
    Widen peak boundaries if the relative difference between the max and boundary intensities
    is lower than a specified threshold.

    Args:
        signal (List[float]): The complete signal data.
        signal_rt (List[float]): Retention time data corresponding to the signal.
        left_boundary (int): Current left boundary index.
        right_boundary (int): Current right boundary index.
        num (float): Expansion sensitivity threshold ratio.

    Returns:
        Tuple[int, int]: Adjusted left and right boundary indices.
    """
    peak_ori = signal[left_boundary:right_boundary + 1]
    
    peak = signal[left_boundary:right_boundary + 1]
    left_boundary_ori = left_boundary
    right_boundary_ori = right_boundary

    if len(peak) > 1:
        # Expand left boundary if threshold is not met
        if max(peak) - peak[0] < num * (max(peak) - min(peak)):
            current_left_index = left_boundary
            while current_left_index > 2 and max(peak) - peak[0] < num * (max(peak) - min(peak)):
                current_left_index -= 1
                peak = signal[current_left_index:right_boundary + 1]
                while current_left_index > 2 and signal[current_left_index] > signal[current_left_index - 1]:
                    current_left_index -= 1
                    peak = signal[current_left_index:right_boundary + 1]
            left_boundary = current_left_index

        # Expand right boundary if threshold is not met
        if max(peak) - peak[-1] < num * (max(peak) - min(peak)):
            current_right_index = right_boundary
            while current_right_index < len(signal) - 2 and max(peak) - peak[-1] < num * (max(peak) - min(peak)):
                current_right_index += 1
                peak = signal[left_boundary:current_right_index + 1]
                while current_right_index < len(signal) - 2 and signal[current_right_index] > signal[current_right_index + 1]:
                    current_right_index += 1
                    peak = signal[left_boundary:current_right_index + 1]
            right_boundary = current_right_index
        
        # Revert changes if expanded peak fails sanity checks
        if max(peak) >= max(peak_ori) or \
           max(peak) - peak[0] < num * (max(peak) - min(peak)) or \
           max(peak) - peak[-1] < num * (max(peak) - min(peak)):
            left_boundary = left_boundary_ori
            right_boundary = right_boundary_ori

    return left_boundary, right_boundary

def expand_and_adjust_peak(
    signal: Union[list, np.ndarray], 
    signal_rt: Union[list, np.ndarray], 
    peak: Union[list, np.ndarray], 
    rt: Union[list, np.ndarray], 
    cls: Optional[int] = None, 
    conf: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combined peak expansion and boundary adjustment.

    This function merges half-peak expansion and dynamic boundary adjustments:
    1. Completes the missing half of the peak if necessary.
    2. Adjusts the boundaries based on signal trends and model confidence.
    3. Maintains original execution behavior and mathematical logic strictly.

    Args:
        signal (Union[list, np.ndarray]): Full signal intensity data.
        signal_rt (Union[list, np.ndarray]): Full retention time array.
        peak (Union[list, np.ndarray]): Current peak signal intensity data.
        rt (Union[list, np.ndarray]): Current peak retention times.
        cls (Optional[int], optional): Peak class for conditional boundary adjustment.
        conf (Optional[float], optional): Peak confidence for conditional boundary adjustment.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the adjusted (peak_signal, peak_signal_rt).
    """
    # 1. Standardize inputs to numpy arrays for vectorized operations
    signal = np.asarray(signal, dtype=float)
    signal_rt = np.asarray(signal_rt, dtype=float)
    peak = np.asarray(peak, dtype=float)
    rt = np.asarray(rt, dtype=float)

    # 2. Locate initial boundaries within the full signal
    left_idx = int(np.searchsorted(signal_rt, rt[0]))
    right_idx = int(np.searchsorted(signal_rt, rt[-1]))

    # =================================================================
    # Phase 1: Half Peak Expansion (Complete missing peak halves)
    # =================================================================
    if rt[0] == np.max(peak):
        # Missing left half peak: Expand leftwards
        current_left = left_idx
        if signal[current_left] >= signal[current_left - 1]:
            while current_left > 1 and signal[current_left] > signal[current_left - 1]:
                current_left -= 1
        else:
            while current_left > 1 and signal[current_left] < signal[current_left - 1]:
                current_left -= 1
            while current_left > 1 and signal[current_left] > signal[current_left - 1]:
                current_left -= 1
        left_idx = current_left

    elif rt[-1] == np.max(peak):
        # Missing right half peak: Expand rightwards
        current_right = right_idx
        if signal[current_right] >= signal[current_right + 1]:
            while current_right < len(signal) - 1 and signal[current_right] > signal[current_right + 1]:
                current_right += 1
        else:
            while current_right < len(signal) - 1 and signal[current_right] < signal[current_right + 1]:
                current_right += 1
            while current_right < len(signal) - 1 and signal[current_right] > signal[current_right + 1]:
                current_right += 1
        right_idx = current_right

    # Update peak arrays after expansion
    peak_signal = signal[left_idx:right_idx + 1]
    peak_signal_rt = signal_rt[left_idx:right_idx + 1]

    # =================================================================
    # Phase 2: Looking Boundaries (Adjust based on model confidence)
    # =================================================================
    needs_adjustment = (
        cls is not None 
        and conf is not None 
        and not (cls == 1 and conf >= 0.9) 
        and cls != 0
    )

    if needs_adjustment:
        max_val = np.max(peak_signal)
        left_val = peak_signal[0]
        right_val = peak_signal[-1]

        if left_val != max_val and right_val != max_val:
            max_idx = int(np.argmax(peak_signal))
            
            # 2.1 Left trim
            left_min_idx = int(np.argmin(peak_signal[:max_idx + 1]))
            if left_val > peak_signal[left_min_idx]:
                left_idx += left_min_idx
                
            # 2.2 Right trim
            # Note: Preserved original relative index addition logic exactly
            right_min_idx = max_idx + int(np.argmin(peak_signal[max_idx:]))
            if right_val > peak_signal[right_min_idx]:
                right_idx = left_idx + right_min_idx

            # 2.3 Expand to signal boundaries based on gradient
            current_left = left_idx
            while current_left > 2 and signal[current_left] > signal[current_left - 1]:
                current_left -= 1
                
            current_right = right_idx
            while current_right < len(signal) - 2 and signal[current_right] > signal[current_right + 1]:
                current_right += 1

            # Finalize adjusted peak arrays
            peak_signal = signal[current_left:current_right + 1]
            peak_signal_rt = signal_rt[current_left:current_right + 1]

    return peak_signal, peak_signal_rt

def find_peak_boundaries_by_slope(peak_signal: List[float], peak_rt: List[float]) -> Tuple[List[float], List[float]]:
    """
    Find peak boundaries based on the numerical derivative (slope) of the signal.
    
    The boundaries are placed where the slope drops below 10% of the maximum slope.

    Args:
        peak_signal (List[float]): List of signal intensities.
        peak_rt (List[float]): Corresponding retention times.

    Returns:
        Tuple[List[float], List[float]]: Bounded signal intensities and retention times.
    """
    left_index = 0
    right_index = len(peak_signal)
    
    peak_signal_ori = peak_signal.copy()
    peak_rt_ori = peak_rt.copy()
    
    if len(peak_signal) >= 3:
        # Calculate the maximum absolute slope
        max_slope = max(abs((peak_signal[i + 1] - peak_signal[i]) / (peak_rt[i + 1] - peak_rt[i])) 
                        for i in range(len(peak_signal) - 1))
        index = peak_signal.index(max(peak_signal))

        # Search backward to find left boundary
        for i in range(index, 0, -1):
            slope = abs(peak_signal_ori[i] - peak_signal_ori[i - 1]) / (peak_rt_ori[i] - peak_rt_ori[i - 1])
            if slope > 0.1 * max_slope:
                left_index = i
                break

        # Search forward to find right boundary
        for i in range(index, len(peak_signal_ori) - 1):
            slope = abs(peak_signal_ori[i + 1] - peak_signal_ori[i]) / (peak_rt_ori[i + 1] - peak_rt_ori[i])
            if slope > 0.1 * max_slope:
                right_index = i + 1
                break

        peak_signal = peak_signal_ori[left_index:right_index]
        peak_rt = peak_rt_ori[left_index:right_index]

    return peak_signal, peak_rt


def calculate_sn(signal: List[float], signal_rt: List[float], peak: List[float], rt: List[float]) -> Tuple[float, float, float, List[float], float, float]:
    """
    Calculate multiple variants of the Signal-to-Noise Ratio (SNR).

    Args:
        signal (List[float]): Complete signal data.
        signal_rt (List[float]): Retention times corresponding to the signal.
        peak (List[float]): Extracted peak intensity data.
        rt (List[float]): Retention times of the peak.

    Returns:
        Tuple[float, float, float, List[float], float, float]:
            - sn (float): Primary signal-to-noise ratio.
            - baseline (float): Computed baseline level.
            - sn_2 (float): SNR evaluated outside the peak boundaries.
            - [sn_3, sn_31, sn_32] (List[float]): SNRs assessing the immediate surroundings.
            - sn_4 (float): SNR against adjacent background levels.
            - sn_5 (float): SNR against maximum surrounding background level.
    """
    constant = 1
    start_index = signal_rt.index(rt[0])
    end_index = signal_rt.index(rt[-1])
    expected_noise = [constant * np.sqrt(abs(i)) for i in signal]

    # Dynamically select Savitzky-Golay window size
    if len(signal) >= 50:
        smoothed_data = savgol_filter(signal, 5, 2)
    else:
        smoothed_data = savgol_filter(signal, 3, 1)

    baseline_removed_data = [signal[i] - smoothed_data[i] for i in range(len(signal))]

    # Compute baseline noise characteristics
    noise_ratio = [baseline_removed_data[i] / expected_noise[i] if expected_noise[i] != 0 else 0 
                   for i in range(len(baseline_removed_data))]
    std = np.std(noise_ratio)
    noise_ratio = [x for x in noise_ratio if -2 * std <= x <= 2 * std]  # Filter outliers
    std = np.std(noise_ratio) if noise_ratio else std

    sorted_numbers = sorted(smoothed_data)
    num_to_extract = int(len(smoothed_data) * 0.3)
    baseline = max(sorted_numbers[num_to_extract], 1)
    noise = std * np.sqrt(baseline)
    
    sn = (max(peak) - min(peak)) / noise if noise != 0 else 50.0

    # Calculate sn_2: based on strictly non-peak area
    signal_2 = signal[:start_index] + signal[end_index:]
    num_to_extract_signal_2 = int(len(smoothed_data) * 0.1)
    
    if len(signal_2) > num_to_extract_signal_2 * 2:
        std_2 = np.std(signal_2[num_to_extract_signal_2:-num_to_extract_signal_2])
    else:
        std_2 = np.std(signal_2) if signal_2 else 1.0
        
    sn_2 = (max(peak) - np.mean(signal_2)) / std_2 if std_2 else sn

    # Calculate sn_3: based on tightly local peak surroundings
    if len(signal) <= 35:
        start_slice = signal[max(start_index - 3, 0):start_index]
        end_slice = signal[end_index:min(end_index + 3, len(signal))]
    else:
        start_slice = signal[max(start_index - 5, 0):start_index]
        end_slice = signal[end_index:min(end_index + 5, len(signal))]

    signal_round = start_slice + end_slice
    std_3 = max(signal_round) + 1 if signal_round else 1.0
    sn_3 = max(peak) / std_3

    sn_31 = max(peak) / (max(start_slice) + 1) if start_slice else max(peak)
    sn_32 = max(peak) / (max(end_slice) + 1) if end_slice else max(peak)

    # Calculate sn_4
    add_1 = max(signal[:start_index]) if start_index > 3 else min(peak)
    add_2 = max(signal[end_index:]) if end_index < len(signal) - 3 else min(peak)
    
    if (signal_rt[start_index] - signal_rt[0]) <= 0.2:
        add = add_2
    elif (signal_rt[-1] - signal_rt[end_index]) <= 0.2:
        add = add_1
    else:
        add = min(add_1, add_2)
        
    sn_4 = max(peak) / (add + 1)

    # Calculate sn_5
    sn_5 = max(peak) / max(signal_2) if signal_2 else max(peak)

    return sn, float(baseline), sn_2, [sn_3, sn_31, sn_32], sn_4, sn_5