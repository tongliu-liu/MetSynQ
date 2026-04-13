# -*- coding: utf-8 -*-
"""
@author: liutong
Modified for GitHub release: Modularized and removed hardcoded paths.
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import re
import pickle
import argparse
import traceback
from copy import deepcopy

import numpy as np
import pandas as pd
import torch as t
import concurrent.futures
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from torch.utils.data import Dataset
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from ultralytics import YOLO
from utils.common.read_data import read_mzml_data
from utils.common.data_processing import (
    Half_peak_expansion,
    looking_boundaries,
    calculate_peak_area,
    calculate_sn
)

def parse_arguments():
    description = 'Converts the training set into a data form acceptable to the training program. \n'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        '-i', '--indir', 
        dest='indir', 
        required=True, 
        type=str, 
        help='Input folder, project folder. [must be specified]'
    )
    parser.add_argument(
        '--threads', 
        default=16, 
        type=int, 
        metavar='N', 
        help='number of data loading workers (default: 16)'
    )
    return parser.parse_args()


def find_closest_index(lst, target):
    """Gets the index of the minimum difference."""
    min_diff = float('inf')
    closest_index = -1 

    for i, num in enumerate(lst):
        diff = abs(num - target)
        if diff < min_diff:
            min_diff = diff
            closest_index = i

    return closest_index


def split_into_batches(data, batch_size):
    """Splits a dataset into batches of a specified size."""
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]


class classfiyDataset(Dataset):
    """ Dataset object for classify of data loading """

    def __init__(self, data_batches, device="cpu", pad_length=256):
        self.batches = data_batches
        self.pad_length = pad_length
        self.length = len(self.batches)
        self.device = device

    def __len__(self):
        return self.length

    def _interpolate(self, peak):
        peak = deepcopy(peak)
        points = len(peak[0][:, 1])
        interpolate = interp1d(np.arange(points), peak[0][:, 1], kind='linear')
        ints = interpolate(np.arange(self.pad_length) / (self.pad_length - 1.) * (points - 1.))
        return ints

    def __getitem__(self, index):
        if index >= self.length:
            raise IndexError
        peaks = self.batches[index]
        x = self._interpolate(peaks)
        x = t.tensor(x, dtype=t.float32, device=self.device).view(1, -1)
        x = x / t.max(x)
        y = t.zeros(256, dtype=t.float)
        y = t.tensor(y, dtype=t.long, device=self.device)
        return x, y


def process_sample(samples, class_dict, all_ions):
    """Processes a list of samples to extract relevant features and classify them."""
    try:  
        data_list = []
        sample_info = []
        result_list = []
        signal_list_batch = []
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.abspath(os.path.join(current_dir, "..", "..", "weights", "best.pt"))
        
        model = YOLO(model_path) 
        num_sample = 0
        
        for sample in samples:
            signal = list(sample[7])
            rt = list(sample[6])
            
            fig, ax = plt.subplots()
            fig.set_size_inches(224/fig.dpi, 224/fig.dpi) 
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

            ax.plot(rt, signal, color='black')
            ax.axis('off')

            canvas = FigureCanvas(fig)
            canvas.draw()

            img_array = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape((224, 224, 4))
            img_array = img_array[:, :, :3] 

            plt.close(fig)
            del(ax)
            fig.clf()
            plt.close()

            sampleID = list([sample[0], sample[1]])

            if "_P" in sampleID[0] and not sampleID[0].endswith('_P'):
                sampleID[0] = sampleID[0].split("_P", 1)[0] + "_P"
            if ("_N" in sampleID[0] or "_HN" in sampleID[0]) and not sampleID[0].endswith('_N') and not sampleID[0].endswith('_HN'):
                if "_HN" in sampleID[0]:
                    sampleID[0] = sampleID[0].split("_HN", 1)[0] + "_HN"
                if "_N" in sampleID[0]:
                    sampleID[0] = sampleID[0].split("_N", 1)[0] + "_N"

            data_list.append(img_array)
            sample_info.append((sampleID[0], sampleID[1]))
            signal_list_batch.append((rt, signal))
            num_sample += 1

            if len(data_list) % 1000 == 0 or num_sample == len(samples):
                results = model.predict(data_list, device="cpu")
                data_list = []

                for num_1, (sampleID_info, (rt_batch, signal_batch), result) in enumerate(zip(sample_info, signal_list_batch, results)):
                    boxes = result.boxes
                    for num, (pix, cls, conf) in enumerate(zip(boxes.xywh, boxes.cls, boxes.conf)):
                        ID = f"{sampleID_info[0] + sampleID_info[1]}_{num}"
                        x_start = max((pix[0] - pix[2] / 2 - 10.18181818) / (213.81818182 - 10.18181818), 0)
                        x_end = min((pix[0] + pix[2] / 2 - 10.18181818) / (213.81818182 - 10.18181818), 1)
                        
                        if cls != 3:
                            rtmin_index = find_closest_index(rt_batch, rt_batch[0] + (rt_batch[-1] - rt_batch[0]) * x_start)
                            rtmax_index = find_closest_index(rt_batch, rt_batch[0] + (rt_batch[-1] - rt_batch[0]) * x_end)
                            
                            peak_signal = signal_batch[rtmin_index: rtmax_index + 1]
                            peak_signal_rt = rt_batch[rtmin_index: rtmax_index + 1]
                            
                            peak_signal, peak_signal_rt = Half_peak_expansion(signal_batch, rt_batch, peak_signal, peak_signal_rt)
                            peak_signal, peak_signal_rt = looking_boundaries(signal_batch, rt_batch, peak_signal, peak_signal_rt)
                            
                            rtmin, rtmax = peak_signal_rt[0], peak_signal_rt[-1]
                            peak_area = calculate_peak_area(peak_signal, peak_signal_rt)
                            peak_rt = peak_signal_rt[np.argmax(peak_signal)]
                            peak_weight = rtmax - rtmin
                            peak_height = max(peak_signal) - min(peak_signal)
                            sn, baseline, sn_2, sn_3, sn_4, sn_5 = calculate_sn(signal_batch, rt_batch, peak_signal, peak_signal_rt)
                            
                            sorted_signal = sorted(signal_batch)
                            baseline = max(sorted_signal[int(len(signal_batch) * 0.4)], 1)
                            ion_type = "Negative" if sampleID_info[0].endswith("N") else ("Positive" if sampleID_info[0].endswith("P") else "no mode")
                            
                            mix_threshold = True
                            if "mix" in sampleID_info[0] and (
                                ((rtmin == rt_batch[0] or rtmin == rt_batch[1]) and peak_signal[0] - peak_signal[-1] >= 0.1 * peak_height) 
                                or ((rtmax == rt_batch[-1] or rtmax == rt_batch[-2]) and peak_signal[-1] - peak_signal[0] >= 0.1 * peak_height)
                                or ((max(peak_signal) - min(peak_signal)) <= 0.1 * (max(signal_batch) - min(signal_batch)))
                                or (max(peak_signal) <= 5000 and baseline == 50 and sn_5 <= 1.5)
                            ):
                                mix_threshold = False
                            
                            if all_ions[sampleID_info[1]].lower() in ["negative", "positive"]:
                                ions_model = all_ions[sampleID_info[1]].lower()
                            elif all_ions[sampleID_info[1]].lower() == "hilic":
                                ions_model = "negative"

                            if ion_type.lower() == ions_model or ion_type.lower() == "no mode":
                                peak_class = class_dict[int(cls.item())]
                                
                                base_result = [
                                    None, sampleID_info[1], peak_rt, rtmin, rtmax, peak_weight, 
                                    peak_height, peak_area, sn, sn_2, sn_3[0], sn_5, 
                                    len(peak_signal), len(signal_batch), peak_class, 
                                    conf.item(), baseline, min(rt_batch), max(rt_batch), 
                                    peak_signal[-1] - peak_signal[0]
                                ]

                                if peak_class in {"A", "B"} and peak_area >= 1000 and len(peak_signal) >= 3 and mix_threshold:
                                    
                                    ignore_cond = (
                                        max(peak_signal) <= 5000 and peak_weight <= 0.15 
                                        and len(peak_signal) < 6 and min(peak_signal) == 50 
                                        and sn_5 < 2 and peak_signal[0] == peak_signal[-1] 
                                        and baseline == 50 and "mix" not in sampleID_info[0]
                                    )
                                    
                                    if not ignore_cond:
                                        if ion_type.lower() == "no mode":
                                            if all_ions[sampleID_info[1]].lower() == "negative":
                                                base_result[0] = sampleID_info[0] + "_N"
                                                result_list.append(base_result)
                                            elif all_ions[sampleID_info[1]].lower() == "positive":
                                                base_result[0] = sampleID_info[0] + "_P"
                                                result_list.append(base_result)
                                        else:
                                            base_result[0] = sampleID_info[0]
                                            result_list.append(base_result)
                                            
                                elif peak_area >= 1000 and len(peak_signal) >= 3 and mix_threshold:
                                    if ion_type.lower() == "no mode":
                                        if all_ions[sampleID_info[1]].lower() == "negative":
                                            base_result[0] = sampleID_info[0] + "_N"
                                            result_list.append(base_result)
                                        elif all_ions[sampleID_info[1]].lower() == "positive":
                                            base_result[0] = sampleID_info[0] + "_P"
                                            result_list.append(base_result)
                                    else:
                                        base_result[0] = sampleID_info[0]
                                        result_list.append(base_result)
                                        
                sample_info = []
                signal_list_batch = []
    except Exception as e:
        print(f"Process error peak: {e}")

    return result_list


def run(out_sample, indir, threads):
    """Processes raw data files to filter useful information and apply a YOLO model."""
    output_dir = f"{indir}/temp"
    os.makedirs(output_dir, exist_ok=True)

    all_ions = pd.read_excel('{}/ALL_ions.xlsx'.format(indir))
    all_ions.rename(columns={'Ion mode': 'ion_mode', 'mw ID': 'mwID'}, inplace=True)
    all_ions = pd.Series(all_ions.ion_mode.values, index=all_ions.mwID).to_dict()
    out_sample = [sample for sample in out_sample if np.max(sample[7]) >= 500]
    
    result_list = []
    class_dict = {0: "A", 1: "B", 2: "C", 3: "D"}

    if threads <= 16:
        batches = split_into_batches(out_sample, int(len(out_sample) / 2 + 1))
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(process_sample, batch, class_dict, all_ions) 
                for batch in batches
            ]
            for future in futures:
                result_list.extend(future.result())
    else:
        batches = split_into_batches(out_sample, int(len(out_sample) / 4 + 1))
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(process_sample, batch, class_dict, all_ions) 
                for batch in batches
            ]
            for future in futures:
                result_list.extend(future.result())
    
    columns_list = [
        "sampleID", "mw ID", "rt", "rtmin", "rtmax", "Width", "int", "area", 
        "sn", "sn_2", "sn_3", "sn_5", "points", "signal_points", "class", 
        "conf", "baseline", "min_rt", "max_rt", "lr_diff"
    ]
    
    results_dataframe = pd.DataFrame(result_list, columns=columns_list).drop_duplicates()
    
    ion_information = (
        pd.read_excel(f"{indir}/ALL_ions.xlsx")[["mw ID", "RT (min)"]]
        .rename(columns={"RT (min)": "rt_theoretic"})
    )
    merged_df = pd.merge(results_dataframe, ion_information, on="mw ID")

    merged_df.to_csv(f"{output_dir}/result.csv")


def main():
    args = parse_arguments()
    indir = os.path.abspath(args.indir)
    num_threads = args.threads
    output_dir = f"{indir}/temp"
    os.makedirs(output_dir, exist_ok=True)

    sample_info = pd.read_csv('{}/sample_info.csv'.format(indir))

    if 'picture_P' in sample_info.columns:
        sample_info_picture_P = sample_info[sample_info['picture_P'].notna()]
        P_overlap_picture = sample_info_picture_P["sample_name"].tolist()
        P_overlap_picture = [
            item if item.endswith('_P') else item + '_P' 
            for item in P_overlap_picture
        ]
        del sample_info['picture_P']
    else:
        P_overlap_picture = []

    if 'picture_N' in sample_info.columns:
        sample_info_picture_N = sample_info[sample_info['picture_N'].notna()]
        N_overlap_picture = sample_info_picture_N["sample_name"].tolist()
        N_overlap_picture = [
            item if item.endswith(('_N', '_HN')) else item + '_N' 
            for item in N_overlap_picture
        ]
        del sample_info['picture_N']
    else:
        N_overlap_picture = []

    sample_info = sample_info.dropna()

    all_ions = pd.read_excel('{}/ALL_ions.xlsx'.format(indir))
    all_ions_mz_dict = all_ions.groupby('mw ID').apply(
        lambda group: group[['Q1 (Da)', 'Q3 (Da)', 'RT (min)']].values.tolist()
    ).to_dict()

    all_ions.rename(columns={'Ion mode': 'ion_mode', 'mw ID': 'mwID'}, inplace=True)
    all_ions_dict = pd.Series(all_ions.ion_mode.values, index=all_ions.mwID).to_dict()

    # List mzML or CDF files
    chrom_files = [
        f for f in os.listdir(os.path.join(indir, "mzML")) 
        if re.search(r'\.(mz[X]?ML|cdf)$', f, re.IGNORECASE)
    ]

    # Check for 'order' columns in sample_info and initialize if missing
    if not any(col in sample_info.columns for col in ["N_order", "P_order", "order"]):
        sample_info['N_order'] = 1
        sample_info['P_order'] = 1
        sample_info['order'] = 1

    # Process sample information
    if round(len(chrom_files) / len(sample_info), 2) >= 2:
        sample_info_temp_N = (
            sample_info[['sample_name', 'N_order']]
            .assign(sampleID=lambda df: df['sample_name'] + "_N")
            .rename(columns={'N_order': 'order'})[['sampleID', 'order']]
        )
        sample_info_temp_N = sample_info_temp_N.dropna()
        
        sample_info_temp_P = (
            sample_info[['sample_name', 'P_order']]
            .assign(sampleID=lambda df: df['sample_name'] + "_P")
            .rename(columns={'P_order': 'order'})[['sampleID', 'order']]
        )
        sample_info_temp_P = sample_info_temp_P.dropna()
        
        sample_info_temp = pd.concat([sample_info_temp_P, sample_info_temp_N])
    else:
        sample_info_temp = (
            sample_info
            .assign(sampleID=lambda df: df['sample_name'])[['sampleID', 'order']]
        )

    # Set missing 'order' values to 1
    sample_info_temp['order'] = sample_info_temp['order'].fillna(1)

    # Create a DataFrame for chromatogram files
    cf = (
        pd.DataFrame({'chrom_files': chrom_files})
        .assign(id=range(1, len(chrom_files) + 1))
        .assign(idx=lambda df: df['chrom_files'].str.extract(r"\((\d+)\)")[0])
        .assign(idx=lambda df: pd.to_numeric(df['idx'], errors='coerce').fillna(1))
        .assign(cfs=lambda df: df['chrom_files'].str.replace(r" \(.+?\)", "", regex=True))
        .assign(sampleID=lambda df: df['cfs'].apply(lambda x: re.split(r"-", x)[-1]))
        .assign(sampleID=lambda df: df['sampleID'].str.replace(r" \(.+?\)|.mzML", "", regex=True))
    )

    # Merge with sample information and filter based on 'order'
    cf = cf.merge(sample_info_temp, how='left', on='sampleID').query('idx == order')

    # Save the filtered chromatogram files information
    cf.to_csv(os.path.join(indir, "temp/raw_chrom_file.csv"), index=False)

    # Check if all samples in sample_info are accounted for
    if len(cf) != len(sample_info_temp):
        missed = sample_info_temp['sampleID'][
            ~sample_info_temp['sampleID'].isin(cf['sampleID'])
        ].tolist()
        raise ValueError(f"Please check sample_info order of: {', '.join(missed)}")

    # Extract and verify final sample names
    chrom_name = cf['cfs'].apply(lambda x: re.split(r"-|\.mzML", x)[-2])
    chrom_name = chrom_name.apply(lambda x: re.split(r"_P$|_N$", x)[0])

    fin_sample_name = chrom_name.isin(
        sample_info['sample_name'].apply(lambda x: re.split(r"_P$|_N$", x)[0])
    )

    chrom_files = cf['chrom_files'][fin_sample_name].tolist()

    # The data is read and interpolated and normalized
    out_sample, all_tic_information = read_mzml_data(indir, chrom_files)
    mzml_ID = set([sublist[1] for sublist in out_sample])
    all_ions_ID = set(all_ions_dict.keys())

    # Calculate the difference between mzml_ID and all_ions_ID
    difference = mzml_ID - all_ions_ID

    if len(difference) > 0:
        with open(os.path.join(indir, "error.txt"), "w") as error_file:
            error_file.write("Difference between mzml_ID and all_ions_ID:\n")
            for item in difference:
                error_file.write(f"{item}\n")
            error_file.write("\nError: all_ions and mzML are not equal. Stopping the program.\n")
        raise ValueError("all_ions and mzML are not equal. Stopping the program.")

    count_P = len([
        key for key, value in all_ions_dict.items() 
        if (value == "Positive" or value == "positive")
    ])
    count_N = len([
        key for key, value in all_ions_dict.items() 
        if (value == "Negative" or value == "negative")
    ])

    if not os.path.exists(f'{indir}/TIC'):
        os.makedirs(f'{indir}/TIC')

    all_tic_dataset = []
    project_type = 0
    for sample in all_tic_information:
        if "_P" in sample[0] and not sample[0].endswith('_P'):
            sample[0] = sample[0].split("_P", 1)[0] + "_P"
        if ("_N" in sample[0] or "_HN" in sample[0]) and not sample[0].endswith('_N') and not sample[0].endswith('_HN'):
            if "_HN" in sample[0]:
                sample[0] = sample[0].split("_HN", 1)[0] + "_HN"
            if "_N" in sample[0]:
                sample[0] = sample[0].split("_N", 1)[0] + "_N"
                
        if ("_P" not in sample[0]) and ("_N" not in sample[0]) and ("_HN" not in sample[0]):
            tmp_sample = sample[0] + "_N"
            all_tic_dataset.append([tmp_sample, sample[1], sample[2]])
            tmp_sample = sample[0] + "_P"
            all_tic_dataset.append([tmp_sample, sample[1], sample[2]])
            project_type = 1
        else:
            all_tic_dataset.append([sample[0], sample[1], sample[2]])

    # QC_MS_tic_overlap-N
    plt.figure(figsize=(12, 6), dpi=300)
    num_N = 0
    plot_colors = ['green', 'red', 'mediumblue']
    if len(N_overlap_picture) > 0:
        for tic_sample in all_tic_dataset:
            if tic_sample[0] in N_overlap_picture and ("_N" in tic_sample[0] or "_HN" in tic_sample[0]) and num_N <= 2:
                if tic_sample[0] == N_overlap_picture[0]:
                    select_chrom_files = ""
                    for tmp_sample in chrom_files:
                        if tic_sample[0] in tmp_sample:
                            select_chrom_files = tmp_sample
                    if select_chrom_files == "":
                        select_chrom_files = chrom_files[0]
                    
                    legend_label = (
                        f'TIC of -MRM({count_N} pairs): from {tic_sample[0]} of '
                        f'{select_chrom_files.replace(select_chrom_files.split("-")[-1], "")[:-1]}'
                    )
                    legend_square = Line2D(
                        [0], [0], color='mediumblue', marker='s', markersize=8, label=legend_label
                    )
                    plt.legend(handles=[legend_square], loc='upper left', fontsize=8, frameon=False)

                if project_type == 0:
                    plt.plot(
                        tic_sample[1], [x - 50 for x in tic_sample[2]], 
                        linewidth=0.8, color=plot_colors[num_N]
                    )
                else:
                    odd_index_values_rt = [tic_sample[1][i] for i in range(len(tic_sample[1])) if i % 2 != 0]
                    odd_index_values_int = [
                        [x - 50 for x in tic_sample[2]][i] 
                        for i in range(len([x - 50 for x in tic_sample[2]])) if i % 2 != 0
                    ]
                    plt.plot(odd_index_values_rt, odd_index_values_int, linewidth=0.8, color=plot_colors[num_N])

                plt.xlabel('Time, min', fontsize=14)
                plt.ylabel('Intensity, cps', fontsize=14)
                ax = plt.gca()
                ax.xaxis.set_major_locator(MultipleLocator(1))
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                num_N += 1
        plt.savefig(f'{indir}/TIC/{os.path.basename(os.path.normpath(indir))}_QC_MS_tic_overlap-N.pdf', format='pdf')
        plt.close()

    # QC_MS_tic_overlap-P
    if len(P_overlap_picture) > 0:
        plt.figure(figsize=(12, 6), dpi=300)
        num_P = 0
        for tic_sample in all_tic_dataset:
            if tic_sample[0] in P_overlap_picture and "_P" in tic_sample[0]:
                if tic_sample[0] == P_overlap_picture[0]:
                    select_chrom_files = ""
                    for tmp_sample in chrom_files:
                        if tic_sample[0] in tmp_sample:
                            select_chrom_files = tmp_sample
                    if select_chrom_files == "":
                        select_chrom_files = chrom_files[0]
                        
                    legend_label = (
                        f'TIC of +MRM({count_P} pairs): from {tic_sample[0]} of '
                        f'{select_chrom_files.replace(select_chrom_files.split("-")[-1], "")[:-1]}'
                    )
                    legend_square = Line2D(
                        [0], [0], color='mediumblue', marker='s', markersize=8, label=legend_label
                    )
                    plt.legend(handles=[legend_square], loc='upper left', fontsize=8, frameon=False)

                if project_type == 0:
                    plt.plot(
                        tic_sample[1], [x - 50 for x in tic_sample[2]], 
                        linewidth=0.8, color=plot_colors[num_P]
                    )
                else:
                    even_index_values_rt = [tic_sample[1][i] for i in range(len(tic_sample[1])) if i % 2 == 0]
                    even_index_values_int = [
                        [x - 50 for x in tic_sample[2]][i] 
                        for i in range(len([x - 50 for x in tic_sample[2]])) if i % 2 == 0
                    ]
                    plt.plot(even_index_values_rt, even_index_values_int, linewidth=0.8, color=plot_colors[num_P])

                plt.xlabel('Time, min', fontsize=14)
                plt.ylabel('Intensity, cps', fontsize=14)
                ax = plt.gca()
                ax.xaxis.set_major_locator(MultipleLocator(1))
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                num_P += 1
        plt.savefig(f'{indir}/TIC/{os.path.basename(os.path.normpath(indir))}_QC_MS_tic_overlap-P.pdf', format='pdf')
        plt.close()

    # QC_MS_TIC-P
    if len(P_overlap_picture) > 0:
        plt.figure(figsize=(12, 6), dpi=300)
        for tic_sample in all_tic_dataset:
            if tic_sample[0] == P_overlap_picture[0] and "_P" in tic_sample[0]:
                select_chrom_files = ""
                for tmp_sample in chrom_files:
                    if tic_sample[0] in tmp_sample:
                        select_chrom_files = tmp_sample
                if select_chrom_files == "":
                    select_chrom_files = chrom_files[0]
                    
                legend_label = (
                    f'TIC of +MRM({count_P} pairs): from {tic_sample[0]} of '
                    f'{select_chrom_files.replace(select_chrom_files.split("-")[-1], "")[:-1]}'
                )
                legend_square = Line2D(
                    [0], [0], color='mediumblue', marker='s', markersize=8, label=legend_label
                )
                
                if project_type == 0:
                    plt.plot(
                        tic_sample[1], [x - 50 for x in tic_sample[2]], 
                        linewidth=0.8, color="mediumblue"
                    )
                else:
                    even_index_values_rt = [tic_sample[1][i] for i in range(len(tic_sample[1])) if i % 2 == 0]
                    even_index_values_int = [
                        [x - 50 for x in tic_sample[2]][i] 
                        for i in range(len([x - 50 for x in tic_sample[2]])) if i % 2 == 0
                    ]
                    plt.plot(even_index_values_rt, even_index_values_int, linewidth=0.8, color="mediumblue")
                
                plt.xlabel('Time, min', fontsize=14)
                plt.ylabel('Intensity, cps', fontsize=14)
                ax = plt.gca()
                ax.xaxis.set_major_locator(MultipleLocator(1))
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.legend(handles=[legend_square], loc='upper left', fontsize=8, frameon=False)
                P_max_value = max(tic_sample[1])
                P_min_value = min(tic_sample[1])
        plt.savefig(f'{indir}/TIC/{os.path.basename(os.path.normpath(indir))}_QC_MS_TIC-P.pdf', format='pdf')
        plt.close()

    # QC_MS_TIC-N
    if len(N_overlap_picture) > 0:
        plt.figure(figsize=(12, 6), dpi=300)
        for tic_sample in all_tic_dataset:
            if tic_sample[0] == N_overlap_picture[0] and ("_N" in tic_sample[0] or "_HN" in tic_sample[0]):
                select_chrom_files = ""
                for tmp_sample in chrom_files:
                    if tic_sample[0] in tmp_sample:
                        select_chrom_files = tmp_sample
                if select_chrom_files == "":
                    select_chrom_files = chrom_files[0]
                    
                legend_label = (
                    f'TIC of -MRM({count_N} pairs): from {tic_sample[0]} of '
                    f'{select_chrom_files.replace(select_chrom_files.split("-")[-1], "")[:-1]}'
                )
                legend_square = Line2D(
                    [0], [0], color='mediumblue', marker='s', markersize=8, label=legend_label
                )
                
                if project_type == 0:
                    plt.plot(
                        tic_sample[1], [x - 50 for x in tic_sample[2]], 
                        linewidth=0.8, color="mediumblue"
                    )
                else:
                    odd_index_values_rt = [tic_sample[1][i] for i in range(len(tic_sample[1])) if i % 2 != 0]
                    odd_index_values_int = [
                        [x - 50 for x in tic_sample[2]][i] 
                        for i in range(len([x - 50 for x in tic_sample[2]])) if i % 2 != 0
                    ]
                    plt.plot(odd_index_values_rt, odd_index_values_int, linewidth=0.8, color="mediumblue")

                plt.xlabel('Time, min', fontsize=14)
                plt.ylabel('Intensity, cps', fontsize=14)
                ax = plt.gca()
                ax.xaxis.set_major_locator(MultipleLocator(1))
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.legend(handles=[legend_square], loc='upper left', fontsize=8, frameon=False)
                N_max_value = max(tic_sample[1])
                N_min_value = min(tic_sample[1])
        plt.savefig(f'{indir}/TIC/{os.path.basename(os.path.normpath(indir))}_QC_MS_TIC-N.pdf', format='pdf')
        plt.close()

    dataset = []
    for sample in out_sample:
        sampleID = list([sample[0], sample[1]])
        if "_P" in sampleID[0] and not sampleID[0].endswith('_P'):
            sampleID[0] = sampleID[0].split("_P", 1)[0] + "_P"
        if ("_N" in sampleID[0] or "_HN" in sampleID[0]) and not sampleID[0].endswith('_N') and not sampleID[0].endswith('_HN'):
            if "_HN" in sampleID[0]:
                sampleID[0] = sampleID[0].split("_HN", 1)[0] + "_HN"
            if "_N" in sampleID[0]:
                sampleID[0] = sampleID[0].split("_N", 1)[0] + "_N"
                
        if ("_P" not in sampleID[0]) and ("_N" not in sampleID[0]) and ("_HN" not in sampleID[0]):
            if all_ions_dict[sampleID[1]].lower() == "negative":
                sampleID[0] = sampleID[0] + "_N"
            if all_ions_dict[sampleID[1]].lower() == "positive":
                sampleID[0] = sampleID[0] + "_P"

        dataset.append([sampleID, sample[6], sample[7]])

    # MRM_detection_of_multimodal_maps-N
    if len(N_overlap_picture) > 0:
        plt.figure(figsize=(12, 6), dpi=300)
        num_N = 0
        for tic_sample in dataset:
            if tic_sample[0][0] == N_overlap_picture[0] and ("_N" in tic_sample[0][0] or "_HN" in tic_sample[0][0]):
                if len(tic_sample[1]) > 2:
                    x_diff = np.diff(tic_sample[1]) 
                    min_diff = x_diff.min()
                    x_full_1 = np.arange(N_min_value, min(list(tic_sample[1])), min_diff)
                    y_full_1 = np.full_like(x_full_1, 50, dtype=float) 
                    x_full_2 = np.arange(max(list(tic_sample[1])), N_max_value + min_diff, min_diff)
                    y_full_2 = np.full_like(x_full_2, 50, dtype=float) 
                    x_value = list(x_full_1) + list(tic_sample[1]) + list(x_full_2)
                    y_value = list(y_full_1) + list(tic_sample[2]) + list(y_full_2)
                    
                    if num_N == 0:
                        legend_label = (
                            f'XIC of -MRM({count_N} pairs): {all_ions_mz_dict[tic_sample[0][1]][0][0]}/'
                            f'{all_ions_mz_dict[tic_sample[0][1]][0][1]} amu Expected RT :'
                            f'{all_ions_mz_dict[tic_sample[0][1]][0][2]} ID: {tic_sample[0][1]} from {tic_sample[0][0]} of ...'
                        )
                        legend_square = Line2D(
                            [0], [0], color='mediumblue', marker='s', markersize=8, label=legend_label
                        )
                    plt.plot(x_value, [x - 50 for x in y_value], linewidth=0.8)
                    plt.xlabel('Time, min', fontsize=14)
                    plt.ylabel('Intensity, cps', fontsize=14)
                    plt.xticks(fontsize=12)
                    plt.yticks(fontsize=12)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(MultipleLocator(1))
                    plt.legend(handles=[legend_square], loc='upper left', fontsize=8, frameon=False)
                    num_N += 1
        plt.savefig(f'{indir}/TIC/{os.path.basename(os.path.normpath(indir))}_MRM_detection_of_multimodal_maps-N.pdf', format='pdf')
        plt.close()

    # MRM_detection_of_multimodal_maps-P
    if len(P_overlap_picture) > 0:
        plt.figure(figsize=(12, 6), dpi=300)
        num_P = 0
        for tic_sample in dataset:
            if tic_sample[0][0] == P_overlap_picture[0] and "_P" in tic_sample[0][0]:
                if len(tic_sample[1]) > 2:
                    x_diff = np.diff(tic_sample[1]) 
                    min_diff = x_diff.min()
                    x_full_1 = np.arange(P_min_value, min(list(tic_sample[1])), min_diff)
                    y_full_1 = np.full_like(x_full_1, 50, dtype=float) 
                    x_full_2 = np.arange(max(list(tic_sample[1])), P_max_value + min_diff, min_diff)
                    y_full_2 = np.full_like(x_full_2, 50, dtype=float) 
                    x_value = list(x_full_1) + list(tic_sample[1]) + list(x_full_2)
                    y_value = list(y_full_1) + list(tic_sample[2]) + list(y_full_2)
                    
                    if num_P == 0:
                        legend_label = (
                            f'XIC of +MRM({count_P} pairs): {all_ions_mz_dict[tic_sample[0][1]][0][0]}/'
                            f'{all_ions_mz_dict[tic_sample[0][1]][0][1]} amu Expected RT :'
                            f'{all_ions_mz_dict[tic_sample[0][1]][0][2]} ID: {tic_sample[0][1]} from {tic_sample[0][0]} of ...'
                        )
                        legend_square = Line2D(
                            [0], [0], color='mediumblue', marker='s', markersize=8, label=legend_label
                        )
                    plt.plot(x_value, [x - 50 for x in y_value], linewidth=0.8)
                    plt.xlabel('Time, min', fontsize=14)
                    plt.ylabel('Intensity, cps', fontsize=14)
                    plt.xticks(fontsize=12)
                    plt.yticks(fontsize=12)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(MultipleLocator(1))
                    plt.legend(handles=[legend_square], loc='upper left', fontsize=8, frameon=False)
                    num_P += 1
        plt.savefig(f'{indir}/TIC/{os.path.basename(os.path.normpath(indir))}_MRM_detection_of_multimodal_maps-P.pdf', format='pdf')
        plt.close()

    with open(os.path.join(indir, 'temp/dataset_all_samples.pkl'), 'wb') as f:
        pickle.dump(dataset, f)

    run(out_sample, indir, num_threads)


if __name__ == '__main__':
    main()