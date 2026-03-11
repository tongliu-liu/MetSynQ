# -*- coding: utf-8 -*-
"""
@author: liutong
Modified for GitHub release: Modularized Untargeted pipeline, dynamic paths, 
and optimized for CPU efficiency & memory management (logic unchanged).
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import glob
import pickle
import argparse
import traceback
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch as _torch
from tqdm import tqdm
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from ultralytics import YOLO

from utils.common.data_processing import (
    expand_and_adjust_peak,
    calculate_peak_area,
    calculate_sn
)

from utils.untargeted.extract_roi import extract_roi


# ----------------- Utility Functions -----------------
def find_closest_index(lst, target):
    """Return the index of the element in lst closest to target."""
    arr = np.asarray(lst)
    return int(np.abs(arr - target).argmin())


def split_into_batches(data, batch_size):
    """Split data list into batches of given size."""
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]


def signal_to_image(signal, rt):
    """Convert 1D signal to 224x224 RGB image using fast Object-Oriented API."""
    # OO API avoids pyplot global state locks, making it much faster and thread-safe
    fig = Figure(figsize=(2.24, 2.24), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])  # Equivalent to adjusting margins to 0/1
    ax.plot(rt, signal, color='black')
    ax.axis('off')

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    
    # Extract RGB channels directly
    img_array = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape((224, 224, 4))
    return img_array[:, :, :3]


def _generate_image_wrapper(sample):
    """Wrapper function to unpack sample tuples for map execution."""
    # sample[1] is rt, sample[2] is signal
    return signal_to_image(sample[2], sample[1])


def process_peak_signal(signal, rt, x_start, x_end, cls, conf):
    """Process peak signal with adjustments and boundary refinement."""
    rtmin_index = find_closest_index(rt, rt[0] + (rt[-1] - rt[0]) * x_start)
    rtmax_index = find_closest_index(rt, rt[0] + (rt[-1] - rt[0]) * x_end)
    
    peak_signal = signal[rtmin_index: rtmax_index + 1]
    peak_signal_rt = rt[rtmin_index: rtmax_index + 1]
    
    peak_signal, peak_signal_rt = expand_and_adjust_peak(
        signal, rt, peak_signal, peak_signal_rt, cls, conf.item()
    )
    return peak_signal, peak_signal_rt


def calculate_peak_properties(peak_signal, peak_signal_rt):
    """Calculate peak properties: RT min/max, area, peak RT, width, and height."""
    rtmin, rtmax = peak_signal_rt[0], peak_signal_rt[-1]
    peak_area = calculate_peak_area(peak_signal, peak_signal_rt)
    peak_rt = peak_signal_rt[np.argmax(peak_signal)]
    peak_weight = rtmax - rtmin
    peak_height = max(peak_signal) - min(peak_signal)
    
    return rtmin, rtmax, peak_area, peak_rt, peak_weight, peak_height


def filter_valid_peaks(sn, sn2, sn_5):
    """Determine whether a peak is valid based on SN thresholds."""
    if sn >= 5 and (sn >= 6 or sn_5 >= 1.5) and sn2 >= 2:
        return True
    return False


# ----------------- Core Processing -----------------
def process_sample(samples, class_dict, model, num_threads=8, chunk_size=200, yolo_batch_size=512):
    """
    Run YOLO single-thread inference + multi-process image generation.
    Returns a list of processed peak records.
    """
    result_records = []
    meta_info = [(s[0][0], s[0][1], s[1], s[2]) for s in samples]

    def parallel_image_generation(data_samples):
        # Create pool ONCE. Use map to preserve the strict order of images relative to meta_info.
        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            # chunksize optimization prevents IPC bottleneck
            calc_chunk = max(1, chunk_size // num_threads)
            results = list(tqdm(
                executor.map(_generate_image_wrapper, data_samples, chunksize=calc_chunk),
                total=len(data_samples),
                desc="Converting signals to images"
            ))
        return results

    # 1. Multi-process image generation (Order is perfectly preserved)
    data_list = parallel_image_generation(samples)

    # 2. YOLO single-thread inference (GPU if available)
    device = "cuda:0" if _torch.cuda.is_available() else "cpu"
    print(f"\nRunning YOLO prediction on device: {device} ...")
    
    for i in tqdm(range(0, len(data_list), yolo_batch_size), desc="YOLO batched predicting"):
        batch_imgs = data_list[i:i + yolo_batch_size]
        batch_meta = meta_info[i:i + yolo_batch_size]

        # YOLO prediction
        batch_results = list(model.predict(batch_imgs, device=device, stream=True, verbose=False))

        # Parse results
        for (sampleID, mwID, rt, signal), result in zip(batch_meta, batch_results):
            boxes = result.boxes
            xywh = boxes.xywh.cpu().numpy()
            cls_all = boxes.cls.cpu().numpy()
            conf_all = boxes.conf.cpu().numpy()
            
            for pix, cls, conf in zip(xywh, cls_all, conf_all):
                if cls == 3:
                    continue

                x_start = max((pix[0] - pix[2] / 2 - 10.18181818) / (213.81818182 - 10.18181818), 0)
                x_end = min((pix[0] + pix[2] / 2 - 10.18181818) / (213.81818182 - 10.18181818), 1)
                
                peak_signal, peak_signal_rt = process_peak_signal(signal, rt, x_start, x_end, cls, conf)
                rtmin, rtmax, area, rt_peak, width, height = calculate_peak_properties(peak_signal, peak_signal_rt)
                
                sn, baseline, sn2, sn3, sn4, sn5 = calculate_sn(signal, rt, peak_signal, peak_signal_rt)

                if filter_valid_peaks(sn, sn2, sn5):
                    result_records.append([
                        sampleID, mwID, rt_peak, rtmin, rtmax, width, height, area, sn, sn2,
                        sn3[0], sn5, len(peak_signal), len(signal),
                        class_dict[int(cls)], float(conf), baseline,
                        min(rt), max(rt), abs(peak_signal[-1] - peak_signal[0]) / (height + 1e-8), 
                        max(signal)
                    ])

    return result_records


# ----------------- Main Execution -----------------
def run(dataset, indir, num_threads):
    output_dir = os.path.join(indir, "temp")
    os.makedirs(output_dir, exist_ok=True)

    class_dict = {0: "A", 1: "B", 2: "C", 3: "D"}
    
    model_path = os.path.abspath(os.path.join(project_root, "weights", "best.pt"))
    if not os.path.exists(model_path):
        fallback_path = os.path.abspath(os.path.join(project_root, "weights", "best.pt"))
        if os.path.exists(fallback_path):
            model_path = fallback_path
            
    model = YOLO(model_path)

    result_list = process_sample(dataset, class_dict, model, num_threads=num_threads)

    # Save results
    df = pd.DataFrame(result_list, columns=[
        "sampleID", "mw ID", "rt", "rtmin", "rtmax", "Width", "int", "area",
        "sn", "sn_2", "sn_3", "sn_5", "points", "signal_points", "class",
        "conf", "baseline", "min_rt", "max_rt", "lr_diff", 'max_signal'
    ]).drop_duplicates()

    # Remove duplicate peaks of class A
    filtered_df_A = df[df['class'] == 'A']
    filtered_df_no_A = df[df['class'] != 'A']
    dedup_A = filtered_df_A.sort_values(by='Width').drop_duplicates(subset='rt', keep='first')
    df = pd.concat([dedup_A, filtered_df_no_A], ignore_index=True)

    ion_info = pd.read_excel(os.path.join(indir, "ALL_ions.xlsx"))[["mw ID", "RT (min)"]].rename(columns={"RT (min)": "rt_theoretic"})
    merged_df = pd.merge(df, ion_info, on="mw ID", how='left')

    out_path = os.path.join(output_dir, "result.csv")
    merged_df.to_csv(out_path, index=False)
    print(f"\n✅ Results saved to {out_path}")


def parse_arguments():
    """解析命令行参数"""
    description = 'Converts the training set into a data form acceptable to the training program.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--indir', dest='indir', required=True, type=str,
                        help='Input folder, project folder. [must be specified]')
    parser.add_argument('--threads', default=16, type=int, metavar='N', help='Number of data loading workers (default: 16)')
    parser.add_argument('--ppm', default=15, type=int, metavar='N', help='ppm value (default: 15)')
    return parser.parse_args()


def main():
    """主函数封装，隔离作用域"""
    args = parse_arguments()

    indir_path = os.path.abspath(args.indir)
    output_directory = os.path.join(indir_path, "temp")
    os.makedirs(output_directory, exist_ok=True)

    # ----------------- Load sample_info -----------------
    file_pattern = glob.glob(os.path.join(indir_path, '*peak_list*.csv'))
    if file_pattern:
        sample_info_df = pd.read_csv(file_pattern[0])
    else:
        raise FileNotFoundError("No matching sample_info CSV file found.")

    mzml_indir_path = os.path.join(indir_path, 'mzML')
    out_sample = extract_roi(mzml_indir_path, sample_info_df, args.ppm)

    # ----------------- Build dataset -----------------
    final_dataset = [[[s[0], s[1]], s[2], [x + 50 for x in s[3]]] for s in out_sample]
        
    with open(os.path.join(output_directory, 'dataset_all_samples.pkl'), 'wb') as f:
        pickle.dump(final_dataset, f)
        
    run(final_dataset, indir_path, args.threads)


if __name__ == '__main__':
    main()