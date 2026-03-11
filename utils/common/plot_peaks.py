# -*- coding: utf-8 -*-
"""
Created on Mon Sep  11 09:38:56 2023

@author: liutong
"""

import os
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator


def find_nearest_index(lst, target: float) -> int:
    """
    Find the index of the element in the list that is closest to the target value.
    Utilizes Numpy vectorization for significant performance improvement over standard loops.
    
    Args:
        lst (list or array-like): The list of numerical values.
        target (float): The target value to find the closest match for.
        
    Returns:
        int: The index of the closest element.
    """
    arr = np.array(lst)
    return (np.abs(arr - target)).argmin()


class PeakPlotter:
    """
    A class to plot peak data for different sample groups.
    """
    def __init__(self, groups: list, outdir: str, peak_information: dict, all_ions: dict, concern_ids=None):
        """
        Initialize PeakPlotter with groups, output directory, and peak information.

        Args:
            groups (list): List of DataFrames containing peak data for different sample groups.
            outdir (str): Base output directory.
            peak_information (dict): Dictionary containing reference peak information.
            all_ions (dict): Dictionary mapping ion names to their theoretical retention times.
            concern_ids (list, optional): List of IDs to mark or concern. Defaults to None.
        """
        self.groups = groups
        self.outdir = outdir
        self.peak_information = peak_information
        self.all_ions = all_ions
        self.concern_ids = concern_ids or []

    def plot_peak(self):
        """
        Iterate through each sample group, generate subplots, and save the figures.
        """
        for group in self.groups:
            first_group_list = self.sort_and_limit_group(group)
            for num, first_group in enumerate(first_group_list):
                fig, axs = plt.subplots(nrows=4, ncols=6, figsize=(15, 7))
                
                # Retrieve the theoretical retention time (rt) for the current group
                quan_name = first_group.iloc[-1].quan
                rt_theory = self.all_ions[quan_name]
                
                # Plot data, adjust layout to remove empty subplots, and save
                axs = self.plot_group_data(axs, first_group, rt_theory)
                self.adjust_subplot_layout(fig, axs, first_group.shape[0])
                self.save_and_close_figure(fig, quan_name, num)

    def sort_and_limit_group(self, group: pd.DataFrame) -> list:
        """
        Sort the group by SampleID, prioritize 'mix' samples, and chunk into blocks of 24.

        Args:
            group (pd.DataFrame): The DataFrame containing peak data for a sample group.

        Returns:
            list: A list of DataFrame chunks, each containing up to 24 rows.
        """
        first_group = group.sort_values(['SampleID'])
        
        # Prioritize samples containing 'mix' in their SampleID
        mask = first_group["SampleID"].str.contains('mix', case=False, na=False)
        first_group = pd.concat([first_group[mask], first_group[~mask]], ignore_index=True)
        
        # Chunk the DataFrame into blocks of maximum 24 rows
        return [first_group.iloc[i:i + 24] for i in range(0, len(first_group), 24)]

    def plot_group_data(self, axs, first_group: pd.DataFrame, rt_theory: float):
        """
        Plot data for each sample inside the chunk on individual subplots.

        Args:
            axs (numpy.ndarray): Matplotlib axes array.
            first_group (pd.DataFrame): The data chunk to plot.
            rt_theory (float): The theoretical retention time reference.

        Returns:
            numpy.ndarray: The modified axes array.
        """
        first_group = first_group.reset_index(drop=True)
        for i, row in first_group.iterrows():
            # Map 1D index to 2D axes grid
            ax = axs[i // 6, i % 6]
            self.plot_subplot(ax, row, rt_theory)
        return axs

    def plot_subplot(self, ax, row: pd.Series, rt_theory: float):
        """
        Plot the retention time versus intensity for a single sample on a given axis.

        Args:
            ax (matplotlib.axes.Axes): The axis to plot on.
            row (pd.Series): The data row containing retention time and intensity arrays.
            rt_theory (float): Theoretical retention time.
        """
        try:
            rt_arr = np.array(row.rt)
            int_arr = np.array(row.int)
            
            # Draw the main intensity curve
            ax.plot(rt_arr, int_arr, color="black")
            
            # Configure basic plot properties
            ax.set_title(f'Sample {row.SampleID}\nrt theory: {round(rt_theory, 2)}', fontsize=10)
            ax.set_xlabel('Rt(mins)', fontsize=6.5)
            ax.set_ylabel('Intensity', fontsize=6.5)
            ax.tick_params(axis='both', labelsize=6.5)
            ax.xaxis.set_major_locator(MultipleLocator(base=0.2))

            # Check if peak area needs to be highlighted based on peak_information
            peak_key = f"{row.SampleID}{row.quan}"
            if peak_key in self.peak_information:
                peak = self.peak_information[peak_key]
                
                rtmin = find_nearest_index(rt_arr, peak[1])
                rtmax = find_nearest_index(rt_arr, peak[2])
                rt_n = round(peak[0], 2)
                
                ax.set_title(f'Sample {row.SampleID}\nrt theory: {round(rt_theory, 2)}; rt: {rt_n}', fontsize=10)
                
                # Extract the peak region and fill the area with red
                fill_int = int_arr[rtmin:rtmax + 1]
                fill_rt = rt_arr[rtmin:rtmax + 1]
                
                if len(fill_int) > 0:
                    min_val = np.min(fill_int)
                    ax.fill_between(fill_rt, min_val, fill_int, color='red', alpha=0.3)
                
                # Add vertical reference lines for theoretical (red) and actual (green) rt
                ax.axvline(x=round(rt_theory, 2), color='r', linestyle='--')
                ax.axvline(x=rt_n, color='g', linestyle='--')

        except Exception as e:
            print(f"Process error plot: {e}")

    def adjust_subplot_layout(self, fig, axs, group_size: int):
        """
        Clean up the layout by removing any unused subplot axes.

        Args:
            fig (matplotlib.figure.Figure): The main figure object.
            axs (numpy.ndarray): Array of axes.
            group_size (int): The actual number of plotted samples.
        """
        # Flatten the axes array and precisely remove unused subplots
        axes_flat = axs.flatten()
        for j in range(group_size, len(axes_flat)):
            fig.delaxes(axes_flat[j])
        
        plt.tight_layout()

    def save_and_close_figure(self, fig, quan: str, num: int):
        """
        Save the figure to the centralized 'picture' directory and release memory.

        Args:
            fig (matplotlib.figure.Figure): The figure to save.
            quan (str): The identifier for the current peak group.
            num (int): The chunk index for this group.
        """
        pic_dir = os.path.join(self.outdir, "picture")
        file_path = os.path.join(pic_dir, f"{quan}_{num}.png")
        
        plt.savefig(file_path)
        fig.clf()
        plt.close(fig)


def plot_peaks_multithread(threads: int, peak_matrix: pd.DataFrame, dataset_dataframe: pd.DataFrame, 
                           indir: str, peak_information: dict, all_ions: dict, 
                           all_ions_mode: pd.DataFrame = None):
    """
    Generate peak plots in parallel using multiprocessing.
    Consolidates data rendering without separating missing/found peaks.

    Args:
        threads (int): Number of worker processes to spawn.
        peak_matrix (pd.DataFrame): Processed peak matrix.
        dataset_dataframe (pd.DataFrame): Raw dataset with rt and intensity values.
        indir (str): Root input/output directory.
        peak_information (dict): Dictionary mapping sample IDs to peak data.
        all_ions (dict): Theoretical retention times for ions.
        all_ions_mode (pd.DataFrame, optional): DataFrame containing ion mode information. 
                                                If provided, data will be filtered by polarity.
    """
    concern_ID = []

    if all_ions_mode is not None and not all_ions_mode.empty:
        dataset_dataframe = pd.merge(dataset_dataframe, all_ions_mode, on='quan')
        dataset_dataframe = dataset_dataframe[dataset_dataframe['SampleID'].str[-1] == dataset_dataframe['ion_mode']]
    
    # Create the unified output directory
    pic_dir = os.path.join(indir, "picture")
    os.makedirs(pic_dir, exist_ok=True)

    if dataset_dataframe.empty:
        return

    # Group the entire dataset by 'quan'
    grouped_df = dataset_dataframe.groupby('quan')
    sub_dfs = [group for _, group in grouped_df]

    if not sub_dfs:
        return

    # Split the groups into equitable chunks for the thread pool
    chunk_size = max(1, int(np.ceil(len(sub_dfs) / threads)))
    chunks = [sub_dfs[i:i + chunk_size] for i in range(0, len(sub_dfs), chunk_size)]
    
    # Execute plot generation using a Process Pool
    with multiprocessing.Pool(processes=threads) as pool:
        for chunk in chunks:
            plotter = PeakPlotter(chunk, indir, peak_information, all_ions, concern_ID)
            pool.apply_async(plotter.plot_peak)
        
        pool.close()
        pool.join()