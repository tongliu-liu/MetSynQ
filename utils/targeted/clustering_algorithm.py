# -*- coding: utf-8 -*-
"""
Created on Mon Sep  11 09:38:56 2023

@author: liutong
"""

import os
import argparse
import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def get_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Peak mapping')
    parser.add_argument('--indir', type=str, default='indir', help='Data folder.')
    parser.add_argument('--threads', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
    parser.add_argument('--type', type=str, default="rp", help='type')
    return parser.parse_args()


def calculate_unique_counts(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the number of unique sample IDs for each label.
    """
    return raw_data.groupby('labels')['sampleID'].nunique().reset_index(name='sample_count')


def select_optimal_cluster(grouped_clustering_data: pd.DataFrame):
    """
    Select optimal cluster based on weighted combination of sample count (30%) and RT error (70%).
    
    Parameters:
    grouped_clustering_data (pd.DataFrame): DataFrame containing clustering results.
        Must contain columns: 'labels', 'count', 'rt_error'
    
    Returns:
    int/str: Label of the optimal cluster
    """
    unique_counts = calculate_unique_counts(grouped_clustering_data)
    grouped_clustering_data = pd.merge(grouped_clustering_data, unique_counts, on='labels')
    
    required_columns = ['labels', 'sample_count', 'rt_error']
    if not all(col in grouped_clustering_data.columns for col in required_columns):
        raise ValueError(f"DataFrame missing required columns: {required_columns}")
    
    if len(grouped_clustering_data) == 0:
        raise ValueError("Input DataFrame is empty")
    
    # Calculate composite score
    df = grouped_clustering_data.copy()
    
    # Normalize sample count (scale to 0-1 range)
    max_count = df['sample_count'].max()
    df['count_norm'] = df['sample_count'] / max_count if max_count > 0 else 0
    
    # Normalize RT error (inverse scaling - lower error gives higher score)
    max_error = df['rt_error'].max()
    df['rt_error_norm'] = 1 - (df['rt_error'] / max_error) if max_error > 0 else 1
    
    # Calculate weighted composite score
    df['score'] = 0.3 * df['count_norm'] + 0.7 * df['rt_error_norm']

    # Select cluster with the highest score
    best_cluster = df.loc[df['score'].idxmax()]
    
    return best_cluster["labels"]


def Clustering_algorithm(peak_result: pd.DataFrame, all_ions: dict, all_sample: list, sample_type: str) -> pd.DataFrame:
    """
    Perform DBSCAN clustering on retention times (RTs) grouped by 'mw ID'
    and select the mean RT of the cluster with the smallest absolute difference to the
    theoretical RT provided in `all_ions`.
    """
    grouped_mwID = peak_result.groupby('mw ID')
    
    # Use a list to collect results for batch concatenation (significantly faster than iterative concat)
    results_list = []
    
    # Pre-instantiate models outside the loop to avoid redundant overhead
    scaler = StandardScaler()
    dbscan = DBSCAN(eps=0.2, min_samples=1)
    
    for name, group in grouped_mwID:
        # Extract RT list and perform scaling
        rt_list = group["rt"].tolist()
        rt_list_2d = [[x] for x in rt_list]
        X = scaler.fit_transform(rt_list_2d)
        
        # Fit DBSCAN model
        labels = dbscan.fit_predict(X)
        
        # Create a DataFrame for clustering data
        clustering_data = pd.DataFrame({
            'rt': rt_list, 
            'labels': labels,
            'sampleID': group["sampleID"].tolist(),
            'area': group["area"].tolist()
        })
        
        # Group by labels and calculate the mean RT for each cluster
        grouped_clustering_data = clustering_data.groupby('labels')['rt'].agg(['mean']).reset_index()
        grouped_clustering_data.columns = ['labels', 'mean_rt']
        grouped_clustering_data["count"] = clustering_data.groupby('labels').size().values
        grouped_clustering_data = pd.merge(grouped_clustering_data, clustering_data, on="labels")

        # Record the order of label and mean_rt
        grouped_clustering_data_label = grouped_clustering_data.drop_duplicates(subset=["labels"])
        label_list = grouped_clustering_data_label['labels'].tolist() 
        mean_rt_list = grouped_clustering_data_label['mean_rt'].tolist() 

        # Merge categories that are close together (diff <= 0.04)
        close_pairs = []
        for i in range(len(mean_rt_list)):
            for j in range(i + 1, len(mean_rt_list)):
                if abs(mean_rt_list[i] - mean_rt_list[j]) <= 0.04:
                    close_pairs.append((i, j)) 

        if close_pairs:
            for i, j in close_pairs:
                grouped_clustering_data["labels"] = grouped_clustering_data["labels"].replace(label_list[j], label_list[i])
                grouped_clustering_data["mean_rt"] = grouped_clustering_data["mean_rt"].replace(mean_rt_list[j], mean_rt_list[i])
        
        # Calculate rt error with theoretical rt
        if len(grouped_clustering_data) > 0:
            grouped_clustering_data["rt_theory"] = all_ions.get(name, np.nan)
            grouped_clustering_data["rt_error"] = abs(grouped_clustering_data["mean_rt"] - grouped_clustering_data["rt_theory"]) 
            
            # Filter out the categories with the required error, and select the optimal cluster
            if sample_type == "rp":
                grouped_clustering_data = grouped_clustering_data[grouped_clustering_data["rt_error"] <= 0.3]

            if len(grouped_clustering_data) > 0:
                optimal_label = select_optimal_cluster(grouped_clustering_data)
                
                grouped_clustering_data = grouped_clustering_data[grouped_clustering_data["labels"] == optimal_label]
                grouped_clustering_data = grouped_clustering_data.drop_duplicates(subset='sampleID')
                grouped_clustering_data = grouped_clustering_data.drop(columns=["rt"])
            
                # Pivot the data: use 'mean_rt' as index, 'sampleID' as columns, and 'area' as values
                grouped_clustering_data_tra = grouped_clustering_data.pivot_table(
                    index=['mean_rt'], 
                    columns='sampleID', 
                    values='area', 
                    fill_value="NA"
                ).reset_index()
                
                # Format pivoted data
                grouped_clustering_data_tra["mw ID"] = name
                grouped_clustering_data_tra.columns.name = None
                grouped_clustering_data_tra.rename(columns={'mean_rt': 'rt'}, inplace=True)
                
                results_list.append(grouped_clustering_data_tra)

    # Perform a single batch concatenation at the end
    if results_list:
        all_grouped_clustering_data_tra = pd.concat(results_list, ignore_index=True)
        # Reorder columns: 'mw ID', 'rt', then dynamically append the rest
        new_order = ['mw ID', 'rt'] + [col for col in all_grouped_clustering_data_tra.columns if col not in ['mw ID', 'rt']]
        all_grouped_clustering_data_tra = all_grouped_clustering_data_tra[new_order]
    else:
        all_grouped_clustering_data_tra = pd.DataFrame()

    return all_grouped_clustering_data_tra


def main():
    args = get_args()

    # Read the peak result CSV file and filter by polarity patterns
    peak_result_path = os.path.join(args.indir, 'temp', 'result.csv')
    peak_result = pd.read_csv(peak_result_path)
    
    peak_result_p = peak_result[peak_result['sampleID'].str.contains('_P', case=False)]
    peak_result_n = peak_result[peak_result['sampleID'].str.contains('_N|_HN', case=False, regex=True)]

    # Read the ALL_ions Excel file and create a dictionary of theoretical RTs
    all_ions_path = os.path.join(args.indir, 'ALL_ions.xlsx')
    all_ions_df = pd.read_excel(all_ions_path)
    all_ions_df.rename(columns={'RT (min)': 'rt_theory', 'mw ID': 'mwID'}, inplace=True)
    all_ions = pd.Series(all_ions_df.rt_theory.values, index=all_ions_df.mwID).to_dict()

    # Read the sample information CSV file
    sample_info_path = os.path.join(args.indir, 'sample_info.csv')
    sample_info = pd.read_csv(sample_info_path)
    all_sample = sample_info["sample_name"].tolist()
    
    # Process both positive and negative models sequentially
    models = [("pos", peak_result_p), ("neg", peak_result_n)]

    for model_name, tmp_peak_result in models:
        if len(tmp_peak_result) > 0:
            # Execute clustering algorithm
            clustering_result = Clustering_algorithm(tmp_peak_result, all_ions, all_sample, args.type)
            
            # Save raw clustering result table
            out_table_path = os.path.join(args.indir, 'temp', f'peak_table_{model_name}_rt.csv')
            clustering_result.to_csv(out_table_path, index=False)

            # Melt the clustering result DataFrame
            melted_clustering_result = clustering_result.melt(
                id_vars=['mw ID'], 
                var_name='sampleID', 
                value_name='area'
            )

            # Merge with the original peak result
            peak_clustering_result = pd.merge(melted_clustering_result, peak_result, on=["mw ID", "sampleID", "area"])

            # Filter and rename specific columns for the final result
            columns_to_keep = [
                "sampleID", "mw ID", "area", "int", "rt", "Width", "sn", "sn_2", "sn_3", "sn_5", 
                "rtmin", "rtmax", "rt_theoretic", "class", "baseline", "points", "min_rt", "max_rt", "lr_diff"
            ]
            new_column_names = [
                "Sample Name", "Component Name", "Area", "Height", "Retention Time", "Width at 50%", 
                "Signal/Noise", "Signal/Noise_2", "Signal/Noise_3", "Signal/Noise_5", "rtmin", "rtmax", 
                "rt_theoretic", "peak_class", "baseline", "points", "min_rt", "max_rt", "lr_diff"
            ]
            
            peak_clustering_result = peak_clustering_result[columns_to_keep]
            peak_clustering_result.columns = new_column_names

            # Save the final peak clustering result to a tab-separated file
            out_list_path = os.path.join(args.indir, 'temp', f'peak_list_{model_name}.txt')
            peak_clustering_result.to_csv(out_list_path, sep="\t")


if __name__ == '__main__':
    main()