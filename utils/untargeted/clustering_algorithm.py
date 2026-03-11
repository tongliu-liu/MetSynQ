# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 09:38:56 2023

@author: liutong
"""

import os
import argparse
import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Peak mapping')
    parser.add_argument('--indir', type=str, default='indir', help='Data folder.')
    parser.add_argument('--threads', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
    parser.add_argument('--type', type=str, default="rp", help='type')
    return parser.parse_args()


def calculate_unique_counts(raw_data):
    """Calculate the number of unique sample IDs for each label."""
    return raw_data.groupby('labels')['sampleID'].nunique().reset_index(name='sample_count')


def select_optimal_cluster(grouped_clustering_data):
    """
    Select optimal cluster based on weighted combination of sample count (70%) and RT error (30%)
    
    Parameters:
    grouped_clustering_data (pd.DataFrame): DataFrame containing clustering results
        Must contain columns: 'labels', 'count', 'rt_error'
    
    Returns:
    int/str: Label of the optimal cluster
    """
    # Data validation
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
    df['count_norm'] = df['sample_count'] / df['sample_count'].max()
    
    # Normalize RT error (inverse scaling - lower error gives higher score)
    df['rt_error_norm'] = 1 - (df['rt_error'] / (df['rt_error'].max() + 1e-6))
    
    # Calculate weighted composite score
    df['score'] = 0.3 * df['count_norm'] + 0.7 * df['rt_error_norm']

    # Select cluster with highest score
    best_cluster = df.loc[df['score'].idxmax()]
    
    return best_cluster["labels"]


def clustering_algorithm(peak_result, all_ions, all_sample, sample_type):
    """
    This function performs DBSCAN clustering on retention times (RTs) grouped by 'mw ID'
    and selects the mean RT of the cluster with the smallest absolute difference to the
    theoretical RT provided in `all_ions`.
  
    Parameters:
    peak_result (pd.DataFrame): A DataFrame containing peak data with 'mw ID' and 'rt' columns.
    all_ions (dict): A dictionary where keys are 'mw ID' and values are theoretical RTs.
    all_sample (list): A list of all sample names.
    sample_type (str): Type of the sample (e.g., 'rp').
  
    Returns:
    pd.DataFrame: A DataFrame with clustered and pivoted results.
    """
    # Group the peak_result DataFrame by 'mw ID'
    grouped_mwID = peak_result.groupby('mw ID')
      
    # Initialize an empty DataFrame to store the selected RTs
    all_grouped_clustering_data_tra = pd.DataFrame()
      
    # Iterate over each group
    for name, group in grouped_mwID:
        # Extract RT list from the group
        rt_list = group["rt"].tolist()
        rt_list_2d = [[x] for x in rt_list]
        X = StandardScaler().fit_transform(rt_list_2d)
          
        # Define DBSCAN parameters
        eps = 0.4
        min_samples = 1
          
        # Fit DBSCAN model
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X)
          
        # Create a DataFrame for clustering data
        clustering_data = pd.DataFrame({'rt': rt_list, 'labels': labels})
        clustering_data["sampleID"] = group["sampleID"].tolist()
        clustering_data["area"] = group["area"].tolist()
          
        # Group by labels and calculate the mean RT for each cluster
        grouped_clustering_data = clustering_data.groupby('labels')['rt'].agg(['mean']).reset_index()
        grouped_clustering_data.columns = ['labels', 'mean_rt']
        grouped_clustering_data["count"] = clustering_data.groupby('labels').size().reset_index(name='count')["count"]
        grouped_clustering_data = pd.merge(grouped_clustering_data, clustering_data, on="labels")

        # Record the order of label and mean_rt
        grouped_clustering_data_label = grouped_clustering_data.drop_duplicates(subset=["labels"])
        label_list = grouped_clustering_data_label['labels'].tolist()
        mean_rt_list = grouped_clustering_data_label['mean_rt'].tolist()

        # Merge categories that are close together
        close_pairs = []
        for i in range(len(mean_rt_list)):
            for j in range(i + 1, len(mean_rt_list)):
                diff = abs(mean_rt_list[i] - mean_rt_list[j])
                if diff <= 0.04:
                    close_pairs.append((i, j))

        if len(close_pairs) > 0:
            for i in close_pairs:
                grouped_clustering_data["labels"] = grouped_clustering_data["labels"].replace(label_list[i[1]], label_list[i[0]])
                grouped_clustering_data["mean_rt"] = grouped_clustering_data["mean_rt"].replace(mean_rt_list[i[1]], mean_rt_list[i[0]])
        
        # Calculate rt error with theoretical rt
        if len(grouped_clustering_data) > 0:
            # Add the theoretical RT for the current 'mw ID'
            grouped_clustering_data["rt_theory"] = all_ions[name]
            grouped_clustering_data["rt_error"] = abs(grouped_clustering_data["mean_rt"] - grouped_clustering_data["rt_theory"])
            
            # Filter out the categories with the required error, and then select the least error from them
            if sample_type == "rp":
                grouped_clustering_data = grouped_clustering_data[grouped_clustering_data["rt_error"] <= 0.8]

            if len(grouped_clustering_data) > 0:
                optimal_label = select_optimal_cluster(grouped_clustering_data)
                grouped_clustering_data = grouped_clustering_data[grouped_clustering_data["labels"] == optimal_label]
                grouped_clustering_data = grouped_clustering_data.drop_duplicates(subset='sampleID')
                del grouped_clustering_data["rt"]
            
                # Pivot the data: use 'mean_rt' as index, 'sampleID' as columns, and 'area' as values
                grouped_clustering_data_tra = grouped_clustering_data.pivot_table(
                    index=['mean_rt'],
                    columns='sampleID',
                    values='area',
                    fill_value="NA"
                ).reset_index()
                
                # Add the 'mw ID' column to the pivoted data
                grouped_clustering_data_tra["mw ID"] = name
                grouped_clustering_data_tra.columns.name = None

                # Rename the 'mean_rt' column to 'rt'
                grouped_clustering_data_tra.rename(columns={'mean_rt': 'rt'}, inplace=True)
            else:
                # If no data remains, create an empty DataFrame
                grouped_clustering_data_tra = pd.DataFrame()
        else:
            # If no data remains, create an empty DataFrame
            grouped_clustering_data_tra = pd.DataFrame()
            
        all_grouped_clustering_data_tra = pd.concat([all_grouped_clustering_data_tra, grouped_clustering_data_tra])

    # Reset index and reorder columns
    all_grouped_clustering_data_tra = all_grouped_clustering_data_tra.reset_index(drop=True)
    new_order = ['mw ID', 'rt'] + [col for col in all_grouped_clustering_data_tra.columns if col not in ['mw ID', 'rt']]
    all_grouped_clustering_data_tra = all_grouped_clustering_data_tra[new_order]

    return all_grouped_clustering_data_tra


if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_arguments()

    # Read the peak result CSV file and filter rows where "mode" is "QN" (Quantitative Normalization)
    peak_result_path = os.path.join(args.indir, 'temp', 'result.csv')
    peak_result = pd.read_csv(peak_result_path)

    # Read the ALL_ions Excel file, rename columns for consistency, and create a dictionary of 'rt_theory'
    all_ions_path = os.path.join(args.indir, 'ALL_ions.xlsx')
    all_ions = pd.read_excel(all_ions_path)
    all_ions.rename(columns={'RT (min)': 'rt_theory', 'mw ID': 'mwID'}, inplace=True)
    all_ions = pd.Series(all_ions.rt_theory.values, index=all_ions.mwID).to_dict()

    # Read the sample information CSV file and convert the 'sample_name' column to a list of all sample names
    sample_info_path = os.path.join(args.indir, 'sample_info.csv')
    sample_info = pd.read_csv(sample_info_path)
    all_sample = sample_info["sample_name"].tolist()
        
    if len(peak_result) > 0:
        # Perform clustering algorithm with the peak result (QN), theoretical ion retention times, and sample names
        clustering_result = clustering_algorithm(peak_result, all_ions, all_sample, args.type)
        
        peak_table_rt_path = os.path.join(args.indir, 'temp', 'peak_table_rt.csv')
        clustering_result.to_csv(peak_table_rt_path, index=False)

        # Melt the clustering result DataFrame to have 'mw ID' as an identifier variable and samples as columns
        melted_clustering_result = clustering_result.melt(
            id_vars=['mw ID'],
            var_name='sampleID',
            value_name='value'
        )

        # Rename the 'value' column to 'area' for clarity
        melted_clustering_result.rename(columns={'value': 'area'}, inplace=True)

        # Merge the melted clustering result with the original peak result to include additional data
        peak_clustering_result = pd.merge(melted_clustering_result, peak_result, on=["mw ID", "sampleID", "area"])

        # Select specific columns for the final result and rename them to more descriptive names
        columns_to_keep = [
            "sampleID", "mw ID", "area", "int", "rt", "Width", "sn", "sn_2", 
            "sn_3", "sn_5", "rtmin", "rtmax", "rt_theoretic", "class", 
            "baseline", "points", "min_rt", "max_rt", "lr_diff"
        ]
        peak_clustering_result = peak_clustering_result[columns_to_keep]
        
        peak_clustering_result.columns = [
            "Sample Name", "Component Name", "Area", "Height", "Retention Time", 
            "Width at 50%", "Signal/Noise", "Signal/Noise_2", "Signal/Noise_3", 
            "Signal/Noise_5", "rtmin", "rtmax", "rt_theoretic", "peak_class", 
            "baseline", "points", "min_rt", "max_rt", "lr_diff"
        ]

        # Save the final peak clustering result to a tab-separated file
        peak_list_path = os.path.join(args.indir, 'temp', 'peak_list.txt')
        peak_clustering_result.to_csv(peak_list_path, sep="\t")