import os
import sys
import warnings
import numpy as np
import pandas as pd
from autogs.data import WindCode
from autogs.data import EnvAvgCal
from autogs.data.tools import ReadVCF

def process_environment_data(file_path, file_names, ref_path):
    # Step 1: Combine environment data
    dataframes = []
    for file_name in file_names:
        data = pd.read_csv(f"{file_path}{file_name}.csv")
        data.insert(0, "Env", file_name)
        dataframes.append(data)
    com_env_data = pd.concat(dataframes, ignore_index=True)
    
    # Step 2: Compute dynamic window average
    ref_df = pd.read_csv(ref_path)
    dynamic_window_avg = []
    for Env in com_env_data['Env'].unique():
        env_train_set_df = com_env_data[com_env_data['Env'] == Env]
        WCWindCal = WindCode.WindCal(env_train_set_df)
        EnvAvg = WCWindCal.dynamic_window(ref_df)
        dynamic_window_avg.append(EnvAvg)
    dynamic_window_avg = pd.concat(dynamic_window_avg, ignore_index=True)
    
    # Step 3: Transform environment data
    start_col_index = dynamic_window_avg.columns.get_loc("DLs_Avg")
    columns_to_include = dynamic_window_avg.columns[start_col_index:]
    env_transformed_data = []
    for env in dynamic_window_avg['Env'].unique():
        env_data = dynamic_window_avg[dynamic_window_avg['Env'] == env]
        concatenated_values = []
        for _, row in env_data.iterrows():
            concatenated_values.extend(row[columns_to_include].astype(str).tolist())
        new_row = pd.Series([env] + concatenated_values, index=['Env'] + [f'Col_{i}' for i in range(1, len(concatenated_values) + 1)])
        env_transformed_data.append(new_row)
    env_transformed_data = pd.concat(env_transformed_data, axis=1).transpose()
    
    return com_env_data, dynamic_window_avg, env_transformed_data

def combine_phenotype_data(file_path, file_names):
    com_phen_data = pd.DataFrame()
    for file_name in file_names:
        phen_data = pd.read_csv(f"{file_path}{file_name}.csv")
        com_phen_data = pd.concat([com_phen_data, phen_data], ignore_index=True)
    return com_phen_data

def process_genotypic_data(geno_path):
    gen = ReadVCF.VCF(geno_path)
    gendata = gen.parse_records()
    colname = gendata['ID']
    gendata = gendata.iloc[:, 9:].transpose()
    gendata.columns = colname
    gendata.reset_index(drop=False, inplace=True)
    gendata.columns.values[0] = 'Hybrid'
    return gendata

def fusion_PGE_data(com_phen_data, gendata, env_transformed_data):
    condata = pd.merge(com_phen_data, gendata, on="Hybrid", how='outer')
    PGE = pd.merge(condata, env_transformed_data, on="Env", how='outer')
    return PGE

def process_data(phen_file_path, env_file_path, geno_file_path, ref_path, file_names):
    com_phen_data = combine_phenotype_data(phen_file_path, file_names)
    com_env_data, dynamic_window_avg, env_transformed_data = process_environment_data(env_file_path, file_names, ref_path)
    gendata = process_genotypic_data(geno_file_path)
    PGE = fusion_PGE_data(com_phen_data, gendata, env_transformed_data)
    
    return com_phen_data, com_env_data, dynamic_window_avg, env_transformed_data, gendata, PGE