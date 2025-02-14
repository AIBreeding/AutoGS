import pathlib
import os
import sys
import pprint
import sklearn
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from autogs.data.tools.reg_metrics import (mae_score as mae,
                             mse_score as mse,
                             rmse_score as rmse,
                             r2_score as r2,
                             rmsle_score as rmsle,
                             mape_score as mape,
                             medae_score as medae,
                             pcc_score as pcc)

from autogs.model import RegAutoGS
from autogs.data import datautils

# read data
phen_file_path = "./test/dataset/trainset/Pheno/"
env_file_path = "./test/dataset/trainset/Env/"
geno_file_path = "./test/dataset/trainset/Geno/YI_All.vcf"
ref_path = "./test/docs/maizeRef_ALL.csv"
file_names = ["DEH1_2020", "DEH1_2021", "IAH2_2021", "IAH3_2021", "IAH4_2021", "WIH2_2020", "WIH2_2021"]

com_phen_data, com_env_data, dynamic_window_avg, env_transformed_data, \
gendata, PGE = datautils.process_data(phen_file_path, env_file_path, geno_file_path, ref_path, file_names)

# Access to phenotype （Yield_Mg_ha） and feature data （Gneo and Env）
columns_to_extract = [0, 1, 8] # Get Columns Env, Hybrid, and Yield_Mg_ha

columns_from_11_to_end = list(range(11, PGE.shape[1])) 
columns_indices = columns_to_extract + columns_from_11_to_end
extracted_columns = PGE.iloc[:, columns_indices]
extracted_columns = pd.DataFrame(extracted_columns.dropna().reset_index(drop=True))

snp = pd.DataFrame(extracted_columns.iloc[:,3:])
scaler = StandardScaler()
scaled_snp = scaler.fit_transform(snp)
X = pd.DataFrame(scaled_snp,columns=snp.columns)
y = extracted_columns['Yield_Mg_ha']
y = pd.core.series.Series(y)

# train AutoGS model for reg prediction
reg = RegAutoGS(
    y=y,
    X=X, 
    test_size=0.2, 
    n_splits=5, 
    n_trial=5, 
    reload_study=True,
    reload_trial=True, 
    write_folder=os.getcwd()+'/results/', 
    metric_optimise=r2, 
    metric_assess=[mae, mse, rmse, pcc, r2, rmsle, mape, medae],
    optimization_objective='maximize', 
    models_optimize=['LightGBM','XGBoost','CatBoost','BayesianRidge'], 
    models_assess=['LightGBM','XGBoost','CatBoost','BayesianRidge'], 
    early_stopping_rounds=5, 
    random_state=2024
)
reg.train() # train model
#reg.CalSHAP(n_train_points=200,n_test_points=200,cluster=False) # cal SHAP value