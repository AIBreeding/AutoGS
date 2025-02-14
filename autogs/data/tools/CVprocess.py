"""Script for different cross-validation methods used in GxE prediction"""

import os
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

def STECV(data, env_column, hybrid_column, n_splits=5, output_dir="."):
    """
    "Scenario 1: Phenotype prediction of untested genotypes in tested environments."
    Performs stratified_environment_cv: stratified K-Fold cross-validation across multiple environments, ensuring that
    validation sets across environments contain the same Hybrid IDs and no overlap with training sets.

    Parameters:
        data (pd.DataFrame): The dataset.
        env_column (str): The column name representing the environment.
        hybrid_column (str): The column name representing the hybrid groups.
        n_splits (int): Number of folds (default is 5).
        output_dir (str): Directory to save CSV files.

    Returns:
        list: A list of tuples, each containing the training and validation datasets for a fold.
    """

    os.makedirs(output_dir, exist_ok=True)
    unique_hybrids = data[hybrid_column].unique()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    hybrid_folds = list(kf.split(unique_hybrids))

    folds = []

    for fold_num, (train_hybrid_idx, val_hybrid_idx) in enumerate(hybrid_folds):
        train_hybrids = unique_hybrids[train_hybrid_idx]
        val_hybrids = unique_hybrids[val_hybrid_idx]
        train_indices = []
        val_indices = []

        for env in data[env_column].unique():
            env_data = data[data[env_column] == env]
            env_train_indices = env_data[env_data[hybrid_column].isin(train_hybrids)].index.tolist()
            env_val_indices = env_data[env_data[hybrid_column].isin(val_hybrids)].index.tolist()
            train_indices.extend(env_train_indices)
            val_indices.extend(env_val_indices)

        train_data = data.loc[train_indices]
        val_data = data.loc[val_indices]
        assert not set(train_data[hybrid_column]).intersection(val_data[hybrid_column]), \
            "Train and validation sets have overlapping Hybrid IDs!"

        folds.append((train_data, val_data))

    return folds
    
def LOECV(data, env_column, output_dir="."):
    """
    "Scenario 2: Phenotype prediction of tested genotypes in untested environments."
    Performs leave_one_environment_out_cv (): Leave-One-Environment-Out Cross-Validation.

    Parameters:
        data (pd.DataFrame): The dataset.
        env_column (str): The column name representing the environment.
        output_dir (str): Directory to save CSV files.

    Returns:
        list: A list of tuples, each containing the training and validation datasets for a fold.
    """
    environments = data[env_column].unique()
    folds = []

    for i, env in enumerate(environments):
        train_data = data[data[env_column] != env]
        val_data = data[data[env_column] == env]
        folds.append((train_data, val_data))

    return folds

def LOESTCV(data, env_column, hybrid_column, test_size=0.2, output_dir="."):
    """
    "Scenario 3: Phenotype prediction of untested genotypes in untested environments."
    Performs leave_one_environment_stratified_cv: cross-validation ensuring that the validation set's Hybrid samples 
    do not overlap with those in the training set.

    Parameters:
        data (pd.DataFrame): The dataset.
        env_column (str): The column name representing the environment.
        hybrid_column (str): The column name representing the hybrid groups.
        test_size (float): Proportion of samples to include in the validation set per environment.
        output_dir (str): Directory to save CSV files.

    Returns:
        list: A list of tuples, each containing the training and validation datasets for a fold.
    """
    environments = data[env_column].unique()
    folds = []

    for i, val_env in enumerate(environments):
        val_indices = []
        train_indices = []

        env_data = data[data[env_column] == val_env]
        hybrids = env_data[hybrid_column].unique()
        sampled_hybrids = pd.Series(hybrids).sample(frac=test_size, random_state=42).tolist()
        env_val_indices = env_data[env_data[hybrid_column].isin(sampled_hybrids)].index.tolist()
        val_indices.extend(env_val_indices)

        excluded_hybrids = set(sampled_hybrids)
        for env in environments:
            if env != val_env:
                env_data = data[data[env_column] == env]
                remaining_hybrids = [hybrid for hybrid in env_data[hybrid_column].unique() if hybrid not in excluded_hybrids]
                env_train_indices = env_data[env_data[hybrid_column].isin(remaining_hybrids)].index.tolist()
                train_indices.extend(env_train_indices)

        val_data = data.loc[val_indices]
        train_data = data.loc[train_indices]

        folds.append((train_data, val_data))

    return folds
