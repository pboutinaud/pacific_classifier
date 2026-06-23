#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pacific Classifier - Exportable Package Creation Pipeline.

This module provides the main pipeline for creating exportable classifier
packages from Pacific synthetic data, customized for external users who have
their own datasets with a subset of Pacific study variables.

The pipeline workflow:
    1. Load reference Pacific synthetic data (557 variables)
    2. Load external user data (subset of variables)
    3. Compute variable INTERSECTION (synthetics ∩ external)
    4. Filter synthetics to common variables
    5. Optional feature reduction
    6. Train ensemble classifier on filtered synthetics
    7. Create exportable package

The classification system identifies four heart failure subject groups (these are
synthetic-data class labels, not clinical diagnoses):
    - healthier: the lower-risk / control-like class
    - rEF: named after heart failure with reduced ejection fraction
    - pEF1: named after HFpEF subtype 1
    - pEF2: named after HFpEF subtype 2

Usage:
    python create_exportable_package.py --external_data <path> \\
        --output_dir <path> --package_name <name>

Copyright (c) 2025 FEALINX - Pacific Project
Licensed under AGPL-3.0 and CC BY-NC-SA 4.0

See Also
--------
    Project documentation: https://github.com/pboutinaud/pacific_classifier
"""

import argparse
import logging
import os
import pickle
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from xgboost import XGBClassifier

# =============================================================================
# Constants
# =============================================================================

# Performance threshold for warning, derived from cross-validation on the real Pacific
# cohort (n=155). The packaged classifier is trained on synthetic data only; this
# threshold is a heuristic and not a measure of real-world or clinical validity.
F1_WARNING_THRESHOLD = 0.7

# Target variable name for classification
TARGET_COLUMN = 'clustername'

# Default hyperparameters for XGBoost, tuned on the real Pacific cohort (n=155).
# The packaged classifier itself is trained on synthetic data only.
DEFAULT_HYPERPARAMETERS = {
    'colsample_bylevel': 0.4,
    'colsample_bynode': 0.6,
    'colsample_bytree': 0.74,
    'gamma': 1.88,
    'learning_rate': 0.03,
    'max_depth': 5,
    'reg_alpha': 2.0,
    'reg_lambda': 2.0,
    'n_estimators': 2000,
    'split': 0.33,
}

# Base XGBoost parameters (fixed for all models)
XGBOOST_BASE_PARAMS = {
    'device': 'cpu',
    'early_stopping_rounds': 100,
    'enable_categorical': False,
    'eval_metric': 'mlogloss',
    'importance_type': 'total_gain',
    'objective': 'multi:softmax',
    'random_state': 42,
    'tree_method': 'hist',
    'verbosity': 0,
}


# =============================================================================
# Data I/O Functions
# =============================================================================

def load_dataset_excel(
    filepath: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, set, pd.DataFrame]:
    """
    Load dataset from Excel file with standard Pacific format.

    The Excel file must contain the following sheets:
        - training_datas: Feature matrix with Participant index
        - objective_datas: Target labels and metadata
        - reverse_dict: Variable to modality mapping
        - categorical_features: List of categorical variable names
        - variable_hierarchy: Variable importance ranking

    Parameters
    ----------
    filepath : Path
        Path to the Excel file containing the dataset.

    Returns
    -------
    tuple
        (training_datas, objective_datas, reverse_dict, 
         categorical_features, variable_hierarchy)

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If required sheets are missing from the Excel file.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    # Load training data (mandatory)
    training_datas = pd.read_excel(
        filepath, sheet_name='training_datas',
        header=0, index_col=0,
        converters={"Participant": str},
    )

    # Load categorical features with fallback (detect from column names with #)
    try:
        categorical_features = pd.read_excel(
            filepath, sheet_name='categorical_features', index_col=0
        )
        categorical_features = set(categorical_features.index)
    except ValueError:
        # Detect categorical features from one-hot encoded column names (contain #)
        categorical_features = {col for col in training_datas.columns if '#' in col}

    # Load objective data with fallback
    try:
        objective_datas = pd.read_excel(
            filepath, sheet_name='objective_datas',
            header=0, index_col=0,
            converters={"Participant": str},
        )
    except ValueError:
        objective_datas = pd.DataFrame(
            columns=["subject_id", "cluster", "clustername", 
                     "is_random", "is_smote"],
            index=training_datas.index
        )
        objective_datas["subject_id"] = training_datas.index

    # Load reverse dictionary with fallback
    try:
        reverse_dict = pd.read_excel(
            filepath, sheet_name='reverse_dict', index_col=0
        )
        reverse_dict = reverse_dict['Modality'].to_dict()
    except ValueError:
        reverse_dict = {k: "unknown" for k in training_datas.columns}

    # Load variable hierarchy with fallback
    try:
        variable_hierarchy = pd.read_excel(
            filepath, sheet_name='variable_hierarchy', index_col=0
        )
    except ValueError:
        variable_hierarchy = pd.DataFrame(
            index=training_datas.columns,
            columns=["ranking", "modality"]
        )
        variable_hierarchy["ranking"] = 2
        variable_hierarchy["modality"] = "unknown"

    return (
        training_datas, objective_datas,
        reverse_dict, categorical_features, variable_hierarchy
    )


def save_dataset_excel(
    training_datas: pd.DataFrame,
    objective_datas: pd.DataFrame,
    reverse_dict: Dict,
    categorical_features: set,
    variable_hierarchy: pd.DataFrame,
    filepath: Path
) -> None:
    """
    Save dataset to Excel file with standard Pacific format.

    Parameters
    ----------
    training_datas : pd.DataFrame
        Feature matrix with Participant index.
    objective_datas : pd.DataFrame
        Target labels and metadata.
    reverse_dict : dict
        Variable to modality mapping.
    categorical_features : set
        Set of categorical variable names.
    variable_hierarchy : pd.DataFrame
        Variable importance ranking.
    filepath : Path
        Output file path.
    """
    with pd.ExcelWriter(filepath) as writer:
        training_datas.to_excel(writer, sheet_name='training_datas')
        objective_datas.to_excel(writer, sheet_name='objective_datas')
        pd.DataFrame.from_dict(
            reverse_dict, orient='index', columns=['Modality']
        ).to_excel(writer, sheet_name='reverse_dict', index=True)
        pd.DataFrame(
            list(categorical_features), columns=['variable']
        ).to_excel(writer, sheet_name='categorical_features', index=False)
        variable_hierarchy.to_excel(
            writer, sheet_name='variable_hierarchy', index=True
        )


# =============================================================================
# Data Processing Functions
# =============================================================================

def punch(
    datas: pd.DataFrame,
    ratio_punch_col: float = 0.2,
    ratio_punch_row: float = 0.2,
    rng: np.random.Generator = None
) -> pd.DataFrame:
    """
    Introduce missing values (punch holes) in the dataset to improve robustness to missingness.

    This function randomly selects a subset of rows and columns, then sets
    the intersection cells to NA. Used to train models robust to missing data.

    Parameters
    ----------
    datas : pd.DataFrame
        Input DataFrame to punch holes in.
    ratio_punch_col : float, default=0.2
        Proportion of columns to punch per selected row.
    ratio_punch_row : float, default=0.2
        Proportion of rows to select for punching.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values introduced.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    datas = datas.copy()
    n_rows = len(datas)
    n_cols = len(datas.columns)

    # Select rows to punch
    rows_to_punch = rng.choice(
        n_rows,
        size=int(n_rows * ratio_punch_row),
        replace=False
    )
    rows_to_punch = datas.index[rows_to_punch]

    # For each selected row, punch a subset of columns
    for row in rows_to_punch:
        cols_to_punch = rng.choice(
            n_cols,
            size=int(n_cols * ratio_punch_col),
            replace=False
        )
        cols_to_punch = datas.columns[cols_to_punch]
        for col in cols_to_punch:
            datas.loc[row, col] = pd.NA

    return datas


def seed_everything(seed: int = 42) -> np.random.Generator:
    """
    Set random seeds for reproducibility across all libraries.

    Parameters
    ----------
    seed : int, default=42
        Random seed value.

    Returns
    -------
    np.random.Generator
        Numpy random generator initialized with the seed.
    """
    import os
    import random
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


# =============================================================================
# Model Training Functions
# =============================================================================

def prepare_hyperparameters(
    base_hyperp: Dict,
    optimized_hyperp: Dict,
    use_synthetic: bool = False
) -> Tuple[Dict, float, bool]:
    """
    Merge and process hyperparameters for model training.

    Combines base hyperparameters with optimized values, extracting
    pipeline-specific parameters (split ratio, punch flag).

    Parameters
    ----------
    base_hyperp : dict
        Base XGBoost hyperparameters.
    optimized_hyperp : dict
        Optimized hyperparameters to merge.
    use_synthetic : bool, default=False
        If True, always enable data punching.

    Returns
    -------
    tuple
        (hyperparameters_dict, split_ratio, should_punch)
    """
    hyperp = base_hyperp.copy()
    hyperp.update(optimized_hyperp)

    # Extract pipeline-specific parameters
    split = hyperp.pop('split', 0.33)
    hyperp.pop('smote', None)
    to_punch = hyperp.pop('punch', 0)
    to_punch = True if use_synthetic else bool(to_punch)
    hyperp.pop('tid', None)
    hyperp.pop('loss', None)

    # Handle None/NaN values
    for key, value in list(hyperp.items()):
        if pd.isna(value):
            hyperp[key] = None

    # XGBoost requires certain parameters to be integers
    int_params = ['max_depth', 'n_estimators', 'early_stopping_rounds', 
                  'random_state', 'verbosity', 'n_jobs']
    for param in int_params:
        if param in hyperp and hyperp[param] is not None:
            hyperp[param] = int(hyperp[param])

    return hyperp, split, to_punch


def prepare_data(
    X: pd.DataFrame,
    y: pd.DataFrame,
    split: float,
    target_col: str,
    to_punch: bool = False,
    add_random: bool = False,
    label_encoder: LabelEncoder = None,
    ratio_punch_col: float = 0.1,
    ratio_punch_row: float = 0.1,
    rng: np.random.Generator = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, 
           np.ndarray, np.ndarray]:
    """
    Prepare train/test splits with optional data augmentation.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.DataFrame
        DataFrame containing target column.
    split : float
        Test set proportion (0-1).
    target_col : str
        Name of target column in y.
    to_punch : bool, default=False
        Whether to introduce missing values in training data.
    add_random : bool, default=False
        Whether to add a random feature for importance baseline.
    label_encoder : LabelEncoder, optional
        Encoder for target labels.
    ratio_punch_col : float, default=0.1
        Column punch ratio if to_punch=True.
    ratio_punch_row : float, default=0.1
        Row punch ratio if to_punch=True.
    rng : np.random.Generator, optional
        Random number generator.
    random_state : int, default=42
        Random state for train_test_split.

    Returns
    -------
    tuple
        (X_train, X_test, Y_train, Y_test, y_train_encoded, y_test_encoded)
    """
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, stratify=y[target_col], shuffle=True,
        test_size=split, random_state=random_state
    )

    # Punch training data to simulate missing values
    if to_punch:
        X_train = punch(
            X_train, 
            ratio_punch_col=ratio_punch_col,
            ratio_punch_row=ratio_punch_row, 
            rng=rng
        )

    # Ensure column alignment
    X_test = X_test[X_train.columns].copy()

    # Add random column for feature importance baseline
    if add_random:
        X_train["random"] = np.random.rand(len(X_train)).astype(np.float32)
        X_test["random"] = np.random.rand(len(X_test)).astype(np.float32)

    # Extract and encode targets
    y_train = Y_train[target_col].values
    y_test = Y_test[target_col].values

    if label_encoder is not None:
        y_train = label_encoder.transform(y_train)
        y_test = label_encoder.transform(y_test)

    return X_train, X_test, Y_train, Y_test, y_train, y_test


def train_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    hyperp: Dict
) -> XGBClassifier:
    """
    Train XGBoost classifier with early stopping.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : np.ndarray
        Training labels (encoded).
    X_test : pd.DataFrame
        Validation features.
    y_test : np.ndarray
        Validation labels (encoded).
    hyperp : dict
        XGBoost hyperparameters.

    Returns
    -------
    XGBClassifier
        Trained XGBoost model.
    """
    xgb = XGBClassifier(**hyperp)
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    return xgb


def evaluate_predictions(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    Y_test: pd.DataFrame,
    target_col: str,
    label_encoder: LabelEncoder
) -> Tuple[pd.DataFrame, float]:
    """
    Generate predictions and calculate performance metrics.

    Parameters
    ----------
    model : XGBClassifier
        Trained classifier.
    X_test : pd.DataFrame
        Test features.
    y_test : np.ndarray
        Test labels (encoded).
    Y_test : pd.DataFrame
        Test metadata DataFrame.
    target_col : str
        Name of target column.
    label_encoder : LabelEncoder
        Encoder for inverse transform.

    Returns
    -------
    tuple
        (predictions_dataframe, f1_macro_score)
    """
    pred_test = model.predict(X_test)
    proba_test = model.predict_proba(X_test)
    f1 = f1_score(y_test, pred_test, average='macro')

    # Create predictions DataFrame
    predicted = Y_test.copy()
    predicted['predictions'] = label_encoder.inverse_transform(pred_test)
    prob = pd.DataFrame(
        proba_test, 
        index=Y_test.index, 
        columns=label_encoder.classes_
    )
    predicted = pd.concat([predicted, prob], axis=1)

    return predicted, f1


def process_importances(
    shap_values_perfold: List,
    current_features: List[str],
    class_dict: Dict[int, str],
    reverse_dict: Dict[str, str],
    variable_hierarchy: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Process SHAP values into feature importances per class and globally.

    Uses mean absolute SHAP values to rank features by their contribution
    to model predictions across all classes.

    Parameters
    ----------
    shap_values_perfold : list
        SHAP values from each cross-validation fold.
    current_features : list
        List of current feature names.
    class_dict : dict
        Mapping from encoded class to class name.
    reverse_dict : dict
        Variable to modality mapping.
    variable_hierarchy : pd.DataFrame
        Pre-computed variable rankings.

    Returns
    -------
    tuple
        (global_importances_df, per_class_importances_dict)
    """
    importances_perclass = {}

    for encoded_class, class_name in class_dict.items():
        importances = []
        for i_fold in range(len(shap_values_perfold)):
            shap_df = pd.DataFrame(
                shap_values_perfold[i_fold][encoded_class].T,
                columns=current_features + ["random"]
            )
            vals = np.abs(shap_df.values).mean(0)
            importances.append(
                pd.DataFrame(
                    list(zip(current_features + ["random"], vals)),
                    columns=['variable', 'importance']
                )
            )

        # Combine and process importances
        importances = pd.concat(importances, axis=0)
        importances = importances.groupby('variable').mean(
            numeric_only=True
        ).sort_values(by='importance', ascending=False)
        importances['modality'] = importances.index.map(reverse_dict)

        try:
            importances = importances.merge(
                variable_hierarchy.drop('modality', axis=1),
                how='left', left_index=True, right_index=True
            )
        except Exception:
            importances['ranking'] = 1

        importances.reset_index(inplace=True)
        importances = importances[['variable', 'modality', 'importance', 'ranking']]
        importances_perclass[class_name] = importances

    # Calculate global importances (mean across classes)
    importances_all = pd.concat(importances_perclass.values(), axis=0)
    importances_all = importances_all.groupby('variable').mean(
        numeric_only=True
    ).sort_values(by='importance', ascending=False)
    importances_all['modality'] = importances_all.index.map(reverse_dict)
    importances_all.reset_index(inplace=True)
    importances_all = importances_all[['variable', 'modality', 'importance', 'ranking']]

    return importances_all, importances_perclass


def calculate_ensemble_metrics(
    total_predictions: List[pd.DataFrame],
    objective_data: pd.DataFrame,
    target_col: str,
    label_encoder: LabelEncoder
) -> Tuple[float, float]:
    """
    Calculate performance metrics for voting and probability ensemble methods.

    Compares two ensemble strategies:
    1. Voting: Each model votes, majority wins
    2. Probability: Average probabilities, argmax prediction

    Parameters
    ----------
    total_predictions : list
        List of prediction DataFrames from each model.
    objective_data : pd.DataFrame
        Ground truth labels.
    target_col : str
        Name of target column.
    label_encoder : LabelEncoder
        Label encoder for class names.

    Returns
    -------
    tuple
        (voted_f1_score, probability_f1_score)
    """
    df = pd.concat(total_predictions, axis=0)

    # Voting ensemble: majority vote across models (group by index = participant)
    votes = df.groupby(level=0).agg(
        predictions=pd.NamedAgg(
            column='predictions',
            aggfunc=lambda x: x.value_counts().index[0]
        ),
        count=pd.NamedAgg(column='predictions', aggfunc='count')
    )
    votes = votes.merge(
        objective_data[target_col], how='left',
        left_index=True, right_index=True
    )
    voted_f1 = f1_score(votes[target_col], votes['predictions'], average='macro')

    # Probability ensemble: average probabilities, take argmax (group by index)
    probas = pd.DataFrame(
        df.groupby(level=0)[label_encoder.classes_].mean().idxmax(axis=1),
        columns=["probed"]
    )
    probas = probas.merge(
        objective_data[target_col], how='left',
        left_index=True, right_index=True
    )
    probed_f1 = f1_score(probas[target_col], probas['probed'], average='macro')

    return voted_f1, probed_f1


# =============================================================================
# Pipeline Steps
# =============================================================================

def step0_validate_inputs(
    reference_synthetics_file: Path,
    external_data_file: Path,
    output_dir: Path,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, set, pd.DataFrame, List[str]]:
    """
    Step 0: Validate inputs, load data, and compute variable intersection.

    Loads both the reference Pacific synthetic data and the external user data,
    then computes the intersection of variables. The reference synthetics are
    filtered to only contain variables present in both datasets.

    Parameters
    ----------
    reference_synthetics_file : Path
        Path to reference Pacific synthetic data Excel file (557 variables).
    external_data_file : Path
        Path to external user data Excel file (subset of variables).
    output_dir : Path
        Output directory path.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    tuple
        (training_datas, objective_datas, reverse_dict, categorical_features,
         variable_hierarchy, common_variables)
        All components filtered to common variables only.

    Raises
    ------
    FileNotFoundError
        If any data file does not exist.
    ValueError
        If data format is invalid or no common variables found.
    """
    logger.info("=" * 80)
    logger.info("STEP 0: INPUT VALIDATION AND VARIABLE INTERSECTION")
    logger.info("=" * 80)

    # Check files exist
    if not reference_synthetics_file.exists():
        raise FileNotFoundError(
            f"Reference synthetic data file not found: {reference_synthetics_file}"
        )
    if not external_data_file.exists():
        raise FileNotFoundError(
            f"External data file not found: {external_data_file}"
        )

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load reference synthetic data (full Pacific variables)
    logger.info(f"Loading reference synthetics from: {reference_synthetics_file}")
    ref_data = load_dataset_excel(reference_synthetics_file)
    ref_training, ref_objective, ref_reverse, ref_categorical, ref_hierarchy = ref_data
    logger.info(f"  Reference data: {len(ref_training)} samples, {len(ref_training.columns)} variables")

    # Load external data (user's subset of variables)
    logger.info(f"Loading external data from: {external_data_file}")
    ext_data = load_dataset_excel(external_data_file)
    ext_training, _, _, _, _ = ext_data
    logger.info(f"  External data: {len(ext_training)} samples, {len(ext_training.columns)} variables")

    # Compute variable intersection
    ref_vars = set(ref_training.columns)
    ext_vars = set(ext_training.columns)
    common_vars = ref_vars.intersection(ext_vars)
    common_vars_list = sorted(list(common_vars))

    logger.info(f"\nVariable intersection:")
    logger.info(f"  Reference variables: {len(ref_vars)}")
    logger.info(f"  External variables: {len(ext_vars)}")
    logger.info(f"  Common variables: {len(common_vars)}")

    if len(common_vars) == 0:
        raise ValueError(
            "No common variables found between reference synthetics and external data. "
            "Please verify variable naming consistency."
        )

    # Log variables only in one dataset
    ref_only = ref_vars - common_vars
    ext_only = ext_vars - common_vars
    if ref_only:
        logger.info(f"  Variables only in reference (not used): {len(ref_only)}")
    if ext_only:
        logger.warning(f"  Variables only in external (ignored): {len(ext_only)}")
        if len(ext_only) <= 10:
            logger.warning(f"    Ignored variables: {sorted(ext_only)}")

    # Filter reference data to common variables only
    logger.info(f"\nFiltering reference synthetics to {len(common_vars)} common variables...")
    training_filtered = ref_training[common_vars_list].copy()
    reverse_filtered = {k: v for k, v in ref_reverse.items() if k in common_vars}
    categorical_filtered = ref_categorical.intersection(common_vars)
    hierarchy_filtered = ref_hierarchy.loc[
        ref_hierarchy.index.isin(common_vars)
    ].copy() if len(common_vars) > 0 else ref_hierarchy.iloc[:0]

    logger.info(f"Filtered data: {len(training_filtered)} samples, {len(training_filtered.columns)} features")
    logger.info(f"Target distribution:\n{ref_objective[TARGET_COLUMN].value_counts()}")

    # Validate required columns
    if TARGET_COLUMN not in ref_objective.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in objective_datas")

    # Save intersection report
    intersection_report = output_dir / "variable_intersection_report.txt"
    with open(intersection_report, 'w') as f:
        f.write("VARIABLE INTERSECTION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Reference file: {reference_synthetics_file}\n")
        f.write(f"External file: {external_data_file}\n\n")
        f.write(f"Reference variables: {len(ref_vars)}\n")
        f.write(f"External variables: {len(ext_vars)}\n")
        f.write(f"Common variables: {len(common_vars)}\n\n")
        f.write("COMMON VARIABLES:\n")
        for v in common_vars_list:
            f.write(f"  {v}\n")
        if ext_only:
            f.write(f"\nEXTERNAL-ONLY VARIABLES (ignored):\n")
            for v in sorted(ext_only):
                f.write(f"  {v}\n")
    logger.info(f"Intersection report saved to: {intersection_report}")

    return (training_filtered, ref_objective, reverse_filtered, 
            categorical_filtered, hierarchy_filtered, common_vars_list)


def step1_feature_reduction(
    training_datas: pd.DataFrame,
    objective_datas: pd.DataFrame,
    reverse_dict: Dict,
    categorical_features: set,
    variable_hierarchy: pd.DataFrame,
    hyperparameters: pd.DataFrame,
    reduction_dir: Path,
    logger: logging.Logger,
    reduce_rounds: int = 50,
    keep_most_important_ratio: float = 0.66,
    max_ranking_threshold: int = 6,
    n_folds: int = 1,
    hyperps_top: int = 25,
    ratio_punch_col: float = 0.1,
    ratio_punch_row: float = 0.1,
    rng: np.random.Generator = None
) -> Tuple[List[str], pd.DataFrame]:
    """
    Step 1: Iterative feature reduction using SHAP-based importance.

    Performs iterative feature elimination based on SHAP importance values.
    Features with importance below a random baseline or with low ranking
    are progressively removed.

    Parameters
    ----------
    training_datas : pd.DataFrame
        Training feature matrix.
    objective_datas : pd.DataFrame
        Objective data with target column.
    reverse_dict : dict
        Variable to modality mapping.
    categorical_features : set
        Set of categorical features.
    variable_hierarchy : pd.DataFrame
        Variable importance ranking.
    hyperparameters : pd.DataFrame
        Hyperparameter sets to sample from.
    reduction_dir : Path
        Directory to save reduction outputs.
    logger : logging.Logger
        Logger instance.
    reduce_rounds : int, default=50
        Maximum reduction iterations.
    keep_most_important_ratio : float, default=0.66
        Proportion of top features to always keep.
    max_ranking_threshold : int, default=6
        Maximum ranking threshold for removal.
    n_folds : int, default=1
        Number of cross-validation folds.
    hyperps_top : int, default=25
        Number of top hyperparameter sets to sample from.
    ratio_punch_col : float, default=0.1
        Column punch ratio.
    ratio_punch_row : float, default=0.1
        Row punch ratio.
    rng : np.random.Generator, optional
        Random generator.

    Returns
    -------
    tuple
        (final_feature_list, f1_scores_dataframe)
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: FEATURE REDUCTION")
    logger.info("=" * 80)

    reduction_dir.mkdir(parents=True, exist_ok=True)

    if rng is None:
        rng = np.random.default_rng(42)

    # Initialize label encoder
    le_reduce = LabelEncoder()
    le_reduce.fit(objective_datas[TARGET_COLUMN])
    class_names = le_reduce.classes_
    class_dict = {i: cl for i, cl in enumerate(class_names)}

    # Base hyperparameters
    hyperp_base = XGBOOST_BASE_PARAMS.copy()
    hyperp_base['num_class'] = len(class_names)
    hyperp_base['n_estimators'] = 1000

    f1_scores = []
    current_features = training_datas.columns.tolist()

    for reduce_round in tqdm(range(reduce_rounds), desc="Feature Reduction"):
        logger.info(f"\nReduction round {reduce_round}, features: {len(current_features)}")

        round_output_dir = reduction_dir / f'reduce_{reduce_round:02d}'
        round_output_dir.mkdir(exist_ok=True)

        current_training_data = training_datas[current_features].copy()

        shap_values_perfold = []
        total_predictions = []
        fold_f1_scores = []

        for i_fold in range(n_folds):
            # Sample hyperparameters
            hyperp_idx = min(len(hyperparameters) - 1, rng.integers(hyperps_top))
            hyperp, split, to_punch = prepare_hyperparameters(
                hyperp_base,
                hyperparameters.iloc[hyperp_idx].to_dict(),
                use_synthetic=True
            )

            # Prepare data
            X_train, X_test, Y_train, Y_test, y_train, y_test = prepare_data(
                current_training_data, objective_datas,
                split, TARGET_COLUMN, to_punch, add_random=True,
                label_encoder=le_reduce, ratio_punch_col=ratio_punch_col,
                ratio_punch_row=ratio_punch_row, rng=rng, random_state=42 + i_fold
            )

            # Train and evaluate
            xgb = train_model(X_train, y_train, X_test, y_test, hyperp)
            predicted, f1 = evaluate_predictions(
                xgb, X_test, y_test, Y_test, TARGET_COLUMN, le_reduce
            )
            total_predictions.append(predicted)
            fold_f1_scores.append(f1)

            # SHAP analysis
            explainer = shap.TreeExplainer(xgb, data=X_train)
            shap_values = explainer.shap_values(X_test, check_additivity=False)
            shap_values_perfold.append(shap_values)

        f1_scores.append(np.mean(fold_f1_scores))
        logger.info(f"Round {reduce_round} - F1 score: {f1_scores[-1]:.4f}")

        # Process importances
        importances_all, importances_perclass = process_importances(
            shap_values_perfold, current_features, class_dict,
            reverse_dict, variable_hierarchy
        )

        # Save importances
        with pd.ExcelWriter(round_output_dir / 'importances.xlsx') as writer:
            importances_all.to_excel(writer, sheet_name='Global')
            for class_name, df in importances_perclass.items():
                df.to_excel(writer, sheet_name=f'Cluster {class_name}')

        # Determine features to remove
        random_importance = importances_all[
            importances_all['variable'] == 'random'
        ]['importance'].values[0]
        stop_reduce = True

        if len(importances_all) > 1:
            filter_remove = pd.Series([False] * len(importances_all))
            for ranking_threshold in range(2, max_ranking_threshold):
                filter_remove = (
                    filter_remove | (
                        (importances_all['ranking'] <= ranking_threshold) &
                        (importances_all.index > max(1, round(len(importances_all) * keep_most_important_ratio)))
                    ) | (
                        (importances_all['ranking'] <= ranking_threshold) &
                        (importances_all['importance'] <= random_importance)
                    )
                )
                to_remove = filter_remove.sum()
                if to_remove >= round(len(importances_all) * (1 - keep_most_important_ratio) / 2):
                    stop_reduce = False
                    break

            if stop_reduce and len(importances_all) > 1:
                try:
                    filter_ranking = importances_all['ranking'] <= ranking_threshold
                    if filter_ranking.any():
                        last_var_idx = importances_all[filter_ranking].index[-1]
                        filter_remove = importances_all.index == last_var_idx
                        stop_reduce = False
                except Exception:
                    stop_reduce = True

        if stop_reduce or filter_remove.sum() < 1:
            logger.info(f"Stopping reduction at round {reduce_round}")
            break

        # Update current features
        remaining_features = importances_all[~filter_remove]
        current_features = remaining_features['variable'].tolist()
        if 'random' in current_features:
            current_features.remove('random')

        # Save reduced variables
        remaining_features.to_csv(round_output_dir / "reduced_variables.csv", index=False)

    # Save F1 scores evolution
    f1_scores_df = pd.DataFrame(f1_scores, columns=["f1"]).reset_index()
    f1_scores_df = f1_scores_df.rename(columns={"index": "round"})
    f1_scores_df.to_csv(reduction_dir / "f1_scores_evolution.csv", index=False)

    # Select best features
    best_round_idx = f1_scores_df['f1'].idxmax()
    best_round = int(f1_scores_df.loc[best_round_idx, 'round'])
    best_f1 = f1_scores_df.loc[best_round_idx, 'f1']

    best_features_file = reduction_dir / f'reduce_{best_round:02d}' / "importances.xlsx"
    if best_features_file.exists():
        best_features_df = pd.read_excel(best_features_file, sheet_name='Global', index_col=0)
        final_features = best_features_df['variable'].tolist()
    else:
        final_features = current_features

    if 'random' in final_features:
        final_features.remove('random')

    logger.info(f"Best F1: {best_f1:.4f} at round {best_round} with {len(final_features)} features")

    return final_features, f1_scores_df


def step2_train_classifier(
    training_datas: pd.DataFrame,
    objective_datas: pd.DataFrame,
    categorical_features: set,
    final_features: List[str],
    hyperparameters: pd.DataFrame,
    classifier_dir: Path,
    logger: logging.Logger,
    n_models: int = 50,
    hyperps_top: int = 25,
    ratio_punch_col: float = 0.1,
    ratio_punch_row: float = 0.1,
    rng: np.random.Generator = None
) -> Tuple[pd.DataFrame, str, float, int]:
    """
    Step 2: Train ensemble of XGBoost classifiers.

    Trains multiple XGBoost models with different random seeds and
    hyperparameter samples, then evaluates both voting and probability
    ensemble strategies.

    Parameters
    ----------
    training_datas : pd.DataFrame
        Training feature matrix.
    objective_datas : pd.DataFrame
        Objective data with target column.
    categorical_features : set
        Set of categorical features.
    final_features : list
        Selected features after reduction.
    hyperparameters : pd.DataFrame
        Hyperparameter sets to sample from.
    classifier_dir : Path
        Directory to save classifier outputs.
    logger : logging.Logger
        Logger instance.
    n_models : int, default=50
        Number of models to train.
    hyperps_top : int, default=25
        Number of top hyperparameter sets to sample from.
    ratio_punch_col : float, default=0.1
        Column punch ratio.
    ratio_punch_row : float, default=0.1
        Row punch ratio.
    rng : np.random.Generator, optional
        Random generator.

    Returns
    -------
    tuple
        (scores_dataframe, best_ensemble_method, best_f1, n_models_to_use)
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: TRAINING CLASSIFIER")
    logger.info("=" * 80)

    classifier_dir.mkdir(parents=True, exist_ok=True)

    if rng is None:
        rng = np.random.default_rng(42)

    # Filter data to final features
    training_final = training_datas[final_features].copy()
    categorical_final = categorical_features.intersection(set(final_features))

    # Create models directory
    model_dir = classifier_dir / 'models'
    model_dir.mkdir(exist_ok=True)

    # Initialize label encoder
    le_classifier = LabelEncoder()
    le_classifier.fit(objective_datas[TARGET_COLUMN])
    class_names = le_classifier.classes_

    # Save label encoder
    with open(model_dir / f'{TARGET_COLUMN}_LabelEncoder.pkl', 'wb') as f:
        pickle.dump(le_classifier, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Base hyperparameters
    hyperp_base = XGBOOST_BASE_PARAMS.copy()
    hyperp_base['num_class'] = len(class_names)
    hyperp_base['n_estimators'] = 5000

    # Train models
    total_predictions = []
    f1s = []
    voted_f1s = []
    probed_f1s = []

    for i_model in tqdm(range(n_models), desc="Training Models"):
        # Sample hyperparameters
        hyperp_idx = min(len(hyperparameters) - 1, rng.integers(hyperps_top))
        hyperp, split, to_punch = prepare_hyperparameters(
            hyperp_base,
            hyperparameters.iloc[hyperp_idx].to_dict(),
            use_synthetic=True
        )

        # Prepare data
        X_train, X_valid, Y_train, Y_valid, y_train, y_valid = prepare_data(
            training_final, objective_datas, split, TARGET_COLUMN,
            to_punch, add_random=False, label_encoder=le_classifier,
            ratio_punch_col=ratio_punch_col, ratio_punch_row=ratio_punch_row,
            rng=rng, random_state=42 + i_model
        )

        # Shuffle training data
        shuffle = np.random.permutation(len(X_train))
        X_train = X_train.iloc[shuffle]
        y_train = y_train[shuffle]
        X_train.reset_index(drop=True, inplace=True)

        # Train and evaluate
        xgb = train_model(X_train, y_train, X_valid, y_valid, hyperp)
        predicted, valid_f1 = evaluate_predictions(
            xgb, X_valid, y_valid, Y_valid, TARGET_COLUMN, le_classifier
        )

        # Save model
        xgb.save_model(model_dir / f"xgb_model_{i_model}.json")

        # Update metrics
        f1s.append(valid_f1)
        total_predictions.append(predicted)

        # Calculate ensemble metrics
        voted_f1, probed_f1 = calculate_ensemble_metrics(
            total_predictions, objective_datas, TARGET_COLUMN, le_classifier
        )
        voted_f1s.append(voted_f1)
        probed_f1s.append(probed_f1)

        logger.info(
            f"Model {i_model + 1} - F1: {valid_f1:.4f}, "
            f"Voted: {voted_f1:.4f}, Probed: {probed_f1:.4f}"
        )

    # Save F1 scores
    scores_df = pd.DataFrame({
        "model_f1": f1s,
        "voted_f1": voted_f1s,
        "probas_f1": probed_f1s
    })
    scores_df.to_excel(model_dir / 'f1_scores.xlsx', index=True)

    # Create confusion matrix
    total_predictions_df = pd.concat(total_predictions, axis=0)
    plotfile = classifier_dir / 'confusion_matrix.pdf'
    with PdfPages(plotfile) as pdf:
        fig, ax = plt.subplots(figsize=(10, 10))
        ConfusionMatrixDisplay.from_predictions(
            total_predictions_df[TARGET_COLUMN],
            total_predictions_df['predictions'],
            ax=ax, labels=le_classifier.classes_,
            cmap=plt.cm.Blues, values_format='',
            xticks_rotation=45, text_kw={"fontsize": 20}, colorbar=False
        )
        ax.set_title(f"Confusion Matrix for {n_models} models")
        pdf.savefig(fig)
        plt.close(fig)

    # Determine best ensemble configuration
    min_model = 3
    if len(scores_df) < min_model:
        raise ValueError(
            f"Not enough models trained. Found {len(scores_df)}, need at least {min_model}"
        )

    voted_max_f1_id = scores_df.iloc[min_model:]['voted_f1'].idxmax()
    voted_max_f1 = scores_df.loc[voted_max_f1_id]['voted_f1']
    probas_max_f1_id = scores_df.iloc[min_model:]['probas_f1'].idxmax()
    probas_max_f1 = scores_df.loc[probas_max_f1_id]['probas_f1']

    if voted_max_f1 > probas_max_f1:
        best_ensemble = 'voted'
        best_f1 = voted_max_f1
        n_models_package = voted_max_f1_id + 1
    else:
        best_ensemble = 'probas'
        best_f1 = probas_max_f1
        n_models_package = probas_max_f1_id + 1

    logger.info(
        f"Best ensemble: '{best_ensemble}' with F1 score: {best_f1:.4f} "
        f"using {n_models_package} models"
    )

    # Warning if performance is low
    if best_f1 < F1_WARNING_THRESHOLD:
        logger.warning(
            f"WARNING: Best F1 score ({best_f1:.4f}) is below threshold "
            f"({F1_WARNING_THRESHOLD}). NOTE: this F1 reflects fit to SYNTHETIC "
            f"data only and is not evidence of real-world or clinical performance. "
            f"Consider reviewing the variable intersection or contacting support."
        )

    return scores_df, best_ensemble, best_f1, n_models_package


def step3_create_package(
    training_datas: pd.DataFrame,
    objective_datas: pd.DataFrame,
    reverse_dict: Dict,
    categorical_features: set,
    variable_hierarchy: pd.DataFrame,
    final_features: List[str],
    classifier_dir: Path,
    package_dir: Path,
    package_name: str,
    best_ensemble: str,
    best_f1: float,
    n_models_package: int,
    synthetics_file: Path,
    f1_scores_reduction: pd.DataFrame,
    template_dir: Path,
    logger: logging.Logger
) -> Path:
    """
    Step 3: Create exportable package with all necessary files.

    Assembles the final package including:
    - Trained models (subset based on best ensemble)
    - Reduced synthetic data
    - Label encoder
    - Template files (README, LICENSE, requirements.txt, etc.)
    - Pipeline summary

    Parameters
    ----------
    training_datas : pd.DataFrame
        Original training data.
    objective_datas : pd.DataFrame
        Objective data.
    reverse_dict : dict
        Variable to modality mapping.
    categorical_features : set
        Set of categorical features.
    variable_hierarchy : pd.DataFrame
        Variable importance ranking.
    final_features : list
        Selected features after reduction.
    classifier_dir : Path
        Directory containing trained models.
    package_dir : Path
        Output directory for package.
    package_name : str
        Name of the package.
    best_ensemble : str
        Best ensemble method ('voted' or 'probas').
    best_f1 : float
        Best F1 score achieved.
    n_models_package : int
        Number of models to include.
    synthetics_file : Path
        Path to original synthetic data file.
    f1_scores_reduction : pd.DataFrame
        F1 scores from reduction step.
    template_dir : Path
        Directory containing template files.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    Path
        Path to created package directory.
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: CREATING PACKAGE")
    logger.info("=" * 80)

    # Create package structure
    package_path = package_dir / package_name
    package_data_dir = package_path / "data"
    package_model_dir = package_data_dir / "models"
    package_model_dir.mkdir(parents=True, exist_ok=True)

    # Copy models
    model_dir = classifier_dir / 'models'
    for i_model in range(n_models_package):
        model_file = model_dir / f"xgb_model_{i_model}.json"
        if model_file.exists():
            shutil.copy(model_file, package_model_dir)

    # Copy label encoder to data directory (required by classifier)
    shutil.copy(
        model_dir / f'{TARGET_COLUMN}_LabelEncoder.pkl',
        package_data_dir / f'{TARGET_COLUMN}_LabelEncoder.pkl'
    )

    # Filter and save synthetic data with reduced features
    training_final = training_datas[final_features].copy()
    categorical_final = categorical_features.intersection(set(final_features))
    reverse_dict_final = {k: v for k, v in reverse_dict.items() if k in final_features}
    
    # Filter variable hierarchy to final features
    variable_hierarchy_final = variable_hierarchy.loc[
        variable_hierarchy.index.isin(final_features)
    ].copy() if len(final_features) > 0 else variable_hierarchy.iloc[:0]

    save_dataset_excel(
        training_final, objective_datas,
        reverse_dict_final, categorical_final,
        variable_hierarchy_final,
        package_data_dir / 'synthetic_pacific_data.xlsx'
    )

    # Save reduced variables list
    pd.DataFrame({'variable': final_features}).to_csv(
        package_data_dir / "reduced_variables.csv", index=False
    )

    # Save best model info
    with open(package_data_dir / f"best_model_{best_ensemble}.txt", 'w') as f:
        f.write(f"Best model: {best_ensemble} with F1 score: {best_f1:.4f} ")
        f.write(f"using {n_models_package} models.\n")
        f.write(f"Pipeline completed with {len(final_features)} features after reduction.\n")
        f.write("NOTE: F1 is computed on SYNTHETIC validation data only and does not "
                "indicate real-world or clinical performance.\n")

    # Save pipeline summary
    best_reduction_f1 = float(f1_scores_reduction['f1'].max()) if f1_scores_reduction is not None else 0.0
    summary = {
        "input_file": str(synthetics_file),
        "initial_features": len(training_datas.columns),
        "final_features": len(final_features),
        "reduction_rounds": len(f1_scores_reduction) if f1_scores_reduction is not None else 0,
        "best_reduction_f1": best_reduction_f1,
        "n_models_trained": n_models_package,
        "best_model_type": best_ensemble,
        "best_model_f1": float(best_f1),
        "n_models_in_package": n_models_package,
        "created_date": datetime.now().isoformat(),
    }
    pd.DataFrame([summary]).to_excel(package_data_dir / "pipeline_summary.xlsx", index=False)

    # Copy template files
    if template_dir.exists():
        # Copy and adapt templates
        template_files = [
            ('template_LICENSE', 'LICENSE'),
            ('template_README.md', 'README.md'),
            ('template_pacific_classifier.py', 'pacific_classifier.py'),
            ('template_requirements.txt', 'requirements.txt'),
        ]
        for src_name, dst_name in template_files:
            src_file = template_dir / src_name
            if src_file.exists():
                dst_file = package_path / dst_name
                shutil.copy(src_file, dst_file)
                logger.info(f"Copied {src_name} -> {dst_name}")

        # Update README with package-specific info
        readme_file = package_path / 'README.md'
        if readme_file.exists():
            content = readme_file.read_text()
            # Replace placeholder file names
            content = content.replace('synthetics_per_group.xlsx', 'synthetic_pacific_data.xlsx')
            content = content.replace('synthetics_per_clustername.xlsx', 'synthetic_pacific_data.xlsx')
            readme_file.write_text(content)
    else:
        logger.warning(f"Template directory not found: {template_dir}")

    logger.info(f"Package created successfully at: {package_path}")
    logger.info("Package contents:")
    logger.info(f"  - {len(final_features)} features")
    logger.info(f"  - {n_models_package} models")
    logger.info(f"  - Best F1 score: {best_f1:.4f}")
    logger.info(f"  - Ensemble method: {best_ensemble}")

    return package_path


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(
    external_data_file: Path,
    output_dir: Path,
    package_name: str,
    reference_synthetics_file: Optional[Path] = None,
    hyperparams_file: Optional[Path] = None,
    skip_reduction: bool = False,
    n_models: int = 50,
    reduce_rounds: int = 50,
    verbose: bool = True
) -> Path:
    """
    Execute the complete exportable package creation pipeline.

    This is the main entry point for creating an exportable classifier
    package. The pipeline:
        1. Loads reference Pacific synthetics and external user data
        2. Computes variable intersection
        3. Trains classifier on intersected synthetics
        4. Creates exportable package

    Parameters
    ----------
    external_data_file : Path
        Path to external user data Excel file (subset of Pacific variables).
    output_dir : Path
        Output directory for pipeline outputs.
    package_name : str
        Name of the package to create.
    reference_synthetics_file : Path, optional
        Path to reference Pacific synthetic data. If None, uses the default
        location: data/synthetics_per_clustername.xlsx (next to this script).
    hyperparams_file : Path, optional
        Path to hyperparameters Excel file.
    skip_reduction : bool, default=False
        If True, skip feature reduction step.
    n_models : int, default=50
        Number of models to train.
    reduce_rounds : int, default=50
        Maximum reduction iterations.
    verbose : bool, default=True
        Enable verbose logging.

    Returns
    -------
    Path
        Path to created package directory.
    """
    # Default reference synthetics file
    script_dir = Path(__file__).parent
    if reference_synthetics_file is None:
        reference_synthetics_file = script_dir / "data" / "synthetics_per_clustername.xlsx"

    # Setup logging
    log_file = output_dir / "pipeline_log.txt"
    output_dir.mkdir(parents=True, exist_ok=True)

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file)
    ]
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO if verbose else logging.WARNING,
        handlers=handlers,
        force=True
    )
    logger = logging.getLogger(__name__)

    # Suppress warnings
    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
    simplefilter(action="ignore", category=FutureWarning)
    simplefilter(action="ignore", category=UserWarning)

    # Setup plotting
    sns.set_theme()
    plt.rcParams.update({'figure.autolayout': True})

    # Initialize random state
    rng = seed_everything(42)

    logger.info("=" * 80)
    logger.info("PACIFIC CLASSIFIER - EXPORTABLE PACKAGE CREATION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Reference synthetics: {reference_synthetics_file}")
    logger.info(f"External data: {external_data_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Package name: {package_name}")
    logger.info(f"Skip reduction: {skip_reduction}")
    logger.info(f"Number of models: {n_models}")

    # Define directories
    reduction_dir = output_dir / 'step1_reduction'
    classifier_dir = output_dir / 'step2_classifier'
    package_dir = output_dir / 'step3_package'

    # Locate template directory (relative to this script)
    template_dir = script_dir / 'templates'
    if not template_dir.exists():
        raise FileNotFoundError(
            f"Template directory not found: {template_dir}. "
            "Please ensure the 'templates' folder exists with required files."
        )

    # Step 0: Validate inputs and compute variable intersection
    result = step0_validate_inputs(
        reference_synthetics_file, external_data_file, output_dir, logger
    )
    training_datas, objective_datas, reverse_dict, categorical_features, variable_hierarchy, common_vars = result

    # Load hyperparameters
    if hyperparams_file and hyperparams_file.exists():
        hyperparameters = pd.read_excel(hyperparams_file, index_col=0)
        logger.info(f"Loaded {len(hyperparameters)} hyperparameter sets from {hyperparams_file}")
    else:
        logger.info("Using default hyperparameters")
        hyperparameters = pd.DataFrame([DEFAULT_HYPERPARAMETERS])

    # Step 1: Feature reduction (optional)
    if skip_reduction:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: FEATURE REDUCTION (SKIPPED)")
        logger.info("=" * 80)
        final_features = training_datas.columns.tolist()
        f1_scores_df = None
    else:
        final_features, f1_scores_df = step1_feature_reduction(
            training_datas, objective_datas, reverse_dict,
            categorical_features, variable_hierarchy, hyperparameters,
            reduction_dir, logger,
            reduce_rounds=reduce_rounds, rng=rng
        )

    # Step 2: Train classifier
    scores_df, best_ensemble, best_f1, n_models_package = step2_train_classifier(
        training_datas, objective_datas, categorical_features,
        final_features, hyperparameters, classifier_dir, logger,
        n_models=n_models, rng=rng
    )

    # Step 3: Create package
    package_path = step3_create_package(
        training_datas, objective_datas, reverse_dict,
        categorical_features, variable_hierarchy, final_features,
        classifier_dir, package_dir, package_name,
        best_ensemble, best_f1, n_models_package,
        external_data_file, f1_scores_df, template_dir, logger
    )

    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info(f"Package available at: {package_path}")

    return package_path


def main():
    """Command-line interface for the pipeline."""
    # Default paths relative to script location
    script_dir = Path(__file__).parent
    default_reference = script_dir / "data" / "synthetics_per_clustername.xlsx"

    parser = argparse.ArgumentParser(
        description="Create exportable Pacific classifier package for external users.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
This pipeline creates a customized classifier package for external users who
have their own dataset with a subset of Pacific study variables.

Workflow:
    1. Load reference Pacific synthetics ({default_reference.name})
    2. Load external user data (subset of variables)
    3. Compute variable INTERSECTION
    4. Train classifier on intersected synthetics
    5. Create exportable package

Examples:
    # Basic usage (external data has ~10 variables)
    python create_exportable_package.py \\
        --external_data /path/to/external_data.xlsx \\
        --output_dir /path/to/output \\
        --package_name MyPackage

    # Skip feature reduction
    python create_exportable_package.py \\
        --external_data /path/to/external_data.xlsx \\
        --output_dir /path/to/output \\
        --package_name MyPackage \\
        --skip_reduction

    # Custom reference synthetics
    python create_exportable_package.py \\
        --external_data /path/to/external_data.xlsx \\
        --reference_synthetics /path/to/custom_synthetics.xlsx \\
        --output_dir /path/to/output \\
        --package_name MyPackage
        """
    )

    parser.add_argument(
        '--external_data', '-e',
        type=str, required=True,
        help='Path to external user data Excel file (subset of Pacific variables)'
    )
    parser.add_argument(
        '--reference_synthetics', '-r',
        type=str, default=str(default_reference),
        help=f'Path to reference Pacific synthetic data (default: {default_reference})'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str, required=True,
        help='Output directory for pipeline outputs'
    )
    parser.add_argument(
        '--package_name', '-n',
        type=str, required=True,
        help='Name of the package to create'
    )
    parser.add_argument(
        '--hyperparams_file', '-p',
        type=str, default=None,
        help='Path to hyperparameters Excel file (optional)'
    )
    parser.add_argument(
        '--skip_reduction',
        action='store_true',
        help='Skip feature reduction step'
    )
    parser.add_argument(
        '--n_models',
        type=int, default=50,
        help='Number of models to train (default: 50)'
    )
    parser.add_argument(
        '--reduce_rounds',
        type=int, default=50,
        help='Maximum reduction iterations (default: 50)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    # Convert to Path objects
    external_data_file = Path(args.external_data)
    reference_synthetics_file = Path(args.reference_synthetics)
    output_dir = Path(args.output_dir)
    hyperparams_file = Path(args.hyperparams_file) if args.hyperparams_file else None

    # Run pipeline
    run_pipeline(
        external_data_file=external_data_file,
        output_dir=output_dir,
        package_name=args.package_name,
        reference_synthetics_file=reference_synthetics_file,
        hyperparams_file=hyperparams_file,
        skip_reduction=args.skip_reduction,
        n_models=args.n_models,
        reduce_rounds=args.reduce_rounds,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
