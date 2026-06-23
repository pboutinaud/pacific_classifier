#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pacific Classifier - HFpEF Subtype Classification Tool.

This module provides the inference pipeline for assigning subjects to
heart-failure subtype cluster labels (HFpEF subtypes) using an ensemble of
pre-trained XGBoost classifiers trained on synthetic data.

The classification system identifies four subject groups (these are
synthetic-data class labels, not clinical diagnoses):
    - healthier: the lower-risk / control-like class
    - rEF: named after heart failure with reduced ejection fraction
    - pEF1: named after HFpEF subtype 1
    - pEF2: named after HFpEF subtype 2

Usage
-----
    # Run with default settings (classifies synthetic data as test)
    python pacific_classifier.py

    # Classify external data
    python pacific_classifier.py -f /path/to/your_data.xlsx

    # See all options
    python pacific_classifier.py --help

Copyright (c) 2025 FEALINX - Pacific Project
Licensed under AGPL-3.0 and CC BY-NC-SA 4.0

See Also
--------
    Project documentation: https://github.com/pboutinaud/pacific_classifier
"""
# %%
import argparse
import os
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from missforest import MissForest
from sklearn.metrics import classification_report, f1_score
from xgboost import XGBClassifier

# =============================================================================
# Constants
# =============================================================================

# Default synthetic data filename (standardized across all packages)
SYNTHETIC_DATA_FILENAME = "synthetic_pacific_data.xlsx"

# Default label encoder filename
LABEL_ENCODER_FILENAME = "clustername_LabelEncoder.pkl"


# =============================================================================
# Data I/O Functions
# =============================================================================

def save_dataset_excel(
        training_datas: pd.DataFrame,
        objective_datas: pd.DataFrame,
        reverse_dict: dict,
        categorical_features: set,
        variable_hierarchy: pd.DataFrame,
        training_file: Path) -> None:
    """
    Save dataset to Excel file with standard Pacific format.

    The output file contains five sheets:
        - training_datas: Feature matrix with Participant index
        - objective_datas: Target labels and metadata
        - reverse_dict: Variable to modality mapping
        - categorical_features: List of categorical variable names
        - variable_hierarchy: Variable importance ranking

    Parameters
    ----------
    training_datas : pd.DataFrame
        Feature matrix with Participant as index.
    objective_datas : pd.DataFrame
        Target labels and metadata DataFrame.
    reverse_dict : dict
        Mapping from variable name to modality.
    categorical_features : set
        Set of categorical variable names.
    variable_hierarchy : pd.DataFrame
        Variable importance ranking DataFrame.
    training_file : Path
        Output file path.
    """
    with pd.ExcelWriter(training_file) as writer:
        training_datas.to_excel(writer, sheet_name='training_datas')
        objective_datas.to_excel(writer, sheet_name='objective_datas')
        pd.DataFrame.from_dict(
            reverse_dict, orient='index', columns=['Modality']
        ).to_excel(
            writer, sheet_name='reverse_dict', index=True
        )
        pd.DataFrame(
            list(categorical_features), columns=['variable']
        ).to_excel(
            writer, sheet_name='categorical_features', index=False
        )
        variable_hierarchy.to_excel(
            writer, sheet_name='variable_hierarchy', index=True
        )


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_dataset_excel(
        training_file: Path
        ) -> Tuple[
        pd.DataFrame, pd.DataFrame, dict, set, pd.DataFrame
        ]:
    """
    Load dataset from Excel file with standard Pacific format.

    Parameters
    ----------
    training_file : Path
        Path to the Excel file.

    Returns
    -------
    tuple
        (training_datas, objective_datas, reverse_dict,
         categorical_features, variable_hierarchy)

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    training_datas = pd.read_excel(
        training_file, sheet_name='training_datas',
        header=0, index_col=0,
        converters={"Participant": str},
        )
    objective_datas = pd.read_excel(
        training_file, sheet_name='objective_datas',
        header=0, index_col=0,
        converters={"Participant": str},
        )
    reverse_dict = pd.read_excel(training_file, sheet_name='reverse_dict', index_col=0)  # noqa: E501
    reverse_dict = reverse_dict['Modality'].to_dict()
    categorical_features = pd.read_excel(training_file, sheet_name='categorical_features', index_col=0)  # noqa: E501
    categorical_features = set(categorical_features.index)
    variable_hierarchy = pd.read_excel(training_file, sheet_name='variable_hierarchy', index_col=0)  # noqa: E501
    return (
        training_datas, objective_datas,
        reverse_dict, categorical_features, variable_hierarchy
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
    Introduce missing values (punch holes) in the dataset to test robustness to missingness.

    This function randomly selects a subset of rows and columns, then sets
    the intersection cells to NA. Used for testing model robustness.

    Parameters
    ----------
    datas : pd.DataFrame
        Input DataFrame.
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
    if not isinstance(datas, pd.DataFrame):
        raise ValueError("datas must be a pandas DataFrame")
    rows_to_punch = rng.choice(
        len(datas),
        size=int(len(datas) * ratio_punch_row),
        replace=False
        )
    rows_to_punch = datas.index[rows_to_punch]
    cols_to_punch = datas.columns
    for row in rows_to_punch:
        subcols_to_punch = rng.choice(
            len(cols_to_punch),
            size=int(len(cols_to_punch) * ratio_punch_col),
            replace=False
            )
        subcols_to_punch = cols_to_punch[subcols_to_punch]
        for col in subcols_to_punch:
            datas.loc[row, col] = pd.NA
    return datas


# =============================================================================
# Imputation Functions
# =============================================================================

def fill_na(
    datas: pd.DataFrame,
    objectives: pd.DataFrame,
    categorical: set,
    group_varname: str = "group",
    X_synthetic: pd.DataFrame = None,
    Y_synthetic: pd.DataFrame = None,
    max_iter: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fill missing values using MissForest algorithm with group-wise imputation.

    Missing values are imputed separately for each group to preserve
    intra-group correlations. If synthetic data is provided, it serves
    as additional context for more accurate imputation.

    Parameters
    ----------
    datas : pd.DataFrame
        Data with missing values to impute.
    objectives : pd.DataFrame
        Metadata including group information.
    categorical : set
        Set of categorical feature names.
    group_varname : str, default="group"
        Column name in objectives containing group labels.
    X_synthetic : pd.DataFrame, optional
        Synthetic feature data for context.
    Y_synthetic : pd.DataFrame, optional
        Synthetic objective data for context.
    max_iter : int, default=5
        Maximum MissForest iterations.
    verbose : bool, default=True
        Print progress information.

    Returns
    -------
    pd.DataFrame
        DataFrame with imputed values.

    Notes
    -----
    The function uses MissForest, which is particularly effective for
    mixed-type data as it preserves feature correlations better than
    simple median/mode imputation.
    """
    use_synthetic = X_synthetic is not None and Y_synthetic is not None

    datas = datas.copy()
    initial_column_names = datas.columns
    punched_datas = []
    initial_index = datas.index
    group_indexes = {
        group: objectives[objectives[group_varname] == group].index
        for group in objectives[group_varname].unique()
        }
    # Fill missing values by group
    for group, indexes in group_indexes.items():
        if verbose:
            print('Cluster ', group)
        group_datas = datas.loc[indexes]
        if len(group_datas) < 1:
            continue
        if group_datas.isnull().any(axis=None):
            # check if there are columns that are null for all subjects
            null_rows = group_datas.isnull().sum(axis=1) == 0
            if use_synthetic:
                # Synthetic data: we can use the synthetic data as context to
                # fill holes and synthetic data does not have null rows
                filter_synth = Y_synthetic[group_varname] == group
                synth_datas = X_synthetic[filter_synth].copy()
                # Rename synthetic indices to avoid collision with real data
                synth_datas.index = [f"__synth_{i}__" for i in range(len(synth_datas))]
                pre_synth_index = group_datas.index
                group_datas = pd.concat(
                    [group_datas, synth_datas], axis=0
                )
            elif not null_rows.any():
                # No row exist w/o any NA : we need at least one
                # Find the one with the least NA
                illocmin = group_datas.isnull().sum(axis=1).argmin()
                culprit = group_datas.iloc[illocmin]
                culprit_loc = culprit.name
                cols = culprit[culprit.isnull()].index
                # Use the median value of each col to fill it
                for col in cols:
                    group_datas.loc[culprit_loc, col] = group_datas[col].median()  # noqa: E501
            mf = MissForest(
                 verbose=verbose,
                 max_iter=max_iter,
                 categorical=list(categorical))
            filled = mf.fit_transform(
                # lightgbm doesn't like some chars in column names
                group_datas
                )
            # filled.columns = initial_column_names
            if use_synthetic:
                # Remove the synthetic data
                filled = filled.loc[pre_synth_index]
            punched_datas.append(filled)
        else:
            punched_datas.append(group_datas)
    return pd.concat(punched_datas, axis=0).reindex(
        labels=initial_index,
        columns=initial_column_names
        )


# =============================================================================
# Classification Functions
# =============================================================================

def run_classifier(
    datas: pd.DataFrame,
    objectives: pd.DataFrame,
    model_dir: Path,
    labelencoder_file: Path,
    categorical: set = None,
    X_synthetic: pd.DataFrame = None,
    Y_synthetic: pd.DataFrame = None,
    fillgroup_varname: str = "group",
    max_iter: int = 5,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Run ensemble classifier on the provided data.

    Loads pre-trained XGBoost models, fills missing values if necessary,
    and generates predictions using both voting and probability averaging
    ensemble methods.

    Parameters
    ----------
    datas : pd.DataFrame
        Feature data to classify.
    objectives : pd.DataFrame
        Metadata including group information for imputation.
    model_dir : Path
        Directory containing trained model files (*.json).
    labelencoder_file : Path
        Path to pickled LabelEncoder file.
    categorical : set, optional
        Set of categorical feature names.
    X_synthetic : pd.DataFrame, optional
        Synthetic features for imputation context.
    Y_synthetic : pd.DataFrame, optional
        Synthetic objectives for imputation context.
    fillgroup_varname : str, default="group"
        Column name for group-wise imputation.
    max_iter : int, default=5
        Maximum MissForest iterations.
    verbose : bool, default=False
        Print progress information.

    Returns
    -------
    pd.DataFrame
        DataFrame containing:
        - 'voted': Majority vote prediction
        - 'probed': Probability average prediction
        - Class probability columns
    """
    # Try to load the LabelEncoder
    with open(labelencoder_file, 'rb') as f:
        labelencoder = pickle.load(f)

    # Reorder columns to match synthetic data (model training order)
    if X_synthetic is not None:
        expected_columns = X_synthetic.columns.tolist()
        # Check that all expected columns are present
        missing_cols = set(expected_columns) - set(datas.columns)
        if missing_cols:
            raise ValueError(
                f"Missing columns in input data: {missing_cols}. "
                f"Expected columns from synthetic data: {expected_columns}"
            )
        # Reorder columns to match training order
        datas = datas[expected_columns]

    # Test if any NAs in the to_classify_df
    if datas.isnull().values.any():
        if verbose:
            print(f"Found missing values in datas, filling them grouped by {fillgroup_varname}...")  # noqa: E501
        # Fill missing values in the to_classify_df
        datas = fill_na(
            datas, objectives,
            categorical=categorical, group_varname=fillgroup_varname,
            X_synthetic=X_synthetic, Y_synthetic=Y_synthetic,
            max_iter=max_iter, verbose=verbose
        )
    probas = []
    preds = []
    for model_file in model_dir.glob("*.json"):
        # Load XGBClassifier and run it on the synthetics data
        xgb = XGBClassifier()
        xgb.load_model(model_file)
        proba = xgb.predict_proba(datas)
        pred = proba.argmax(axis=1)
        probas.append(pd.DataFrame(proba, index=datas.index))
        preds.append(pred)

    votes = pd.DataFrame(
        np.array(preds), columns=datas.index
    ).mode(axis=0).iloc[0].astype(int)
    votes = votes.to_frame("voted")
    probas_df = pd.concat(probas, axis=0)
    probas_df = probas_df.groupby(probas_df.index.name).mean()
    probas_df = probas_df.div(probas_df.sum(axis=1), axis=0)
    probed = probas_df.idxmax(axis=1).to_frame("probed")

    preds_df = votes.merge(
        probed, left_index=True, right_index=True, how='inner'
    )

    # Get the prediction labels
    preds_df['voted'] = labelencoder.inverse_transform(preds_df['voted'])
    preds_df['probed'] = labelencoder.inverse_transform(preds_df['probed'])
    probas_df = probas_df.rename(
        columns=lambda x: labelencoder.inverse_transform([x])[0])
    result = preds_df.merge(
        probas_df, left_index=True, right_index=True, how='inner'
    )
    return result


# =============================================================================
# Main Function
# =============================================================================

def main():
    """
    Main entry point for the Pacific classifier.

    Reads input data, applies missing value imputation, runs the ensemble
    classifier, and saves results to an Excel file. Supports both a self-test
    mode (with the bundled synthetic data) and a research-inference mode (with
    a user-supplied dataset).
    """
    me = Path(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Run Pacific classifier")
    parser.add_argument("--classifier_package_dir", "-p", type=str, help="Path to classifier package directory (default: parent of INSTALL_DIR)")  # noqa: E501
    parser.add_argument("--to_classify_file", "-f", type=str, help="Path to Excel file to classify, default to synthetic data in CLASSIFIER_PACKAGE_DIR/data/synthetic_pacific_data.xlsx. Refer to the first sheet of the synthetic data for an example of data format.")  # noqa: E501
    parser.add_argument("--output_dir", "-o", type=str, help="Output directory (default: CLASSIFIER_PACKAGE_DIR/output)")  # noqa: E501
    parser.add_argument("--ensemble", "-e", choices=["vote", "proba"], required=False, help="Force method of ensemble else the one specified in the CLASSIFIER_PACKAGE_DIR will be choosen")  # noqa: E501
    parser.add_argument("--testing", "-t", type=float,  help="Run in testing mode punch holes in the dataset with given probability (default: None)")  # noqa: E501
    parser.add_argument("--max_iter", "-m", type=int,  help="max_iter for MissForest missing values filling (default: 5)")  # noqa: E501
    parser.add_argument("--quiet", "-q", action="store_true", help="Verbose mode (default: False)")  # noqa: E501
    args = parser.parse_args()

    classifier_package_dir = Path(args.classifier_package_dir) if args.classifier_package_dir else me.parent / "data"  # noqa: E501
    output_dir = Path(args.output_dir) if args.output_dir else classifier_package_dir.parent / "output"  # noqa: E501
    ensemble = args.ensemble if args.ensemble else None
    testing = args.testing if args.testing is not None else None
    max_iter = args.max_iter if args.max_iter is not None else 5  # noqa: E501
    verbose = False if args.quiet else True
    if args.to_classify_file is None:
        # If no file is provided, use the default synthetic data file
        to_classify_file = classifier_package_dir / SYNTHETIC_DATA_FILENAME
        if testing is None:
            testing = 0.2
            if args.quiet is None:
                verbose = True
    else:
        to_classify_file = Path(args.to_classify_file)
    if not to_classify_file.exists():
        raise FileNotFoundError(f"File {to_classify_file} does not exist. Please provide a valid file (-f).")  # noqa: E501

    if verbose:
        print(f"classifier_package_dir: {classifier_package_dir}")
        print(f"to_classify_file: {to_classify_file}")
        print(f"output_dir: {output_dir}")
        print(f"ensemble: {ensemble}")
        print(f"max_iter: {max_iter}")
        print(f"testing: {testing}")
        print(f"quiet: {not verbose}")

    # ----------------------------------------------------------------------------
    # Read data
    output_dir.mkdir(exist_ok=True, parents=True)
    model_dir = classifier_package_dir / "models"
    labelencoder_file = classifier_package_dir / LABEL_ENCODER_FILENAME

    to_classify_df = pd.read_excel(
        to_classify_file, index_col=0,
        sheet_name=0,
        converters={"Participant": str},
    )
    try:  # Try to read an objective dataframe
        objective_df = pd.read_excel(
            to_classify_file, index_col=0, sheet_name="objective_datas",
            converters={"Participant": str},
            )
        if verbose:
            print(f"Objective data loaded from {to_classify_file}")
            if 'group' in objective_df.columns:
                print(f"Groups found in objective data: {objective_df['group'].value_counts().to_dict()}")
    except (ValueError, KeyError) as e:
        # If it fails, create an empty DataFrame with the default group
        objective_df = pd.DataFrame(index=to_classify_df.index)
        objective_df['group'] = "Unknown"
        objective_df['subject_id'] = to_classify_df.index
        print(f"WARNING: No objective data found in {to_classify_file} (error: {e})")
        print(f"         Creating default objective data with group Unknown for all subjects.")
        print(f"If your file contains group information (\"noHF\", \"HFpEF\", \"HFrEF\"), make sure it's in the column 'group' and in the sheet is named 'objective_datas'.")

    # Define the ensemble method
    if ensemble is None:
        # Find the ensemble method used in the package
        best_model_file = classifier_package_dir / "best_model_probas.txt"
        if best_model_file.exists():
            ensemble = 'proba'
        else:
            best_model_file = classifier_package_dir / "best_model_voted.txt"
            if best_model_file.exists():
                ensemble = 'vote'
            else:
                raise ValueError(f"No ensemble method found in the package dir {str(best_model_file)}.\nPlease specify --ensemble option.")  # noqa: E501

    # Load synthetics data for imputation context
    (
        synthetic_datas, synthetic_objective_datas,
        reverse_dict, categorical_features, variable_hierarchy
        ) = load_dataset_excel(classifier_package_dir / SYNTHETIC_DATA_FILENAME)

    # Run classifier
    if testing is not None and isinstance(testing, float) and testing > 0.0:
        # In test mode, punch holes in the dataset
        if verbose:
            print("Running in testing mode, punching holes in the dataset...")
        # Punch holes in the dataset
        test_df = punch(
            to_classify_df,
            ratio_punch_col=testing,
            ratio_punch_row=testing,
        )
        new_index = [f"test_{i}" for i in range(len(test_df))]
        old_index_name = test_df.index.name
        test_df["NEWINDEX"] = new_index
        test_df.set_index("NEWINDEX", drop=True, inplace=True)
        test_df.index.name = old_index_name

        old_index_name = objective_df.index.name
        objective_df["NEWINDEX"] = new_index
        objective_df.set_index("NEWINDEX", drop=True, inplace=True)
        objective_df.index.name = old_index_name
        if verbose:
            print("\nHoles punched in the dataset:")
            holes = test_df.isna().sum()
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # noqa: E501
                print(holes)
        result = run_classifier(
            test_df,
            objective_df,
            model_dir,
            labelencoder_file,
            categorical_features,
            X_synthetic=synthetic_datas,
            Y_synthetic=synthetic_objective_datas,
            fillgroup_varname='group',
            verbose=verbose,
            max_iter=max_iter,
            )
    else:
        # In normal mode, no holes punched in the dataset
        if verbose:
            print("Running in normal mode, no holes punched in the dataset.")
        result = run_classifier(
            to_classify_df,
            objective_df,
            model_dir,
            labelencoder_file,
            categorical_features,
            X_synthetic=synthetic_datas,
            Y_synthetic=synthetic_objective_datas,
            fillgroup_varname='group',
            verbose=verbose,
            max_iter=max_iter,
            )
    # Load the LabelEncoder
    with open(labelencoder_file, 'rb') as f:
        labelencoder = pickle.load(f)

    # Ensemble the results and save the classified data
    # NB: `ensemble` is 'vote' or 'proba' (the --ensemble choices / auto-detected
    # values), so this must compare against 'vote', not 'voted'.
    if ensemble == 'vote':
        if verbose:
            print("Using voted ensemble method for classification.")
        objective_df["predicted_clustername"] = result['voted']
        objective_df["predicted_cluster"] = labelencoder.transform(result['voted'])  # noqa: E501
    else:
        if verbose:
            print("Using averaged probability ensemble method for classification.")  # noqa: E501
        objective_df["predicted_clustername"] = result['probed']
        objective_df["predicted_cluster"] = labelencoder.transform(result['probed'])  # noqa: E501

    result_file = output_dir / "classified_data.xlsx"
    save_dataset_excel(
        to_classify_df,
        objective_df,
        reverse_dict,
        categorical_features,
        variable_hierarchy,
        result_file,
    )
    print(f"Saved classified synthetics data to {result_file}")

    # Compute the f1_score of the classification if the objective data is
    # available
    if verbose and 'clustername' in objective_df.columns:
        y_true = objective_df['clustername']
        y_pred = objective_df['predicted_clustername']
        f1 = f1_score(y_true, y_pred, average='weighted')
        print(f"F1 Score: {f1:.4f}")
        print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    main()

# %%
