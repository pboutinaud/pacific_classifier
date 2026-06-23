#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate External-like Test Data for Pipeline Validation.

This script creates a test dataset that simulates external user data by:
1. Removing a large subset of columns (variables) - default keeps only ~10 variables
2. Introducing missing values (NaN) in remaining cells - default 1%
3. Preserving the data format and types

The generated test data simulates the scenario where an external user has
their own dataset with only a small subset of the Pacific study variables.
This allows testing the complete pipeline workflow:
    Reference synthetics (557 vars) + External data (~10 vars)
    → Intersection → Training → Package creation

Usage:
    python generate_test_data.py --output_file <path>

Default behavior:
    - Source: data/synthetics_per_clustername.xlsx (next to this script)
    - Removes ~98% of columns (keeps ~10 variables)
    - Introduces 1% NaN values

Copyright (c) 2025 FEALINX - Pacific Project
Licensed under AGPL-3.0 and CC BY-NC-SA 4.0
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# Default paths relative to this script
SCRIPT_DIR = Path(__file__).parent
DEFAULT_SYNTHETICS = SCRIPT_DIR / "data" / "synthetics_per_clustername.xlsx"

# Default parameters for external-like test-data generation
DEFAULT_RATIO_REMOVE = 0.98  # Keep only ~10 variables out of 557
DEFAULT_RATIO_NAN = 0.01     # 1% missing values


def load_dataset_excel(filepath: Path) -> Tuple[pd.DataFrame, pd.DataFrame, dict, set, pd.DataFrame]:
    """
    Load dataset from Excel file with standard Pacific format.

    Parameters
    ----------
    filepath : Path
        Path to the Excel file.

    Returns
    -------
    tuple
        (training_datas, objective_datas, reverse_dict, 
         categorical_features, variable_hierarchy)
    """
    training_datas = pd.read_excel(
        filepath, sheet_name='training_datas',
        header=0, index_col=0,
        converters={"Participant": str},
    )
    
    objective_datas = pd.read_excel(
        filepath, sheet_name='objective_datas',
        header=0, index_col=0,
        converters={"Participant": str},
    )
    
    reverse_dict = pd.read_excel(filepath, sheet_name='reverse_dict', index_col=0)
    reverse_dict = reverse_dict['Modality'].to_dict()
    
    categorical_features = pd.read_excel(filepath, sheet_name='categorical_features', index_col=0)
    categorical_features = set(categorical_features.index)
    
    variable_hierarchy = pd.read_excel(filepath, sheet_name='variable_hierarchy', index_col=0)
    
    return training_datas, objective_datas, reverse_dict, categorical_features, variable_hierarchy


def save_dataset_excel(
    training_datas: pd.DataFrame,
    objective_datas: pd.DataFrame,
    reverse_dict: dict,
    categorical_features: set,
    variable_hierarchy: pd.DataFrame,
    filepath: Path
) -> None:
    """
    Save dataset to Excel file with standard Pacific format.

    Parameters
    ----------
    training_datas : pd.DataFrame
        Feature matrix.
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
        variable_hierarchy.to_excel(writer, sheet_name='variable_hierarchy', index=True)


def remove_columns(
    training_datas: pd.DataFrame,
    reverse_dict: dict,
    categorical_features: set,
    variable_hierarchy: pd.DataFrame,
    ratio_remove: float = DEFAULT_RATIO_REMOVE,
    seed: int = 42
) -> Tuple[pd.DataFrame, dict, set, pd.DataFrame, List[str]]:
    """
    Remove a random subset of columns to simulate missing variables.

    Parameters
    ----------
    training_datas : pd.DataFrame
        Original training data.
    reverse_dict : dict
        Variable to modality mapping.
    categorical_features : set
        Set of categorical features.
    variable_hierarchy : pd.DataFrame
        Variable importance ranking.
    ratio_remove : float, default=0.98
        Proportion of columns to remove (0-1). Default is 98% to keep only
        ~10 variables, producing external-like test data with very limited
        variable overlap.
    seed : int, default=42
        Random seed.

    Returns
    -------
    tuple
        (filtered_training, filtered_reverse_dict, filtered_categorical,
         filtered_hierarchy, removed_columns)
    """
    rng = np.random.default_rng(seed)
    columns = training_datas.columns.tolist()
    n_remove = int(len(columns) * ratio_remove)
    
    # Randomly select columns to remove
    remove_indices = rng.choice(len(columns), size=n_remove, replace=False)
    columns_to_remove = [columns[i] for i in remove_indices]
    columns_to_keep = [c for c in columns if c not in columns_to_remove]
    
    # Filter training data
    filtered_training = training_datas[columns_to_keep].copy()
    
    # Filter reverse dict
    filtered_reverse = {k: v for k, v in reverse_dict.items() if k in columns_to_keep}
    
    # Filter categorical features
    filtered_categorical = categorical_features.intersection(set(columns_to_keep))
    
    # Filter variable hierarchy
    filtered_hierarchy = variable_hierarchy.loc[
        variable_hierarchy.index.isin(columns_to_keep)
    ].copy()
    
    return filtered_training, filtered_reverse, filtered_categorical, filtered_hierarchy, columns_to_remove


def introduce_missing_values(
    training_datas: pd.DataFrame,
    categorical_features: set,
    ratio_nan: float = 0.05,
    seed: int = 42
) -> pd.DataFrame:
    """
    Introduce random missing values (NaN) in the dataset.

    Parameters
    ----------
    training_datas : pd.DataFrame
        Input training data.
    categorical_features : set
        Set of categorical features (will be handled separately).
    ratio_nan : float, default=0.05
        Proportion of cells to set to NaN (0-1).
    seed : int, default=42
        Random seed.

    Returns
    -------
    pd.DataFrame
        DataFrame with introduced missing values.
    """
    rng = np.random.default_rng(seed)
    result = training_datas.copy()
    
    n_rows, n_cols = result.shape
    n_cells = n_rows * n_cols
    n_nan = int(n_cells * ratio_nan)
    
    # Randomly select cells to set to NaN
    flat_indices = rng.choice(n_cells, size=n_nan, replace=False)
    row_indices = flat_indices // n_cols
    col_indices = flat_indices % n_cols
    
    for row_idx, col_idx in zip(row_indices, col_indices):
        col_name = result.columns[col_idx]
        result.iloc[row_idx, col_idx] = np.nan
    
    return result


def generate_external_like_data(
    synthetics_file: Path,
    output_file: Path,
    ratio_remove_columns: float = DEFAULT_RATIO_REMOVE,
    ratio_nan: float = DEFAULT_RATIO_NAN,
    seed: int = 42,
    verbose: bool = True
) -> None:
    """
    Generate external-like test data from synthetic data.

    This function simulates the scenario where an external user has their
    own dataset with only a small subset of the Pacific study variables.

    Parameters
    ----------
    synthetics_file : Path
        Path to source synthetic data file (default: data/synthetics_per_clustername.xlsx).
    output_file : Path
        Path for output test data file.
    ratio_remove_columns : float, default=0.98
        Proportion of columns to remove. Default is 98% to keep only ~10
        variables, producing external-like test data with very limited overlap.
    ratio_nan : float, default=0.01
        Proportion of cells to set to NaN (default 1%).
    seed : int, default=42
        Random seed.
    verbose : bool, default=True
        Print progress information.
    """
    if verbose:
        print(f"Loading synthetic data from: {synthetics_file}")
    
    # Load original data
    training_datas, objective_datas, reverse_dict, categorical_features, variable_hierarchy = \
        load_dataset_excel(synthetics_file)
    
    if verbose:
        print(f"Original data: {training_datas.shape[0]} samples, {training_datas.shape[1]} features")
    
    # Remove columns
    training_reduced, reverse_reduced, categorical_reduced, hierarchy_reduced, removed_cols = \
        remove_columns(
            training_datas, reverse_dict, categorical_features, variable_hierarchy,
            ratio_remove=ratio_remove_columns, seed=seed
        )
    
    if verbose:
        print(f"After column removal: {training_reduced.shape[1]} features ({len(removed_cols)} removed)")
        print(f"Removed columns: {removed_cols[:5]}..." if len(removed_cols) > 5 else f"Removed columns: {removed_cols}")
    
    # Introduce missing values
    training_with_nan = introduce_missing_values(
        training_reduced, categorical_reduced,
        ratio_nan=ratio_nan, seed=seed + 1
    )
    
    n_nan = training_with_nan.isna().sum().sum()
    n_cells = training_with_nan.shape[0] * training_with_nan.shape[1]
    actual_nan_ratio = n_nan / n_cells
    
    if verbose:
        print(f"Introduced {n_nan} NaN values ({actual_nan_ratio:.2%} of cells)")
    
    # Update objective data to mark as test data
    objective_test = objective_datas.copy()
    # Keep original labels for validation, use 'ext_' prefix to avoid conflicts
    # with synthetic data indices when used together in MissForest
    new_index = [f"ext_{i:04d}" for i in range(len(objective_test))]
    training_with_nan.index = pd.Index(new_index, name='Participant')
    objective_test.index = pd.Index(new_index, name='Participant')
    objective_test['subject_id'] = new_index
    
    # Save output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_dataset_excel(
        training_with_nan, objective_test,
        reverse_reduced, categorical_reduced, hierarchy_reduced,
        output_file
    )
    
    if verbose:
        print(f"Test data saved to: {output_file}")
        print(f"\nSummary:")
        print(f"  - Original features: {training_datas.shape[1]}")
        print(f"  - Remaining features: {training_reduced.shape[1]}")
        print(f"  - Features removed: {len(removed_cols)}")
        print(f"  - NaN cells: {n_nan} ({actual_nan_ratio:.2%})")
        print(f"  - Samples: {len(training_with_nan)}")
    
    # Save removed columns for reference
    removed_cols_file = output_file.parent / "removed_columns.txt"
    with open(removed_cols_file, 'w') as f:
        f.write('\n'.join(removed_cols))
    if verbose:
        print(f"  - Removed columns list: {removed_cols_file}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate external-like test data from Pacific synthetic data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
This script simulates the scenario where an external user has their own
dataset with only a small subset of the Pacific study variables.

Default behavior:
  - Source: {DEFAULT_SYNTHETICS}
  - Removes {DEFAULT_RATIO_REMOVE*100:.0f}% of columns (keeps ~10 variables)
  - Introduces {DEFAULT_RATIO_NAN*100:.0f}% NaN values

Examples:
    # Basic usage with defaults (~10 variables, 1% NaN)
    python generate_test_data.py --output_file /path/to/external_data.xlsx
    
    # Keep more variables (50)
    python generate_test_data.py \\
        --output_file /path/to/external_data.xlsx \\
        --ratio_remove 0.91
    
    # Custom source file
    python generate_test_data.py \\
        --synthetics_file /path/to/custom_synthetics.xlsx \\
        --output_file /path/to/external_data.xlsx
        """
    )
    
    parser.add_argument(
        '--synthetics_file', '-s',
        type=str, default=str(DEFAULT_SYNTHETICS),
        help=f'Path to source synthetic data file (default: {DEFAULT_SYNTHETICS})'
    )
    parser.add_argument(
        '--output_file', '-o',
        type=str, required=True,
        help='Path for output test data file'
    )
    parser.add_argument(
        '--ratio_remove',
        type=float, default=DEFAULT_RATIO_REMOVE,
        help=f'Proportion of columns to remove (default: {DEFAULT_RATIO_REMOVE} = {DEFAULT_RATIO_REMOVE*100:.0f}%%)'
    )
    parser.add_argument(
        '--ratio_nan',
        type=float, default=DEFAULT_RATIO_NAN,
        help=f'Proportion of cells to set to NaN (default: {DEFAULT_RATIO_NAN} = {DEFAULT_RATIO_NAN*100:.0f}%%)'
    )
    parser.add_argument(
        '--seed',
        type=int, default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    generate_external_like_data(
        synthetics_file=Path(args.synthetics_file),
        output_file=Path(args.output_file),
        ratio_remove_columns=args.ratio_remove,
        ratio_nan=args.ratio_nan,
        seed=args.seed,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
