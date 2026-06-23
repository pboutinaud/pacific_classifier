"""
Tests for step0_validate_inputs function.

These tests verify the input validation and variable intersection
logic of the exportable classifier pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Import functions to test
import sys
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

try:
    from create_exportable_package import (
        step0_validate_inputs,
        save_dataset_excel,
        TARGET_COLUMN,
    )
    _IMPORTS_AVAILABLE = True
except ImportError as e:
    _IMPORTS_AVAILABLE = False
    _IMPORT_ERROR = str(e)


pytestmark = pytest.mark.skipif(
    not _IMPORTS_AVAILABLE,
    reason=f"Could not import required modules: {_IMPORT_ERROR if not _IMPORTS_AVAILABLE else ''}"
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def logger():
    """Create a test logger."""
    logger = logging.getLogger("test_step0")
    logger.setLevel(logging.DEBUG)
    return logger


@pytest.fixture
def reference_synthetic_data():
    """Create reference synthetic data with 20 variables."""
    np.random.seed(42)
    n_samples = 100
    
    training = pd.DataFrame({
        f'var_{i}': np.random.randn(n_samples) for i in range(15)
    })
    # Add categorical variables
    training['cat#a'] = np.random.choice([0.0, 1.0], n_samples)
    training['cat#b'] = np.random.choice([0.0, 1.0], n_samples)
    training['cat#c'] = np.random.choice([0.0, 1.0], n_samples)
    training['olink_VAR1'] = np.random.randn(n_samples)
    training['olink_VAR2'] = np.random.randn(n_samples)
    training.index = [f'synth_{i:05d}' for i in range(n_samples)]
    
    objective = pd.DataFrame({
        'subject_id': training.index,
        'group': np.random.choice(['noHF', 'HFpEF', 'HFrEF'], n_samples),
        'cluster': np.random.choice([0, 1, 2, 3], n_samples),
        'clustername': np.random.choice(['healthier', 'rEF', 'pEF1', 'pEF2'], n_samples),
        'is_random': 'false',
        'is_smote': 'false',
    }, index=training.index)
    
    reverse_dict = {col: 'clinical' for col in training.columns[:10]}
    reverse_dict.update({col: 'laboratory' for col in training.columns[10:15]})
    reverse_dict.update({col: 'clinical' for col in ['cat#a', 'cat#b', 'cat#c']})
    reverse_dict.update({col: 'olink' for col in ['olink_VAR1', 'olink_VAR2']})
    
    categorical = {'cat#a', 'cat#b', 'cat#c'}
    
    hierarchy = pd.DataFrame({
        'ranking': range(1, len(training.columns) + 1),
        'modality': [reverse_dict.get(c, 'unknown') for c in training.columns]
    }, index=training.columns)
    
    return training, objective, reverse_dict, categorical, hierarchy


@pytest.fixture
def external_data_subset():
    """Create external data with subset of variables (10 out of 20)."""
    np.random.seed(123)
    n_samples = 50
    
    # Only 10 variables that match reference
    training = pd.DataFrame({
        f'var_{i}': np.random.randn(n_samples) for i in range(8)  # 8 numeric
    })
    training['cat#a'] = np.random.choice([0.0, 1.0], n_samples)
    training['olink_VAR1'] = np.random.randn(n_samples)
    training.index = [f'ext_{i:05d}' for i in range(n_samples)]
    
    objective = pd.DataFrame({
        'subject_id': training.index,
        'group': 'Unknown',
        'cluster': np.nan,
        'clustername': np.nan,
        'is_random': 'false',
        'is_smote': 'false',
    }, index=training.index)
    
    reverse_dict = {col: 'unknown' for col in training.columns}
    categorical = {'cat#a'}
    hierarchy = pd.DataFrame({
        'ranking': [2] * len(training.columns),
        'modality': ['unknown'] * len(training.columns)
    }, index=training.columns)
    
    return training, objective, reverse_dict, categorical, hierarchy


@pytest.fixture
def external_data_extra_vars():
    """Create external data with some extra variables not in reference."""
    np.random.seed(456)
    n_samples = 50
    
    # Mix of common and extra variables
    training = pd.DataFrame({
        'var_0': np.random.randn(n_samples),  # Common
        'var_1': np.random.randn(n_samples),  # Common
        'extra_var1': np.random.randn(n_samples),  # Extra
        'extra_var2': np.random.randn(n_samples),  # Extra
    })
    training.index = [f'ext_{i:05d}' for i in range(n_samples)]
    
    objective = pd.DataFrame({
        'subject_id': training.index,
        'group': 'Unknown',
    }, index=training.index)
    
    return training, objective, {}, set(), pd.DataFrame()


@pytest.fixture
def reference_file(tmp_path, reference_synthetic_data):
    """Create reference synthetic Excel file."""
    training, objective, reverse_dict, categorical, hierarchy = reference_synthetic_data
    filepath = tmp_path / "reference_synthetics.xlsx"
    save_dataset_excel(training, objective, reverse_dict, categorical, hierarchy, filepath)
    return filepath


@pytest.fixture
def external_file_subset(tmp_path, external_data_subset):
    """Create external data Excel file with subset of variables."""
    training, objective, reverse_dict, categorical, hierarchy = external_data_subset
    filepath = tmp_path / "external_subset.xlsx"
    save_dataset_excel(training, objective, reverse_dict, categorical, hierarchy, filepath)
    return filepath


@pytest.fixture
def external_file_extra(tmp_path, external_data_extra_vars):
    """Create external data Excel file with extra variables."""
    training, objective, reverse_dict, categorical, hierarchy = external_data_extra_vars
    filepath = tmp_path / "external_extra.xlsx"
    save_dataset_excel(training, objective, reverse_dict, categorical, hierarchy, filepath)
    return filepath


# =============================================================================
# Tests for step0_validate_inputs
# =============================================================================

class TestStep0ValidateInputs:
    """Tests for step0_validate_inputs function."""

    def test_returns_correct_structure(self, reference_file, external_file_subset, 
                                        tmp_path, logger):
        """Verify function returns expected tuple structure."""
        output_dir = tmp_path / "output"
        
        result = step0_validate_inputs(
            reference_file, external_file_subset, output_dir, logger
        )
        
        assert len(result) == 6
        training, objective, reverse_dict, categorical, hierarchy, common_vars = result
        
        assert isinstance(training, pd.DataFrame)
        assert isinstance(objective, pd.DataFrame)
        assert isinstance(reverse_dict, dict)
        assert isinstance(categorical, set)
        assert isinstance(hierarchy, pd.DataFrame)
        assert isinstance(common_vars, list)

    def test_computes_intersection(self, reference_file, external_file_subset,
                                    tmp_path, logger):
        """Verify variable intersection is computed correctly."""
        output_dir = tmp_path / "output"
        
        training, _, _, _, _, common_vars = step0_validate_inputs(
            reference_file, external_file_subset, output_dir, logger
        )
        
        # External has 10 variables, all should be in common
        # (var_0 to var_7, cat#a, olink_VAR1)
        assert len(common_vars) == 10
        assert 'var_0' in common_vars
        assert 'cat#a' in common_vars
        assert 'olink_VAR1' in common_vars

    def test_filters_to_common_variables(self, reference_file, external_file_subset,
                                          tmp_path, logger):
        """Verify training data is filtered to common variables only."""
        output_dir = tmp_path / "output"
        
        training, _, _, _, _, common_vars = step0_validate_inputs(
            reference_file, external_file_subset, output_dir, logger
        )
        
        assert len(training.columns) == len(common_vars)
        assert set(training.columns) == set(common_vars)

    def test_preserves_sample_count(self, reference_file, external_file_subset,
                                     tmp_path, logger, reference_synthetic_data):
        """Verify sample count from reference is preserved."""
        output_dir = tmp_path / "output"
        ref_training, _, _, _, _ = reference_synthetic_data
        
        training, _, _, _, _, _ = step0_validate_inputs(
            reference_file, external_file_subset, output_dir, logger
        )
        
        assert len(training) == len(ref_training)

    def test_filters_categorical_features(self, reference_file, external_file_subset,
                                           tmp_path, logger):
        """Verify categorical features are filtered to common variables."""
        output_dir = tmp_path / "output"
        
        _, _, _, categorical, _, common_vars = step0_validate_inputs(
            reference_file, external_file_subset, output_dir, logger
        )
        
        # Only cat#a should be in common
        assert 'cat#a' in categorical
        assert 'cat#b' not in categorical
        assert 'cat#c' not in categorical

    def test_filters_reverse_dict(self, reference_file, external_file_subset,
                                   tmp_path, logger):
        """Verify reverse dict is filtered to common variables."""
        output_dir = tmp_path / "output"
        
        _, _, reverse_dict, _, _, common_vars = step0_validate_inputs(
            reference_file, external_file_subset, output_dir, logger
        )
        
        assert set(reverse_dict.keys()) == set(common_vars)

    def test_filters_hierarchy(self, reference_file, external_file_subset,
                                tmp_path, logger):
        """Verify hierarchy is filtered to common variables."""
        output_dir = tmp_path / "output"
        
        _, _, _, _, hierarchy, common_vars = step0_validate_inputs(
            reference_file, external_file_subset, output_dir, logger
        )
        
        assert set(hierarchy.index) == set(common_vars)

    def test_creates_output_directory(self, reference_file, external_file_subset,
                                       tmp_path, logger):
        """Verify output directory is created."""
        output_dir = tmp_path / "new_output_dir"
        
        step0_validate_inputs(
            reference_file, external_file_subset, output_dir, logger
        )
        
        assert output_dir.exists()

    def test_creates_intersection_report(self, reference_file, external_file_subset,
                                          tmp_path, logger):
        """Verify intersection report is created."""
        output_dir = tmp_path / "output"
        
        step0_validate_inputs(
            reference_file, external_file_subset, output_dir, logger
        )
        
        report_file = output_dir / "variable_intersection_report.txt"
        assert report_file.exists()
        
        content = report_file.read_text()
        assert "VARIABLE INTERSECTION REPORT" in content
        assert "Common variables:" in content

    def test_handles_extra_external_variables(self, reference_file, external_file_extra,
                                               tmp_path, logger):
        """Verify extra variables in external data are ignored."""
        output_dir = tmp_path / "output"
        
        training, _, _, _, _, common_vars = step0_validate_inputs(
            reference_file, external_file_extra, output_dir, logger
        )
        
        # Only var_0 and var_1 are common
        assert len(common_vars) == 2
        assert 'var_0' in common_vars
        assert 'var_1' in common_vars
        assert 'extra_var1' not in common_vars
        assert 'extra_var2' not in common_vars

    def test_raises_on_missing_reference_file(self, external_file_subset,
                                               tmp_path, logger):
        """Verify FileNotFoundError for missing reference file."""
        output_dir = tmp_path / "output"
        fake_reference = tmp_path / "nonexistent.xlsx"
        
        with pytest.raises(FileNotFoundError, match="Reference synthetic data"):
            step0_validate_inputs(
                fake_reference, external_file_subset, output_dir, logger
            )

    def test_raises_on_missing_external_file(self, reference_file,
                                              tmp_path, logger):
        """Verify FileNotFoundError for missing external file."""
        output_dir = tmp_path / "output"
        fake_external = tmp_path / "nonexistent.xlsx"
        
        with pytest.raises(FileNotFoundError, match="External data file"):
            step0_validate_inputs(
                reference_file, fake_external, output_dir, logger
            )

    def test_raises_on_no_common_variables(self, tmp_path, logger, 
                                            reference_synthetic_data):
        """Verify ValueError when no common variables exist."""
        # Create reference file
        training, objective, reverse_dict, categorical, hierarchy = reference_synthetic_data
        ref_file = tmp_path / "reference.xlsx"
        save_dataset_excel(training, objective, reverse_dict, categorical, hierarchy, ref_file)
        
        # Create external with completely different variables
        ext_training = pd.DataFrame({
            'totally_different_var_1': np.random.randn(50),
            'totally_different_var_2': np.random.randn(50),
        })
        ext_training.index = [f'ext_{i}' for i in range(50)]
        ext_objective = pd.DataFrame({'subject_id': ext_training.index}, index=ext_training.index)
        ext_file = tmp_path / "external.xlsx"
        save_dataset_excel(ext_training, ext_objective, {}, set(), pd.DataFrame(), ext_file)
        
        output_dir = tmp_path / "output"
        
        with pytest.raises(ValueError, match="No common variables"):
            step0_validate_inputs(ref_file, ext_file, output_dir, logger)


# =============================================================================
# Run tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
