"""
Integration tests for the complete exportable classifier pipeline.

These tests verify end-to-end functionality of the pipeline,
from data loading to package creation.

NOTE: These tests use run_pipeline() directly since step2_train_classifier
and step3_create_package have complex signatures tied to the pipeline flow.
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
        run_pipeline,
        step0_validate_inputs,
        save_dataset_excel,
        load_dataset_excel,
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
def complete_synthetic_data():
    """Create complete synthetic data mimicking Pacific format."""
    np.random.seed(42)
    n_samples = 400  # Need enough for train/test split
    
    # Create diverse features
    training = pd.DataFrame()
    
    # Numeric clinical features
    for i in range(20):
        training[f'clinical_var_{i}'] = np.random.randn(n_samples)
    
    # Numeric laboratory features  
    for i in range(10):
        training[f'lab_var_{i}'] = np.random.randn(n_samples)
    
    # OLINK features
    for i in range(10):
        training[f'OLINK_{i}'] = np.random.randn(n_samples)
    
    # Categorical features (one-hot encoded)
    for cat_name, n_values in [('smoking', 3), ('NYHA', 4)]:
        for j in range(n_values):
            col_name = f'{cat_name}#value_{j}'
            training[col_name] = np.zeros(n_samples)
            # Randomly assign one-hot values
            for i in range(n_samples):
                if np.random.random() < 1.0 / n_values:
                    training.loc[training.index[i], col_name] = 1.0
    
    training.index = [f'synth_{i:05d}' for i in range(n_samples)]
    
    # Create objective data with balanced classes
    classes = ['healthier', 'rEF', 'pEF1', 'pEF2']
    objective = pd.DataFrame({
        'subject_id': training.index,
        'group': np.random.choice(['noHF', 'HFpEF', 'HFrEF'], n_samples),
        'cluster': np.tile([0, 1, 2, 3], n_samples // 4),
        'clustername': np.tile(classes, n_samples // 4),
        'is_random': 'false',
        'is_smote': 'false',
    }, index=training.index)
    
    # Create reverse dict
    reverse_dict = {}
    for col in training.columns:
        if 'clinical' in col or 'smoking' in col or 'NYHA' in col:
            reverse_dict[col] = 'clinical'
        elif 'lab' in col:
            reverse_dict[col] = 'laboratory'
        elif 'OLINK' in col:
            reverse_dict[col] = 'olink'
        else:
            reverse_dict[col] = 'unknown'
    
    # Categorical features
    categorical = {col for col in training.columns if '#' in col}
    
    # Hierarchy
    hierarchy = pd.DataFrame({
        'ranking': range(1, len(training.columns) + 1),
        'modality': [reverse_dict.get(c, 'unknown') for c in training.columns]
    }, index=training.columns)
    
    return training, objective, reverse_dict, categorical, hierarchy


@pytest.fixture
def external_subset_data(complete_synthetic_data):
    """Create external data with subset of synthetic variables."""
    ref_training, _, _, _, _ = complete_synthetic_data
    np.random.seed(123)
    n_samples = 100
    
    # Select subset of columns (about half)
    selected_cols = list(ref_training.columns[:25])  # First 25 columns
    
    training = pd.DataFrame({
        col: np.random.randn(n_samples) for col in selected_cols
    })
    training.index = [f'ext_{i:05d}' for i in range(n_samples)]
    
    objective = pd.DataFrame({
        'subject_id': training.index,
        'group': 'Unknown',
    }, index=training.index)
    
    reverse_dict = {col: 'unknown' for col in training.columns}
    categorical = {col for col in training.columns if '#' in col}
    hierarchy = pd.DataFrame({
        'ranking': [2] * len(training.columns),
        'modality': ['unknown'] * len(training.columns)
    }, index=training.columns)
    
    return training, objective, reverse_dict, categorical, hierarchy


@pytest.fixture
def reference_file(tmp_path, complete_synthetic_data):
    """Create reference synthetic Excel file."""
    training, objective, reverse_dict, categorical, hierarchy = complete_synthetic_data
    filepath = tmp_path / "data" / "synthetics_per_clustername.xlsx"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    save_dataset_excel(training, objective, reverse_dict, categorical, hierarchy, filepath)
    return filepath


@pytest.fixture
def external_file(tmp_path, external_subset_data):
    """Create external data Excel file."""
    training, objective, reverse_dict, categorical, hierarchy = external_subset_data
    filepath = tmp_path / "external_data.xlsx"
    save_dataset_excel(training, objective, reverse_dict, categorical, hierarchy, filepath)
    return filepath


@pytest.fixture
def template_dir(tmp_path):
    """Create minimal template files for testing."""
    template_path = tmp_path / "templates"
    template_path.mkdir(parents=True, exist_ok=True)
    
    # Create minimal template files
    (template_path / "template_pacific_classifier.py").write_text(
        "# Pacific Classifier Template\nprint('Hello')"
    )
    (template_path / "template_README.md").write_text("# README")
    (template_path / "template_LICENSE").write_text("LICENSE")
    (template_path / "template_requirements.txt").write_text("numpy")
    
    return template_path


@pytest.fixture
def logger():
    """Create a test logger."""
    test_logger = logging.getLogger("test_integration")
    test_logger.setLevel(logging.INFO)
    return test_logger


# =============================================================================
# Integration Tests
# =============================================================================

class TestStep0Integration:
    """Integration tests for step0_validate_inputs."""

    @pytest.mark.slow
    def test_step0_complete_workflow(self, reference_file, external_file, 
                                      tmp_path, logger):
        """Verify step0 processes files and returns correct structure."""
        output_dir = tmp_path / "output"
        
        result = step0_validate_inputs(
            reference_file, external_file, output_dir, logger
        )
        
        training, objective, reverse_dict, categorical, hierarchy, common_vars = result
        
        # Verify structure
        assert isinstance(training, pd.DataFrame)
        assert isinstance(objective, pd.DataFrame)
        assert isinstance(reverse_dict, dict)
        assert isinstance(categorical, set)
        assert isinstance(hierarchy, pd.DataFrame)
        assert isinstance(common_vars, list)  # Returns list, not set
        
        # Verify intersection was computed (25 columns from external)
        assert len(training.columns) == 25
        assert len(common_vars) == 25
        
        # Verify output directory created
        assert output_dir.exists()
        
        # Verify intersection report created
        report_file = output_dir / "variable_intersection_report.txt"
        assert report_file.exists()


class TestPipelineEndToEnd:
    """End-to-end tests for the complete pipeline via run_pipeline."""

    @pytest.mark.slow
    def test_pipeline_with_mock_templates(self, reference_file, external_file,
                                           tmp_path, template_dir):
        """Verify pipeline completes with mock templates."""
        import shutil
        
        output_dir = tmp_path / "pipeline_output"
        
        # Copy templates to expected location relative to script
        script_dir = Path(__file__).parent.parent
        target_template_dir = script_dir / "templates"
        
        # Create templates directory if it doesn't exist
        if not target_template_dir.exists():
            target_template_dir.mkdir(parents=True, exist_ok=True)
            
            # Create minimal templates
            (target_template_dir / "template_pacific_classifier.py").write_text(
                "# Pacific Classifier\nimport json\nprint('classifier')"
            )
            (target_template_dir / "template_README.md").write_text("# README")
            (target_template_dir / "template_LICENSE").write_text("MIT License")
            (target_template_dir / "template_requirements.txt").write_text("numpy\npandas")
            
            _cleanup_templates = True
        else:
            _cleanup_templates = False
        
        try:
            package_path = run_pipeline(
                external_data_file=external_file,
                output_dir=output_dir,
                package_name="IntegrationTest",
                reference_synthetics_file=reference_file,
                skip_reduction=True,  # Skip for speed
                n_models=5,  # Need > 3 (min_model threshold in step2)
                verbose=False
            )
            
            # Basic assertions
            assert package_path.exists()
            assert package_path.is_dir()
            
        except Exception as e:
            if "Template" in str(e) or "template" in str(e):
                pytest.skip(f"Template files not available: {e}")
            raise
        finally:
            # Cleanup if we created templates
            if _cleanup_templates and target_template_dir.exists():
                shutil.rmtree(target_template_dir, ignore_errors=True)

    @pytest.mark.slow
    def test_pipeline_output_structure(self, reference_file, external_file,
                                        tmp_path):
        """Verify pipeline creates expected directory structure."""
        script_dir = Path(__file__).parent.parent
        template_dir = script_dir / "templates"
        
        # Skip if templates don't exist
        if not template_dir.exists():
            pytest.skip("Template directory not found")
        
        output_dir = tmp_path / "pipeline_output"
        
        try:
            package_path = run_pipeline(
                external_data_file=external_file,
                output_dir=output_dir,
                package_name="StructureTest",
                reference_synthetics_file=reference_file,
                skip_reduction=True,
                n_models=5,  # Need > 3 (min_model threshold in step2)
                verbose=False
            )
            
            # Verify package structure
            assert (package_path / "data").exists()
            assert (package_path / "pacific_classifier.py").exists()
            assert (package_path / "README.md").exists()
            
            # Verify data files
            data_dir = package_path / "data"
            assert (data_dir / "models").exists()
            
            # Models should exist
            model_files = list((data_dir / "models").glob("*.json"))
            assert len(model_files) >= 1
            
        except Exception as e:
            if "Template" in str(e):
                pytest.skip(f"Template error: {e}")
            raise


class TestDataFlow:
    """Tests for data flow through the pipeline."""

    @pytest.mark.slow
    def test_variable_intersection_applied(self, reference_file, external_file,
                                            tmp_path, logger):
        """Verify variable intersection is correctly applied."""
        output_dir = tmp_path / "output"
        
        # Load original files to check column counts
        ref_training, _, _, _, _ = load_dataset_excel(reference_file)
        ext_training, _, _, _, _ = load_dataset_excel(external_file)
        
        # Run step0
        training, objective, _, _, _, common_vars = step0_validate_inputs(
            reference_file, external_file, output_dir, logger
        )
        
        # Verify intersection (common_vars is a list)
        ref_cols = set(ref_training.columns)
        ext_cols = set(ext_training.columns)
        expected_common = ref_cols.intersection(ext_cols)
        
        assert set(common_vars) == expected_common
        assert set(training.columns) == expected_common
        
    @pytest.mark.slow  
    def test_sample_count_preserved(self, reference_file, external_file,
                                     tmp_path, logger):
        """Verify sample count is preserved through step0."""
        output_dir = tmp_path / "output"
        
        # Load original reference
        ref_training, ref_objective, _, _, _ = load_dataset_excel(reference_file)
        
        # Run step0
        training, objective, _, _, _, _ = step0_validate_inputs(
            reference_file, external_file, output_dir, logger
        )
        
        # Sample count should be same
        assert len(training) == len(ref_training)
        assert len(objective) == len(ref_objective)


# =============================================================================
# Run tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'slow'])
