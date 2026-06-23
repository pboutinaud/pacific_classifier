"""
Pytest configuration for the Pacific classifier package-builder tests.

Run tests with:
    conda run -n pacific python -m pytest tests/ -v

Run fast tests only:
    conda run -n pacific python -m pytest tests/ -v -m "not slow"
"""

import pytest
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a session-scoped temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture
def sample_training_dataframe():
    """Create a sample training DataFrame for testing."""
    np.random.seed(42)
    n_samples = 100
    
    df = pd.DataFrame({
        'numeric_var1': np.random.randn(n_samples),
        'numeric_var2': np.random.randn(n_samples),
        'numeric_var3': np.random.randn(n_samples),
        'cat#value_a': np.random.choice([0.0, 1.0], n_samples),
        'cat#value_b': np.random.choice([0.0, 1.0], n_samples),
    })
    df.index = [f'patient_{i:03d}' for i in range(n_samples)]
    
    return df


@pytest.fixture
def sample_objective_dataframe(sample_training_dataframe):
    """Create a sample objective DataFrame for testing."""
    n_samples = len(sample_training_dataframe)
    
    df = pd.DataFrame({
        'subject_id': sample_training_dataframe.index,
        'group': np.random.choice(['noHF', 'HFpEF', 'HFrEF'], n_samples),
        'cluster': np.random.choice([0, 1, 2, 3], n_samples),
        'clustername': np.random.choice(['healthier', 'rEF', 'pEF1', 'pEF2'], n_samples),
        'is_random': 'false',
        'is_smote': 'false',
    }, index=sample_training_dataframe.index)
    
    return df


@pytest.fixture
def sample_reverse_dict(sample_training_dataframe):
    """Create a sample reverse dictionary for testing."""
    return {col: 'unknown' for col in sample_training_dataframe.columns}


@pytest.fixture
def sample_categorical_features(sample_training_dataframe):
    """Create a sample categorical features set for testing."""
    return {col for col in sample_training_dataframe.columns if '#' in col}


@pytest.fixture
def sample_variable_hierarchy(sample_training_dataframe, sample_reverse_dict):
    """Create a sample variable hierarchy for testing."""
    return pd.DataFrame({
        'ranking': range(1, len(sample_training_dataframe.columns) + 1),
        'modality': [sample_reverse_dict.get(c, 'unknown') 
                     for c in sample_training_dataframe.columns]
    }, index=sample_training_dataframe.columns)
