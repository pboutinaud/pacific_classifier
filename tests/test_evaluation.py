"""
Tests for evaluation and metrics functions.

These tests verify the prediction evaluation and ensemble metrics
computation of the exportable classifier pipeline.

Note: Some complex functions (evaluate_predictions, process_importances)
require full pipeline context and are tested in test_integration.py.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

# Import functions to test
import sys
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

try:
    from create_exportable_package import (
        calculate_ensemble_metrics,
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
def label_encoder():
    """Create label encoder for cluster names."""
    le = LabelEncoder()
    le.fit(['healthier', 'pEF1', 'pEF2', 'rEF'])
    return le


@pytest.fixture
def sample_predictions_df(label_encoder):
    """Create sample prediction DataFrames for ensemble testing."""
    np.random.seed(42)
    n_samples = 50
    n_models = 3
    
    predictions = []
    classes = label_encoder.classes_
    
    for model_idx in range(n_models):
        # Create predictions with some variation
        df = pd.DataFrame({
            'Participant': [f'patient_{i:03d}' for i in range(n_samples)],
            'predictions': np.random.choice(classes, n_samples),
        })
        # Add probability columns
        probas = np.random.dirichlet(np.ones(len(classes)), n_samples)
        for i, cls in enumerate(classes):
            df[cls] = probas[:, i]
        
        df.set_index('Participant', inplace=True)
        predictions.append(df)
    
    return predictions


@pytest.fixture
def sample_objective_df(label_encoder):
    """Create sample objective DataFrame for testing."""
    np.random.seed(42)
    n_samples = 50
    classes = label_encoder.classes_
    
    df = pd.DataFrame({
        TARGET_COLUMN: np.random.choice(classes, n_samples),
    }, index=[f'patient_{i:03d}' for i in range(n_samples)])
    
    return df


# =============================================================================
# Tests for calculate_ensemble_metrics
# =============================================================================

class TestCalculateEnsembleMetrics:
    """Tests for calculate_ensemble_metrics function."""

    def test_returns_two_floats(self, sample_predictions_df, sample_objective_df,
                                 label_encoder):
        """Verify function returns two float scores."""
        voted_f1, probed_f1 = calculate_ensemble_metrics(
            sample_predictions_df,
            sample_objective_df,
            TARGET_COLUMN,
            label_encoder
        )
        
        assert isinstance(voted_f1, float)
        assert isinstance(probed_f1, float)

    def test_scores_in_valid_range(self, sample_predictions_df, sample_objective_df,
                                    label_encoder):
        """Verify F1 scores are in valid range [0, 1]."""
        voted_f1, probed_f1 = calculate_ensemble_metrics(
            sample_predictions_df,
            sample_objective_df,
            TARGET_COLUMN,
            label_encoder
        )
        
        assert 0 <= voted_f1 <= 1
        assert 0 <= probed_f1 <= 1

    def test_perfect_ensemble(self, label_encoder):
        """Verify perfect predictions give F1=1.0."""
        n_samples = 40
        classes = label_encoder.classes_
        
        # Create ground truth
        true_labels = np.tile(classes, n_samples // len(classes))
        objective_df = pd.DataFrame({
            TARGET_COLUMN: true_labels,
        }, index=[f'patient_{i:03d}' for i in range(n_samples)])
        
        # Create perfect predictions from multiple models
        predictions = []
        for _ in range(3):
            df = pd.DataFrame({
                'Participant': objective_df.index,
                'predictions': true_labels,
            })
            # Add probability columns with certainty
            for cls in classes:
                df[cls] = (true_labels == cls).astype(float)
            df.set_index('Participant', inplace=True)
            predictions.append(df)
        
        voted_f1, probed_f1 = calculate_ensemble_metrics(
            predictions,
            objective_df,
            TARGET_COLUMN,
            label_encoder
        )
        
        assert voted_f1 == 1.0
        assert probed_f1 == 1.0


# =============================================================================
# Tests for F1 score computation (utility tests)
# =============================================================================

class TestF1ScoreComputation:
    """Tests for F1 score computation logic."""

    def test_macro_f1_multiclass(self, label_encoder):
        """Verify macro F1 score computation for multiclass."""
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3, 0, 1, 2, 3])  # Perfect
        
        f1 = f1_score(y_true, y_pred, average='macro')
        assert f1 == 1.0

    def test_macro_f1_with_errors(self, label_encoder):
        """Verify macro F1 score decreases with errors."""
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0])  # All predict class 0
        
        f1 = f1_score(y_true, y_pred, average='macro')
        assert f1 < 0.5  # Should be poor


# =============================================================================
# Run tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
