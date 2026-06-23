"""
Tests for prepare_data and train_model functions.

These tests verify the data preparation and model training components
of the exportable classifier pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# Import functions to test
import sys
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

try:
    from create_exportable_package import (
        prepare_data,
        train_model,
        punch,
        seed_everything,
        prepare_hyperparameters,
        DEFAULT_HYPERPARAMETERS,
        XGBOOST_BASE_PARAMS,
    )
    _IMPORTS_AVAILABLE = True
except ImportError as e:
    _IMPORTS_AVAILABLE = False
    _IMPORT_ERROR = str(e)


# Skip all tests if imports fail
pytestmark = pytest.mark.skipif(
    not _IMPORTS_AVAILABLE,
    reason=f"Could not import required modules: {_IMPORT_ERROR if not _IMPORTS_AVAILABLE else ''}"
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_classification_data():
    """Create sample data for classification tests."""
    np.random.seed(42)
    n_samples = 200
    
    # Features
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
        'cat#a': np.random.choice([0.0, 1.0], n_samples),
        'cat#b': np.random.choice([0.0, 1.0], n_samples),
    })
    
    # Target - ensure all classes are present and balanced enough for stratification
    y = pd.DataFrame({
        'clustername': np.random.choice(
            ['healthier', 'rEF', 'pEF1', 'pEF2'], 
            n_samples,
            p=[0.25, 0.25, 0.25, 0.25]
        )
    })
    
    return X, y


@pytest.fixture
def label_encoder():
    """Create label encoder for cluster names."""
    le = LabelEncoder()
    le.fit(['healthier', 'pEF1', 'pEF2', 'rEF'])
    return le


@pytest.fixture
def sample_hyperparameters():
    """Sample hyperparameters for testing."""
    return {
        'colsample_bylevel': 0.4,
        'colsample_bynode': 0.6,
        'colsample_bytree': 0.74,
        'gamma': 1.88,
        'learning_rate': 0.03,
        'max_depth': 5,
        'reg_alpha': 2.0,
        'reg_lambda': 2.0,
        'n_estimators': 100,  # Reduced for testing
        'split': 0.33,
    }


# =============================================================================
# Tests for seed_everything
# =============================================================================

class TestSeedEverything:
    """Tests for seed_everything function."""

    def test_returns_generator(self):
        """Verify function returns numpy Generator."""
        rng = seed_everything(42)
        assert isinstance(rng, np.random.Generator)

    def test_reproducibility(self):
        """Verify same seed produces same random numbers."""
        rng1 = seed_everything(42)
        values1 = [rng1.random() for _ in range(5)]
        
        rng2 = seed_everything(42)
        values2 = [rng2.random() for _ in range(5)]
        
        assert values1 == values2

    def test_different_seeds(self):
        """Verify different seeds produce different results."""
        rng1 = seed_everything(42)
        values1 = [rng1.random() for _ in range(5)]
        
        rng2 = seed_everything(123)
        values2 = [rng2.random() for _ in range(5)]
        
        assert values1 != values2


# =============================================================================
# Tests for punch function
# =============================================================================

class TestPunchFunction:
    """Tests for punch (missing value introduction) function."""

    def test_introduces_missing_values(self, sample_classification_data):
        """Verify punch introduces NaN values."""
        X, _ = sample_classification_data
        rng = seed_everything(42)
        
        X_punched = punch(X.copy(), ratio_punch_col=0.3, ratio_punch_row=0.3, rng=rng)
        
        # Should have some NaN values
        assert X_punched.isna().sum().sum() > 0

    def test_preserves_shape(self, sample_classification_data):
        """Verify punch preserves DataFrame shape."""
        X, _ = sample_classification_data
        rng = seed_everything(42)
        
        X_punched = punch(X.copy(), ratio_punch_col=0.3, ratio_punch_row=0.3, rng=rng)
        
        assert X_punched.shape == X.shape

    def test_zero_ratio_no_changes(self, sample_classification_data):
        """Verify zero punch ratio makes no changes."""
        X, _ = sample_classification_data
        rng = seed_everything(42)
        
        X_punched = punch(X.copy(), ratio_punch_col=0.0, ratio_punch_row=0.0, rng=rng)
        
        # Should have no NaN values introduced
        original_nan_count = X.isna().sum().sum()
        punched_nan_count = X_punched.isna().sum().sum()
        assert punched_nan_count == original_nan_count


# =============================================================================
# Tests for prepare_data
# =============================================================================

class TestPrepareData:
    """Tests for prepare_data function."""

    def test_returns_correct_structure(self, sample_classification_data, label_encoder):
        """Verify prepare_data returns expected tuple structure."""
        X, y = sample_classification_data
        
        result = prepare_data(
            X, y, split=0.3, target_col='clustername',
            label_encoder=label_encoder
        )
        
        assert len(result) == 6
        X_train, X_test, Y_train, Y_test, y_train, y_test = result
        
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(Y_train, pd.DataFrame)
        assert isinstance(Y_test, pd.DataFrame)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_test, np.ndarray)

    def test_split_proportions(self, sample_classification_data, label_encoder):
        """Verify train/test split proportions."""
        X, y = sample_classification_data
        
        X_train, X_test, _, _, _, _ = prepare_data(
            X, y, split=0.3, target_col='clustername',
            label_encoder=label_encoder
        )
        
        total = len(X)
        test_ratio = len(X_test) / total
        
        # Allow 5% tolerance
        assert abs(test_ratio - 0.3) < 0.05

    def test_label_encoding(self, sample_classification_data, label_encoder):
        """Verify labels are encoded correctly."""
        X, y = sample_classification_data
        
        _, _, _, _, y_train, y_test = prepare_data(
            X, y, split=0.3, target_col='clustername',
            label_encoder=label_encoder
        )
        
        # Encoded values should be integers 0-3
        assert set(y_train).issubset({0, 1, 2, 3})
        assert set(y_test).issubset({0, 1, 2, 3})

    def test_add_random_feature(self, sample_classification_data, label_encoder):
        """Verify random feature is added when requested."""
        X, y = sample_classification_data
        
        X_train, X_test, _, _, _, _ = prepare_data(
            X, y, split=0.3, target_col='clustername',
            label_encoder=label_encoder, add_random=True
        )
        
        assert 'random' in X_train.columns
        assert 'random' in X_test.columns

    def test_column_alignment(self, sample_classification_data, label_encoder):
        """Verify train and test have same columns."""
        X, y = sample_classification_data
        
        X_train, X_test, _, _, _, _ = prepare_data(
            X, y, split=0.3, target_col='clustername',
            label_encoder=label_encoder
        )
        
        assert list(X_train.columns) == list(X_test.columns)

    def test_with_punching(self, sample_classification_data, label_encoder):
        """Verify punching introduces missing values in training set."""
        X, y = sample_classification_data
        rng = seed_everything(42)
        
        X_train, X_test, _, _, _, _ = prepare_data(
            X, y, split=0.3, target_col='clustername',
            label_encoder=label_encoder,
            to_punch=True, ratio_punch_col=0.3, ratio_punch_row=0.3,
            rng=rng
        )
        
        # Training set should have NaN values from punching
        assert X_train.isna().sum().sum() > 0


# =============================================================================
# Tests for train_model
# =============================================================================

class TestTrainModel:
    """Tests for train_model function."""

    def test_returns_classifier(self, sample_classification_data, 
                                 label_encoder, sample_hyperparameters):
        """Verify train_model returns XGBClassifier."""
        X, y = sample_classification_data
        
        X_train, X_test, _, _, y_train, y_test = prepare_data(
            X, y, split=0.3, target_col='clustername',
            label_encoder=label_encoder
        )
        
        # Fill NaN for XGBoost
        X_train_filled = X_train.fillna(0)
        X_test_filled = X_test.fillna(0)
        
        model = train_model(
            X_train_filled, y_train,
            X_test_filled, y_test,
            sample_hyperparameters
        )
        
        from xgboost import XGBClassifier
        assert isinstance(model, XGBClassifier)

    def test_model_can_predict(self, sample_classification_data,
                                label_encoder, sample_hyperparameters):
        """Verify trained model can make predictions."""
        X, y = sample_classification_data
        
        X_train, X_test, _, _, y_train, y_test = prepare_data(
            X, y, split=0.3, target_col='clustername',
            label_encoder=label_encoder
        )
        
        X_train_filled = X_train.fillna(0)
        X_test_filled = X_test.fillna(0)
        
        model = train_model(
            X_train_filled, y_train,
            X_test_filled, y_test,
            sample_hyperparameters
        )
        
        predictions = model.predict(X_test_filled)
        
        assert len(predictions) == len(X_test)
        assert set(predictions).issubset({0, 1, 2, 3})

    def test_model_has_feature_importances(self, sample_classification_data,
                                            label_encoder, sample_hyperparameters):
        """Verify model has feature importances."""
        X, y = sample_classification_data
        
        X_train, X_test, _, _, y_train, y_test = prepare_data(
            X, y, split=0.3, target_col='clustername',
            label_encoder=label_encoder
        )
        
        X_train_filled = X_train.fillna(0)
        X_test_filled = X_test.fillna(0)
        
        model = train_model(
            X_train_filled, y_train,
            X_test_filled, y_test,
            sample_hyperparameters
        )
        
        importances = model.feature_importances_
        assert len(importances) == len(X_train.columns)


# =============================================================================
# Tests for prepare_hyperparameters
# =============================================================================

class TestPrepareHyperparameters:
    """Tests for prepare_hyperparameters function."""

    def test_returns_tuple(self):
        """Verify function returns a tuple."""
        result = prepare_hyperparameters(
            XGBOOST_BASE_PARAMS.copy(),
            DEFAULT_HYPERPARAMETERS.copy()
        )
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_extracts_split_ratio(self):
        """Verify split is extracted from hyperparameters."""
        hyperp, split, _ = prepare_hyperparameters(
            XGBOOST_BASE_PARAMS.copy(),
            {'split': 0.25, 'learning_rate': 0.1}
        )
        
        assert split == 0.25
        assert 'split' not in hyperp

    def test_merges_with_base_params(self):
        """Verify base XGBoost params are merged."""
        hyperp, _, _ = prepare_hyperparameters(
            XGBOOST_BASE_PARAMS.copy(),
            DEFAULT_HYPERPARAMETERS.copy()
        )
        
        # Check base params are present
        assert 'objective' in hyperp
        assert 'eval_metric' in hyperp
        # Check optimized params are present
        assert 'learning_rate' in hyperp

    def test_converts_int_params(self):
        """Verify integer parameters are converted correctly."""
        hyperp, _, _ = prepare_hyperparameters(
            XGBOOST_BASE_PARAMS.copy(),
            {'max_depth': 5.0, 'n_estimators': 100.0}
        )
        
        assert isinstance(hyperp['max_depth'], int)
        assert isinstance(hyperp['n_estimators'], int)


# =============================================================================
# Run tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
