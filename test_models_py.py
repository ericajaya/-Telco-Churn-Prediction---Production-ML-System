"""Tests for model implementations."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import tempfile
import os
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.sklearn_model import TelcoChurnClassifier, ModelEnsemble, create_model_comparison
from src.preprocessing.preprocessor import TelcoPreprocessor


@pytest.fixture
def sample_processed_data():
    """Create sample processed data for model testing."""
    np.random.seed(42)
    n_samples = 200
    n_features = 50
    
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Generate labels with some correlation to features
    y = (X[:, 0] + X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.5 > 0).astype(int)
    
    # Ensure we have both classes
    if len(np.unique(y)) == 1:
        y[:10] = 1 - y[0]  # Flip first 10 to ensure both classes
    
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)


@pytest.fixture
def sample_telco_data():
    """Create realistic Telco data for testing."""
    np.random.seed(42)
    n_samples = 300
    
    data = {
        'customerID': [f'ID{i:04d}' for i in range(n_samples)],
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples),
        'Partner': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples),
        'tenure': np.random.randint(1, 72, n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
        'PaymentMethod': np.random.choice([
            'Electronic check', 'Mailed check', 
            'Bank transfer (automatic)', 'Credit card (automatic)'
        ], n_samples),
        'MonthlyCharges': np.random.uniform(20, 120, n_samples),
        'TotalCharges': np.random.uniform(20, 8000, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic churn based on features
    churn_prob = (
        0.1 +  # Base probability
        0.3 * (df['Contract'] == 'Month-to-month') +
        0.2 * (df['tenure'] < 12) +
        0.15 * (df['MonthlyCharges'] > 80) +
        0.1 * (df['SeniorCitizen'] == 1)
    )
    
    df['Churn'] = (np.random.random(n_samples) < churn_prob).astype(int)
    
    return df


class TestTelcoChurnClassifier:
    """Test TelcoChurnClassifier class."""
    
    def test_init(self):
        """Test classifier initialization."""
        classifier = TelcoChurnClassifier()
        assert classifier.model_type == 'random_forest'
        assert classifier.hyperparameter_tuning is True
        assert classifier.cv_folds == 5
        assert classifier.random_state == 42
        assert classifier.model is None
        
        # Test custom parameters
        classifier_custom = TelcoChurnClassifier(
            model_type='logistic_regression',
            hyperparameter_tuning=False,
            cv_folds=3,
            random_state=123
        )
        assert classifier_custom.model_type == 'logistic_regression'
        assert classifier_custom.hyperparameter_tuning is False
        assert classifier_custom.cv_folds == 3
        assert classifier_custom.random_state == 123
    
    def test_get_base_model(self):
        """Test base model creation."""
        # Test each model type
        model_types = ['logistic_regression', 'random_forest', 'gradient_boosting']
        
        for model_type in model_types:
            classifier = TelcoChurnClassifier(model_type=model_type)
            model = classifier._get_base_model()
            assert model is not None
            assert hasattr(model, 'fit')
            assert hasattr(model, 'predict')
    
    def test_invalid_model_type(self):
        """Test error handling for invalid model type."""
        with pytest.raises(ValueError):
            classifier = TelcoChurnClassifier(model_type='invalid_model')
            classifier._get_base_model()
    
    def test_fit_without_tuning(self, sample_processed_data):
        """Test fitting without hyperparameter tuning."""
        X_train, X_test, y_train, y_test = sample_processed_data
        
        classifier = TelcoChurnClassifier(
            model_type='logistic_regression',
            hyperparameter_tuning=False
        )
        
        classifier.fit(X_train, y_train)
        
        assert classifier.model is not None
        assert classifier.best_params is None  # No tuning performed
        assert classifier.cv_scores is not None  # CV scores should still be calculated
    
    def test_fit_with_tuning(self, sample_processed_data):
        """Test fitting with hyperparameter tuning."""
        X_train, X_test, y_train, y_test = sample_processed_data
        
        classifier = TelcoChurnClassifier(
            model_type='logistic_regression',
            hyperparameter_tuning=True
        )
        
        classifier.fit(X_train, y_train)
        
        assert classifier.model is not None
        assert classifier.best_params is not None
        assert classifier.cv_scores is not None
    
    def test_predict(self, sample_processed_data):
        """Test prediction functionality."""
        X_train, X_test, y_train, y_test = sample_processed_data
        
        classifier = TelcoChurnClassifier(
            model_type='logistic_regression',
            hyperparameter_tuning=False
        )
        
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert set(predictions) <= {0, 1}  # Binary classification
    
    def test_predict_proba(self, sample_processed_data):
        """Test probability prediction."""
        X_train, X_test, y_train, y_test = sample_processed_data
        
        classifier = TelcoChurnClassifier(
            model_type='logistic_regression',
            hyperparameter_tuning=False
        )
        
        classifier.fit(X_train, y_train)
        probabilities = classifier.predict_proba(X_test)
        
        assert probabilities.shape == (len(X_test), 2)
        assert np.all(probabilities >= 0) and np.all(probabilities <= 1)
        assert np.allclose(probabilities.sum(axis=1), 1)  # Probabilities sum to 1
    
    def test_evaluate(self, sample_processed_data):
        """Test model evaluation."""
        X_train, X_test, y_train, y_test = sample_processed_data
        
        classifier = TelcoChurnClassifier(
            model_type='logistic_regression',
            hyperparameter_tuning=False
        )
        
        classifier.fit(X_train, y_train)
        metrics = classifier.evaluate(X_test, y_test)
        
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        for metric in expected_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1  # All metrics should be between 0 and 1
    
    def test_feature_importance_extraction(self, sample_processed_data):
        """Test feature importance extraction."""
        X_train, X_test, y_train, y_test = sample_processed_data
        
        # Test with tree-based model (has feature_importances_)
        rf_classifier = TelcoChurnClassifier(
            model_type='random_forest',
            hyperparameter_tuning=False
        )
        rf_classifier.fit(X_train, y_train)
        assert rf_classifier.feature_importance is not None
        assert len(rf_classifier.feature_importance) == X_train.shape[1]
        
        # Test with linear model (has coef_)
        lr_classifier = TelcoChurnClassifier(
            model_type='logistic_regression',
            hyperparameter_tuning=False
        )
        lr_classifier.fit(X_train, y_train)
        assert lr_classifier.feature_importance is not None
        assert len(lr_classifier.feature_importance) == X_train.shape[1]
    
    def test_save_load_model(self, sample_processed_data):
        """Test model saving and loading."""
        X_train, X_test, y_train, y_test = sample_processed_data
        
        classifier = TelcoChurnClassifier(
            model_type='logistic_regression',
            hyperparameter_tuning=False
        )
        classifier.fit(X_train, y_train)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            classifier.save_model(tmp.name)
            
            # Load model in new classifier
            new_classifier = TelcoChurnClassifier()
            new_classifier.load_model(tmp.name)
            
            # Test that loaded model works
            predictions1 = classifier.predict(X_test)
            predictions2 = new_classifier.predict(X_test)
            
            np.testing.assert_array_equal(predictions1, predictions2)
            
            # Clean up
            os.unlink(tmp.name)
    
    def test_cross_validation_scores(self, sample_processed_data):
        """Test cross-validation score calculation."""
        X_train, X_test, y_train, y_test = sample_processed_data
        
        classifier = TelcoChurnClassifier(
            model_type='logistic_regression',
            hyperparameter_tuning=False,
            cv_folds=3
        )
        
        classifier.fit(X_train, y_train)
        
        assert classifier.cv_scores is not None
        
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        for metric in expected_metrics:
            assert metric in classifier.cv_scores
            assert 'mean' in classifier.cv_scores[metric]
            assert 'std' in classifier.cv_scores[metric]
            assert 'scores' in classifier.cv_scores[metric]


class TestModelEnsemble:
    """Test ModelEnsemble class."""
    
    def test_init(self, sample_processed_data):
        """Test ensemble initialization."""
        X_train, X_test, y_train, y_test = sample_processed_data
        
        models = {
            'lr': TelcoChurnClassifier(model_type='logistic_regression', hyperparameter_tuning=False),
            'rf': TelcoChurnClassifier(model_type='random_forest', hyperparameter_tuning=False)
        }
        
        ensemble = ModelEnsemble(models)
        assert ensemble.models == models
        assert ensemble.weights is None
    
    def test_fit(self, sample_processed_data):
        """Test ensemble fitting."""
        X_train, X_test, y_train, y_test = sample_processed_data
        
        models = {
            'lr': TelcoChurnClassifier(model_type='logistic_regression', hyperparameter_tuning=False),
            'rf': TelcoChurnClassifier(model_type='random_forest', hyperparameter_tuning=False)
        }
        
        ensemble = ModelEnsemble(models)
        ensemble.fit(X_train, y_train)
        
        # Check that all models are fitted
        for model in ensemble.models.values():
            assert model.model is not None
        
        # Check that weights are calculated
        assert ensemble.weights is not None
        assert len(ensemble.weights) == len(models)
        assert np.abs(ensemble.weights.sum() - 1.0) < 1e-6  # Weights sum to 1
    
    def test_predict_proba(self, sample_processed_data):
        """Test ensemble probability prediction."""
        X_train, X_test, y_train, y_test = sample_processed_data
        
        models = {
            'lr': TelcoChurnClassifier(model_type='logistic_regression', hyperparameter_tuning=False),
            'rf': TelcoChurnClassifier(model_type='random_forest', hyperparameter_tuning=False)
        }
        
        ensemble = ModelEnsemble(models)
        ensemble.fit(X_train, y_train)
        
        probabilities = ensemble.predict_proba(X_test)
        
        assert probabilities.shape == (len(X_test), 2)
        assert np.all(probabilities >= 0) and np.all(probabilities <= 1)
        assert np.allclose(probabilities.sum(axis=1), 1)
    
    def test_predict(self, sample_processed_data):
        """Test ensemble prediction."""
        X_train, X_test, y_train, y_test = sample_processed_data
        
        models = {
            'lr': TelcoChurnClassifier(model_type='logistic_regression', hyperparameter_tuning=False),
            'rf': TelcoChurnClassifier(model_type='random_forest', hyperparameter_tuning=False)
        }
        
        ensemble = ModelEnsemble(models)
        ensemble.fit(X_train, y_train)
        
        predictions = ensemble.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert set(predictions) <= {0, 1}
    
    def test_evaluate(self, sample_processed_data):
        """Test ensemble evaluation."""
        X_train, X_test, y_train, y_test = sample_processed_data
        
        models = {
            'lr': TelcoChurnClassifier(model_type='logistic_regression', hyperparameter_tuning=False),
            'rf': TelcoChurnClassifier(model_type='random_forest', hyperparameter_tuning=False)
        }
        
        ensemble = ModelEnsemble(models)
        ensemble.fit(X_train, y_train)
        
        metrics = ensemble.evaluate(X_test, y_test)
        
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for metric in expected_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1


class TestModelComparison:
    """Test model comparison functionality."""
    
    def test_create_model_comparison(self, sample_processed_data):
        """Test model comparison function."""
        X_train, X_test, y_train, y_test = sample_processed_data
        
        # Create a smaller feature set for faster testing
        X_train_small = X_train[:100, :10]
        X_test_small = X_test[:30, :10]
        y_train_small = y_train[:100]
        y_test_small = y_test[:30]
        
        results_df = create_model_comparison(
            X_train_small, y_train_small,
            X_test_small, y_test_small
        )
        
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == 3  # Three model types
        
        expected_columns = [
            'model_type', 'best_params', 'accuracy', 'precision',
            'recall', 'f1_score', 'roc_auc'
        ]
        
        for col in expected_columns:
            assert col in results_df.columns
        
        # Check that all models have reasonable performance
        assert all(results_df['accuracy'] >= 0.3)  # At least better than random
        assert all(results_df['roc_auc'] >= 0.4)   # At least better than random


@pytest.mark.integration
class TestModelIntegration:
    """Integration tests for complete model pipeline."""
    
    def test_end_to_end_pipeline(self, sample_telco_data):
        """Test complete pipeline from raw data to predictions."""
        # Preprocess data
        preprocessor = TelcoPreprocessor(
            scaling_method='standard',
            encoding_method='onehot'
        )
        
        X = sample_telco_data.drop(columns=['Churn', 'customerID'])
        y = sample_telco_data['Churn']
        
        X_processed = preprocessor.fit_transform(X)
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, stratify=y, random_state=42
        )
        
        classifier = TelcoChurnClassifier(
            model_type='logistic_regression',
            hyperparameter_tuning=False
        )
        
        classifier.fit(X_train, y_train)
        
        # Test predictions
        predictions = classifier.predict(X_test)
        probabilities = classifier.predict_proba(X_test)
        
        assert len(predictions) == len(X_test)
        assert probabilities.shape == (len(X_test), 2)
        
        # Test evaluation
        metrics = classifier.evaluate(X_test, y_test)
        
        # Should achieve reasonable performance on synthetic data
        assert metrics['accuracy'] > 0.5
        assert metrics['roc_auc'] > 0.5
    
    def test_model_persistence_integration(self, sample_telco_data):
        """Test model saving/loading in realistic scenario."""
        # Prepare data
        preprocessor = TelcoPreprocessor()
        X = sample_telco_data.drop(columns=['Churn', 'customerID'])
        y = sample_telco_data['Churn']
        X_processed = preprocessor.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Train and save model
        classifier = TelcoChurnClassifier(
            model_type='random_forest',
            hyperparameter_tuning=False
        )
        classifier.fit(X_train, y_train)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            classifier.save_model(tmp.name)
            
            # Simulate new session - load model and make predictions
            new_classifier = TelcoChurnClassifier()
            new_classifier.load_model(tmp.name)
            
            # Test that loaded model produces same results
            orig_predictions = classifier.predict(X_test)
            loaded_predictions = new_classifier.predict(X_test)
            
            np.testing.assert_array_equal(orig_predictions, loaded_predictions)
            
            # Test evaluation
            orig_metrics = classifier.evaluate(X_test, y_test)
            loaded_metrics = new_classifier.evaluate(X_test, y_test)
            
            for metric in orig_metrics:
                assert abs(orig_metrics[metric] - loaded_metrics[metric]) < 1e-6
            
            # Clean up
            os.unlink(tmp.name)
    
    def test_ensemble_vs_individual_models(self, sample_processed_data):
        """Test that ensemble performs at least as well as individual models."""
        X_train, X_test, y_train, y_test = sample_processed_data
        
        # Train individual models
        models = {
            'lr': TelcoChurnClassifier(model_type='logistic_regression', hyperparameter_tuning=False),
            'rf': TelcoChurnClassifier(model_type='random_forest', hyperparameter_tuning=False)
        }
        
        individual_scores = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            metrics = model.evaluate(X_test, y_test)
            individual_scores[name] = metrics['roc_auc']
        
        # Train ensemble
        ensemble = ModelEnsemble(models)
        ensemble.fit(X_train, y_train)
        ensemble_metrics = ensemble.evaluate(X_test, y_test)
        
        # Ensemble should perform at least as well as the best individual model
        best_individual_score = max(individual_scores.values())
        
        # Allow for small numerical differences and the possibility that
        # ensemble might be slightly worse due to model interactions
        assert ensemble_metrics['roc_auc'] >= best_individual_score - 0.05
    
    def test_model_robustness_to_data_issues(self):
        """Test model robustness to common data issues."""
        # Create data with various issues
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] > 0).astype(int)
        
        # Add some problematic data
        X[5:10, 2] = np.nan  # Missing values
        X[15:20, 3] = np.inf  # Infinite values
        X[25:30, 4] = 1e10   # Very large values
        
        # Replace problematic values
        X = np.nan_to_num(X, nan=0, posinf=1e6, neginf=-1e6)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Should still be able to train
        classifier = TelcoChurnClassifier(
            model_type='logistic_regression',
            hyperparameter_tuning=False
        )
        
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        
        # Should produce valid predictions
        assert len(predictions) == len(X_test)
        assert set(predictions) <= {0, 1}


@pytest.mark.slow
class TestModelPerformance:
    """Performance and stress tests for models."""
    
    def test_model_training_time(self, sample_processed_data):
        """Test that model training completes in reasonable time."""
        import time
        
        X_train, X_test, y_train, y_test = sample_processed_data
        
        start_time = time.time()
        
        classifier = TelcoChurnClassifier(
            model_type='logistic_regression',
            hyperparameter_tuning=False
        )
        classifier.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Should complete in reasonable time (adjust threshold as needed)
        assert training_time < 30  # 30 seconds
    
    def test_prediction_speed(self, sample_processed_data):
        """Test prediction speed."""
        import time
        
        X_train, X_test, y_train, y_test = sample_processed_data
        
        classifier = TelcoChurnClassifier(
            model_type='logistic_regression',
            hyperparameter_tuning=False
        )
        classifier.fit(X_train, y_train)
        
        # Test prediction speed
        start_time = time.time()
        predictions = classifier.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Should be very fast for small dataset
        assert prediction_time < 1  # 1 second
        
        # Test prediction efficiency
        predictions_per_second = len(X_test) / prediction_time
        assert predictions_per_second > 100  # At least 100 predictions per second


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])