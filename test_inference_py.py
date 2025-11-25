"""Tests for inference pipeline."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import tempfile
import os
import json
from unittest.mock import Mock, patch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from pipelines.inference_pipeline import InferencePipeline, BatchInferencePipeline
from src.models.sklearn_model import TelcoChurnClassifier
from src.preprocessing.preprocessor import TelcoPreprocessor


@pytest.fixture
def sample_model_and_preprocessor():
    """Create sample trained model and preprocessor for testing."""
    # Create synthetic training data
    np.random.seed(42)
    n_samples = 200
    
    # Create realistic feature data
    data = {
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
    
    # Create target variable
    churn_prob = (
        0.1 +  # Base probability
        0.3 * (df['Contract'] == 'Month-to-month') +
        0.2 * (df['tenure'] < 12) +
        0.15 * (df['MonthlyCharges'] > 80)
    )
    y = (np.random.random(n_samples) < churn_prob).astype(int)
    
    # Train preprocessor
    preprocessor = TelcoPreprocessor(
        scaling_method='standard',
        encoding_method='onehot'
    )
    X_processed = preprocessor.fit_transform(df)
    
    # Train model
    classifier = TelcoChurnClassifier(
        model_type='logistic_regression',
        hyperparameter_tuning=False
    )
    classifier.fit(X_processed, y)
    
    return classifier, preprocessor, df


@pytest.fixture
def sample_inference_data():
    """Create sample data for inference testing."""
    np.random.seed(123)
    n_samples = 50
    
    data = {
        'customerID': [f'TEST{i:04d}' for i in range(n_samples)],
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
    
    return pd.DataFrame(data)


class TestInferencePipeline:
    """Test InferencePipeline class."""
    
    def test_init_with_file_paths(self, sample_model_and_preprocessor):
        """Test initialization with file paths."""
        classifier, preprocessor, _ = sample_model_and_preprocessor
        
        # Save model and preprocessor
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.pkl')
            preprocessor_path = os.path.join(tmpdir, 'test_preprocessor.pkl')
            
            classifier.save_model(model_path)
            preprocessor.save_pipeline(preprocessor_path)
            
            # Initialize pipeline
            pipeline = InferencePipeline(
                model_path=model_path,
                preprocessor_path=preprocessor_path
            )
            
            assert pipeline.model is not None
            assert pipeline.preprocessor is not None
            assert pipeline.model_info['source'] == 'local_file'
    
    def test_init_without_paths(self):
        """Test initialization without explicit paths."""
        # Should raise error if no model found
        with pytest.raises((FileNotFoundError, ValueError)):
            InferencePipeline()
    
    @patch('mlflow.sklearn.load_model')
    @patch('mlflow.models.get_model_info')
    def test_init_with_mlflow(self, mock_get_model_info, mock_load_model):
        """Test initialization with MLflow registry."""
        # Mock MLflow responses
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        mock_model_info = Mock()
        mock_model_info.version = '1'
        mock_model_info.current_stage = 'Production'
        mock_model_info.run_id = 'test_run_id'
        mock_get_model_info.return_value = mock_model_info
        
        # Initialize pipeline
        pipeline = InferencePipeline(
            model_name='test_model',
            model_version='1'
        )
        
        assert pipeline.model is not None
        assert pipeline.model_info['source'] == 'mlflow_registry'
        assert pipeline.model_info['version'] == '1'
    
    def test_validate_input_data(self, sample_model_and_preprocessor, sample_inference_data):
        """Test input data validation."""
        classifier, preprocessor, _ = sample_model_and_preprocessor
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.pkl')
            preprocessor_path = os.path.join(tmpdir, 'test_preprocessor.pkl')
            
            classifier.save_model(model_path)
            preprocessor.save_pipeline(preprocessor_path)
            
            pipeline = InferencePipeline(
                model_path=model_path,
                preprocessor_path=preprocessor_path
            )
            
            # Test valid data
            validation_results = pipeline.validate_input_data(sample_inference_data)
            assert validation_results['is_valid']
            assert len(validation_results['issues']) == 0
            
            # Test empty data
            empty_data = pd.DataFrame()
            validation_results = pipeline.validate_input_data(empty_data)
            assert not validation_results['is_valid']
            assert 'empty' in validation_results['issues'][0].lower()
            
            # Test missing columns
            incomplete_data = sample_inference_data[['customerID', 'gender']].copy()
            validation_results = pipeline.validate_input_data(incomplete_data)
            assert not validation_results['is_valid']
            assert 'missing' in validation_results['issues'][0].lower()
    
    def test_predict(self, sample_model_and_preprocessor, sample_inference_data):
        """Test prediction functionality."""
        classifier, preprocessor, _ = sample_model_and_preprocessor
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.pkl')
            preprocessor_path = os.path.join(tmpdir, 'test_preprocessor.pkl')
            
            classifier.save_model(model_path)
            preprocessor.save_pipeline(preprocessor_path)
            
            pipeline = InferencePipeline(
                model_path=model_path,
                preprocessor_path=preprocessor_path
            )
            
            # Make predictions
            results = pipeline.predict(sample_inference_data, return_probabilities=True)
            
            # Validate results structure
            assert 'predictions' in results
            assert 'probabilities' in results
            assert 'confidence_scores' in results
            assert 'metadata' in results
            assert 'statistics' in results
            
            # Validate predictions
            predictions = results['predictions']
            assert len(predictions) == len(sample_inference_data)
            assert all(pred in [0, 1] for pred in predictions)
            
            # Validate probabilities
            probabilities = results['probabilities']
            assert 'class_0' in probabilities
            assert 'class_1' in probabilities
            assert len(probabilities['class_0']) == len(sample_inference_data)
            assert len(probabilities['class_1']) == len(sample_inference_data)
            
            # Validate confidence scores
            confidence_scores = results['confidence_scores']
            assert len(confidence_scores) == len(sample_inference_data)
            assert all(0 <= score <= 1 for score in confidence_scores)
            
            # Validate statistics
            stats = results['statistics']
            assert 'total_predictions' in stats
            assert 'predicted_churn_count' in stats
            assert 'predicted_churn_rate' in stats
            assert stats['total_predictions'] == len(sample_inference_data)
    
    def test_predict_single(self, sample_model_and_preprocessor):
        """Test single customer prediction."""
        classifier, preprocessor, _ = sample_model_and_preprocessor
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.pkl')
            preprocessor_path = os.path.join(tmpdir, 'test_preprocessor.pkl')
            
            classifier.save_model(model_path)
            preprocessor.save_pipeline(preprocessor_path)
            
            pipeline = InferencePipeline(
                model_path=model_path,
                preprocessor_path=preprocessor_path
            )
            
            # Single customer data
            customer_data = {
                'gender': 'Male',
                'SeniorCitizen': 0,
                'Partner': 'Yes',
                'Dependents': 'No',
                'tenure': 12,
                'PhoneService': 'Yes',
                'MultipleLines': 'No',
                'InternetService': 'DSL',
                'OnlineSecurity': 'No',
                'OnlineBackup': 'Yes',
                'DeviceProtection': 'No',
                'TechSupport': 'No',
                'StreamingTV': 'No',
                'StreamingMovies': 'No',
                'Contract': 'Month-to-month',
                'PaperlessBilling': 'Yes',
                'PaymentMethod': 'Electronic check',
                'MonthlyCharges': 50.0,
                'TotalCharges': 600.0
            }
            
            result = pipeline.predict_single(customer_data)
            
            # Validate single prediction result
            assert 'prediction' in result
            assert 'churn_probability' in result
            assert 'confidence' in result
            assert 'prediction_label' in result
            assert 'metadata' in result
            
            assert result['prediction'] in [0, 1]
            assert 0 <= result['churn_probability'] <= 1
            assert 0 <= result['confidence'] <= 1
            assert result['prediction_label'] in ['Churn', 'No Churn']
    
    def test_predict_in_batches(self, sample_model_and_preprocessor, sample_inference_data):
        """Test batch prediction functionality."""
        classifier, preprocessor, _ = sample_model_and_preprocessor
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.pkl')
            preprocessor_path = os.path.join(tmpdir, 'test_preprocessor.pkl')
            
            classifier.save_model(model_path)
            preprocessor.save_pipeline(preprocessor_path)
            
            pipeline = InferencePipeline(
                model_path=model_path,
                preprocessor_path=preprocessor_path
            )
            
            # Test with small batch size
            results = pipeline.predict(
                sample_inference_data, 
                return_probabilities=True,
                batch_size=10
            )
            
            # Should produce same results as non-batched prediction
            assert len(results['predictions']) == len(sample_inference_data)
    
    def test_save_predictions(self, sample_model_and_preprocessor, sample_inference_data):
        """Test saving predictions to file."""
        classifier, preprocessor, _ = sample_model_and_preprocessor
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.pkl')
            preprocessor_path = os.path.join(tmpdir, 'test_preprocessor.pkl')
            
            classifier.save_model(model_path)
            preprocessor.save_pipeline(preprocessor_path)
            
            pipeline = InferencePipeline(
                model_path=model_path,
                preprocessor_path=preprocessor_path
            )
            
            results = pipeline.predict(sample_inference_data, return_probabilities=True)
            
            # Save predictions
            output_file = os.path.join(tmpdir, 'test_predictions.csv')
            saved_path = pipeline.save_predictions(
                results, 
                output_file=output_file,
                include_input_data=True,
                input_data=sample_inference_data
            )
            
            # Validate saved file
            assert os.path.exists(saved_path)
            saved_df = pd.read_csv(saved_path)
            
            # Should include original data + predictions
            assert len(saved_df) == len(sample_inference_data)
            assert 'prediction' in saved_df.columns
            assert 'churn_probability' in saved_df.columns
            assert 'confidence_score' in saved_df.columns
            assert 'prediction_label' in saved_df.columns
            
            # Check metadata file
            metadata_file = saved_path.replace('.csv', '.json')
            assert os.path.exists(metadata_file)
    
    def test_get_model_info(self, sample_model_and_preprocessor):
        """Test model info retrieval."""
        classifier, preprocessor, _ = sample_model_and_preprocessor
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.pkl')
            preprocessor_path = os.path.join(tmpdir, 'test_preprocessor.pkl')
            
            classifier.save_model(model_path)
            preprocessor.save_pipeline(preprocessor_path)
            
            pipeline = InferencePipeline(
                model_path=model_path,
                preprocessor_path=preprocessor_path
            )
            
            model_info = pipeline.get_model_info()
            
            assert 'model_info' in model_info
            assert 'model_type' in model_info
            assert 'preprocessor_loaded' in model_info
            assert model_info['preprocessor_loaded'] is True
    
    def test_explain_prediction(self, sample_model_and_preprocessor):
        """Test prediction explanation functionality."""
        classifier, preprocessor, _ = sample_model_and_preprocessor
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.pkl')
            preprocessor_path = os.path.join(tmpdir, 'test_preprocessor.pkl')
            
            classifier.save_model(model_path)
            preprocessor.save_pipeline(preprocessor_path)
            
            pipeline = InferencePipeline(
                model_path=model_path,
                preprocessor_path=preprocessor_path
            )
            
            # Test explanation for single customer
            customer_data = {
                'gender': 'Male',
                'SeniorCitizen': 0,
                'Partner': 'Yes',
                'Dependents': 'No',
                'tenure': 12,
                'PhoneService': 'Yes',
                'MultipleLines': 'No',
                'InternetService': 'DSL',
                'OnlineSecurity': 'No',
                'OnlineBackup': 'Yes',
                'DeviceProtection': 'No',
                'TechSupport': 'No',
                'StreamingTV': 'No',
                'StreamingMovies': 'No',
                'Contract': 'Month-to-month',
                'PaperlessBilling': 'Yes',
                'PaymentMethod': 'Electronic check',
                'MonthlyCharges': 50.0,
                'TotalCharges': 600.0
            }
            
            explanation = pipeline.explain_prediction(customer_data, top_n_features=5)
            
            assert 'prediction' in explanation
            assert 'churn_probability' in explanation
            assert explanation['prediction'] in [0, 1]
            assert 0 <= explanation['churn_probability'] <= 1
    
    def test_error_handling(self, sample_model_and_preprocessor):
        """Test error handling in various scenarios."""
        classifier, preprocessor, _ = sample_model_and_preprocessor
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.pkl')
            preprocessor_path = os.path.join(tmpdir, 'test_preprocessor.pkl')
            
            classifier.save_model(model_path)
            preprocessor.save_pipeline(preprocessor_path)
            
            pipeline = InferencePipeline(
                model_path=model_path,
                preprocessor_path=preprocessor_path
            )
            
            # Test with invalid data
            invalid_data = pd.DataFrame({'invalid_column': [1, 2, 3]})
            
            with pytest.raises(ValueError):
                pipeline.predict(invalid_data)
            
            # Test prediction without fitted model
            pipeline.model = None
            with pytest.raises(ValueError):
                pipeline.predict(pd.DataFrame({'col': [1]}))


class TestBatchInferencePipeline:
    """Test BatchInferencePipeline class."""
    
    def test_init(self, sample_model_and_preprocessor):
        """Test batch pipeline initialization."""
        classifier, preprocessor, _ = sample_model_and_preprocessor
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.pkl')
            preprocessor_path = os.path.join(tmpdir, 'test_preprocessor.pkl')
            
            classifier.save_model(model_path)
            preprocessor.save_pipeline(preprocessor_path)
            
            inference_pipeline = InferencePipeline(
                model_path=model_path,
                preprocessor_path=preprocessor_path
            )
            
            batch_pipeline = BatchInferencePipeline(inference_pipeline)
            assert batch_pipeline.inference_pipeline is inference_pipeline
    
    def test_process_file(self, sample_model_and_preprocessor, sample_inference_data):
        """Test batch file processing."""
        classifier, preprocessor, _ = sample_model_and_preprocessor
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.pkl')
            preprocessor_path = os.path.join(tmpdir, 'test_preprocessor.pkl')
            input_file = os.path.join(tmpdir, 'input_data.csv')
            output_file = os.path.join(tmpdir, 'output_predictions.csv')
            
            # Save model and preprocessor
            classifier.save_model(model_path)
            preprocessor.save_pipeline(preprocessor_path)
            
            # Save input data
            sample_inference_data.to_csv(input_file, index=False)
            
            # Create pipelines
            inference_pipeline = InferencePipeline(
                model_path=model_path,
                preprocessor_path=preprocessor_path
            )
            batch_pipeline = BatchInferencePipeline(inference_pipeline)
            
            # Process file
            summary = batch_pipeline.process_file(
                input_file=input_file,
                output_file=output_file,
                batch_size=10,
                chunk_size=20
            )
            
            # Validate summary
            assert 'total_processed' in summary
            assert 'total_churn_predicted' in summary
            assert 'churn_rate' in summary
            assert 'output_file' in summary
            
            assert summary['total_processed'] == len(sample_inference_data)
            assert 0 <= summary['churn_rate'] <= 1
            
            # Validate output file
            assert os.path.exists(output_file)
            output_df = pd.read_csv(output_file)
            assert len(output_df) == len(sample_inference_data)
            assert 'prediction' in output_df.columns
            assert 'churn_probability' in output_df.columns


@pytest.mark.integration
class TestInferenceIntegration:
    """Integration tests for inference pipeline."""
    
    def test_end_to_end_inference(self, sample_model_and_preprocessor, sample_inference_data):
        """Test complete end-to-end inference workflow."""
        classifier, preprocessor, training_data = sample_model_and_preprocessor
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save model and preprocessor
            model_path = os.path.join(tmpdir, 'model.pkl')
            preprocessor_path = os.path.join(tmpdir, 'preprocessor.pkl')
            
            classifier.save_model(model_path)
            preprocessor.save_pipeline(preprocessor_path)
            
            # Initialize inference pipeline
            pipeline = InferencePipeline(
                model_path=model_path,
                preprocessor_path=preprocessor_path
            )
            
            # Make predictions
            results = pipeline.predict(sample_inference_data, return_probabilities=True)
            
            # Save predictions
            predictions_file = pipeline.save_predictions(
                results,
                include_input_data=True,
                input_data=sample_inference_data
            )
            
            # Validate saved predictions
            predictions_df = pd.read_csv(predictions_file)
            
            # Should have all original columns plus prediction columns
            expected_pred_cols = ['prediction', 'churn_probability', 'confidence_score', 'prediction_label']
            for col in expected_pred_cols:
                assert col in predictions_df.columns
            
            # All predictions should be valid
            assert predictions_df['prediction'].isin([0, 1]).all()
            assert (predictions_df['churn_probability'] >= 0).all()
            assert (predictions_df['churn_probability'] <= 1).all()
            assert predictions_df['prediction_label'].isin(['Churn', 'No Churn']).all()
    
    def test_inference_consistency(self, sample_model_and_preprocessor, sample_inference_data):
        """Test that inference produces consistent results."""
        classifier, preprocessor, _ = sample_model_and_preprocessor
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'model.pkl')
            preprocessor_path = os.path.join(tmpdir, 'preprocessor.pkl')
            
            classifier.save_model(model_path)
            preprocessor.save_pipeline(preprocessor_path)
            
            # Create two identical pipelines
            pipeline1 = InferencePipeline(
                model_path=model_path,
                preprocessor_path=preprocessor_path
            )
            
            pipeline2 = InferencePipeline(
                model_path=model_path,
                preprocessor_path=preprocessor_path
            )
            
            # Make predictions with both
            results1 = pipeline1.predict(sample_inference_data, return_probabilities=True)
            results2 = pipeline2.predict(sample_inference_data, return_probabilities=True)
            
            # Results should be identical
            np.testing.assert_array_equal(results1['predictions'], results2['predictions'])
            np.testing.assert_allclose(results1['probabilities']['class_1'], 
                                     results2['probabilities']['class_1'], 
                                     rtol=1e-10)
    
    def test_inference_with_csv_file(self, sample_model_and_preprocessor, sample_inference_data):
        """Test inference using CSV file input."""
        classifier, preprocessor, _ = sample_model_and_preprocessor
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save model and preprocessor
            model_path = os.path.join(tmpdir, 'model.pkl')
            preprocessor_path = os.path.join(tmpdir, 'preprocessor.pkl')
            data_file = os.path.join(tmpdir, 'inference_data.csv')
            
            classifier.save_model(model_path)
            preprocessor.save_pipeline(preprocessor_path)
            sample_inference_data.to_csv(data_file, index=False)
            
            # Initialize pipeline
            pipeline = InferencePipeline(
                model_path=model_path,
                preprocessor_path=preprocessor_path
            )
            
            # Make predictions using file path
            results = pipeline.predict(data_file, return_probabilities=True)
            
            assert len(results['predictions']) == len(sample_inference_data)
            assert 'probabilities' in results
            assert 'statistics' in results
    
    def test_batch_vs_single_predictions(self, sample_model_and_preprocessor, sample_inference_data):
        """Test that batch and single predictions produce same results."""
        classifier, preprocessor, _ = sample_model_and_preprocessor
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'model.pkl')
            preprocessor_path = os.path.join(tmpdir, 'preprocessor.pkl')
            
            classifier.save_model(model_path)
            preprocessor.save_pipeline(preprocessor_path)
            
            pipeline = InferencePipeline(
                model_path=model_path,
                preprocessor_path=preprocessor_path
            )
            
            # Batch predictions
            batch_results = pipeline.predict(sample_inference_data, return_probabilities=True)
            
            # Single predictions
            single_predictions = []
            single_probabilities = []
            
            for idx in range(len(sample_inference_data)):
                customer_data = sample_inference_data.iloc[idx].to_dict()
                result = pipeline.predict_single(customer_data)
                single_predictions.append(result['prediction'])
                single_probabilities.append(result['churn_probability'])
            
            # Compare results
            np.testing.assert_array_equal(batch_results['predictions'], single_predictions)
            np.testing.assert_allclose(batch_results['probabilities']['class_1'], 
                                     single_probabilities, 
                                     rtol=1e-10)


@pytest.mark.performance
class TestInferencePerformance:
    """Performance tests for inference pipeline."""
    
    def test_prediction_speed(self, sample_model_and_preprocessor):
        """Test inference speed."""
        import time
        
        classifier, preprocessor, _ = sample_model_and_preprocessor
        
        # Create larger dataset for speed test
        np.random.seed(42)
        n_samples = 1000
        
        large_data = pd.DataFrame({
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
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'model.pkl')
            preprocessor_path = os.path.join(tmpdir, 'preprocessor.pkl')
            
            classifier.save_model(model_path)
            preprocessor.save_pipeline(preprocessor_path)
            
            pipeline = InferencePipeline(
                model_path=model_path,
                preprocessor_path=preprocessor_path
            )
            
            # Time the predictions
            start_time = time.time()
            results = pipeline.predict(large_data, return_probabilities=True)
            prediction_time = time.time() - start_time
            
            # Should process reasonable number of records per second
            records_per_second = n_samples / prediction_time
            
            # This threshold can be adjusted based on hardware
            assert records_per_second > 100  # At least 100 records per second
            
            # Validate results
            assert len(results['predictions']) == n_samples


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 