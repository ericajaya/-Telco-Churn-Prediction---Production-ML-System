"""Inference pipeline for Telco Churn Prediction."""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import json
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_loader import DataLoader
from src.preprocessing.preprocessor import TelcoPreprocessor
from src.models.sklearn_model import TelcoChurnClassifier
from config.config import (
    MODEL_DIR, PREDICTIONS_PATH, MLFLOW_TRACKING_URI,
    create_directories, LOGS_DIR
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'inference_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class InferencePipeline:
    """Production inference pipeline for Telco Churn Prediction."""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 preprocessor_path: Optional[str] = None,
                 model_name: Optional[str] = None,
                 model_version: Optional[str] = None):
        """
        Initialize inference pipeline.
        
        Args:
            model_path: Path to saved model file
            preprocessor_path: Path to saved preprocessor
            model_name: MLflow model registry name (alternative to model_path)
            model_version: MLflow model version (used with model_name)
        """
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model_name = model_name
        self.model_version = model_version
        
        self.model = None
        self.preprocessor = None
        self.model_info = {}
        
        # Load model and preprocessor
        self._load_model_and_preprocessor()
    
    def _load_model_and_preprocessor(self):
        """Load model and preprocessor from files or MLflow."""
        logger.info("Loading model and preprocessor...")
        
        try:
            # Load from MLflow registry if model_name is provided
            if self.model_name:
                self._load_from_mlflow()
            else:
                self._load_from_files()
            
            logger.info("Model and preprocessor loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model/preprocessor: {e}")
            raise
    
    def _load_from_mlflow(self):
        """Load model from MLflow registry."""
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Build model URI
        if self.model_version:
            model_uri = f"models:/{self.model_name}/{self.model_version}"
        else:
            model_uri = f"models:/{self.model_name}/latest"
        
        logger.info(f"Loading model from MLflow: {model_uri}")
        
        # Load MLflow model
        self.model = mlflow.sklearn.load_model(model_uri)
        
        # Get model info
        model_info = mlflow.models.get_model_info(model_uri)
        self.model_info = {
            'name': self.model_name,
            'version': model_info.version,
            'stage': model_info.current_stage,
            'run_id': model_info.run_id,
            'source': 'mlflow_registry'
        }
        
        # Load preprocessor from the same run if available
        try:
            run_info = mlflow.get_run(model_info.run_id)
            artifacts = mlflow.list_artifacts(model_info.run_id)
            
            # Look for preprocessor artifact
            for artifact in artifacts:
                if 'preprocessor' in artifact.path.lower():
                    preprocessor_path = mlflow.artifacts.download_artifacts(
                        artifact_uri=f"runs:/{model_info.run_id}/{artifact.path}"
                    )
                    
                    self.preprocessor = TelcoPreprocessor()
                    self.preprocessor.load_pipeline(preprocessor_path)
                    break
            
        except Exception as e:
            logger.warning(f"Could not load preprocessor from MLflow: {e}")
    
    def _load_from_files(self):
        """Load model and preprocessor from local files."""
        # Determine paths
        if self.model_path is None:
            # Try to find the latest model
            model_files = list(MODEL_DIR.glob("*_model.pkl"))
            if not model_files:
                raise FileNotFoundError(f"No model files found in {MODEL_DIR}")
            
            # Sort by modification time, get latest
            self.model_path = max(model_files, key=os.path.getmtime)
            logger.info(f"Using latest model: {self.model_path}")
        
        if self.preprocessor_path is None:
            self.preprocessor_path = MODEL_DIR / "preprocessor.pkl"
        
        # Load model
        logger.info(f"Loading model from {self.model_path}")
        classifier = TelcoChurnClassifier()
        classifier.load_model(str(self.model_path))
        self.model = classifier.model
        
        self.model_info = {
            'path': str(self.model_path),
            'source': 'local_file'
        }
        
        # Load preprocessor
        if Path(self.preprocessor_path).exists():
            logger.info(f"Loading preprocessor from {self.preprocessor_path}")
            self.preprocessor = TelcoPreprocessor()
            self.preprocessor.load_pipeline(str(self.preprocessor_path))
        else:
            logger.warning(f"Preprocessor not found at {self.preprocessor_path}")
            raise FileNotFoundError(f"Preprocessor not found: {self.preprocessor_path}")
    
    def validate_input_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate input data for inference.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Validation results
        """
        validation_results = {
            'is_valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Check basic requirements
        if data.empty:
            validation_results['is_valid'] = False
            validation_results['issues'].append("Input data is empty")
            return validation_results
        
        # Check required columns (excluding target)
        from config.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
        
        required_columns = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
        missing_columns = set(required_columns) - set(data.columns)
        
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['issues'].append(
                f"Missing required columns: {missing_columns}"
            )
        
        # Check data types and ranges
        for col in NUMERICAL_FEATURES:
            if col in data.columns:
                try:
                    pd.to_numeric(data[col], errors='raise')
                except ValueError:
                    validation_results['warnings'].append(
                        f"Column {col} contains non-numeric values"
                    )
        
        # Check for excessive missing values
        missing_threshold = 0.8
        for col in data.columns:
            missing_ratio = data[col].isnull().sum() / len(data)
            if missing_ratio > missing_threshold:
                validation_results['warnings'].append(
                    f"Column {col} has {missing_ratio:.2%} missing values"
                )
        
        return validation_results
    
    def predict(self, 
                data: Union[pd.DataFrame, str],
                return_probabilities: bool = True,
                batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Make predictions on input data.
        
        Args:
            data: Input data (DataFrame or file path)
            return_probabilities: Whether to return prediction probabilities
            batch_size: Batch size for large datasets
            
        Returns:
            Predictions and metadata
        """
        logger.info("Starting inference...")
        
        # Load data if path provided
        if isinstance(data, str):
            logger.info(f"Loading data from {data}")
            data = pd.read_csv(data)
        
        # Validate input data
        validation_results = self.validate_input_data(data)
        
        if not validation_results['is_valid']:
            raise ValueError(f"Input validation failed: {validation_results['issues']}")
        
        if validation_results['warnings']:
            for warning in validation_results['warnings']:
                logger.warning(warning)
        
        # Store original data info
        original_shape = data.shape
        original_columns = list(data.columns)
        
        try:
            # Preprocess data
            logger.info("Preprocessing data...")
            if self.preprocessor is None:
                raise ValueError("Preprocessor not loaded")
            
            X_processed = self.preprocessor.transform(data)
            logger.info(f"Data preprocessed: {original_shape} -> {X_processed.shape}")
            
            # Make predictions in batches if specified
            if batch_size and len(X_processed) > batch_size:
                predictions, probabilities = self._predict_in_batches(
                    X_processed, batch_size, return_probabilities
                )
            else:
                # Single batch prediction
                predictions = self.model.predict(X_processed)
                probabilities = (
                    self.model.predict_proba(X_processed) 
                    if return_probabilities else None
                )
            
            # Prepare results
            results = {
                'predictions': predictions.tolist(),
                'metadata': {
                    'model_info': self.model_info,
                    'input_shape': original_shape,
                    'processed_shape': X_processed.shape,
                    'prediction_timestamp': datetime.now().isoformat(),
                    'validation_results': validation_results
                }
            }
            
            if return_probabilities and probabilities is not None:
                results['probabilities'] = {
                    'class_0': probabilities[:, 0].tolist(),
                    'class_1': probabilities[:, 1].tolist()
                }
                results['confidence_scores'] = np.max(probabilities, axis=1).tolist()
            
            # Add prediction statistics
            results['statistics'] = {
                'total_predictions': len(predictions),
                'predicted_churn_count': int(np.sum(predictions)),
                'predicted_churn_rate': float(np.mean(predictions)),
                'high_confidence_predictions': (
                    int(np.sum(np.max(probabilities, axis=1) > 0.8))
                    if probabilities is not None else None
                )
            }
            
            logger.info(f"Inference completed: {len(predictions)} predictions made")
            logger.info(f"Predicted churn rate: {results['statistics']['predicted_churn_rate']:.2%}")
            
            return results
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
    
    def _predict_in_batches(self, 
                           X: np.ndarray,
                           batch_size: int,
                           return_probabilities: bool) -> tuple:
        """Make predictions in batches for memory efficiency."""
        logger.info(f"Making predictions in batches of {batch_size}")
        
        n_samples = X.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        all_predictions = []
        all_probabilities = [] if return_probabilities else None
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            batch_X = X[start_idx:end_idx]
            
            # Make predictions for batch
            batch_predictions = self.model.predict(batch_X)
            all_predictions.append(batch_predictions)
            
            if return_probabilities:
                batch_probabilities = self.model.predict_proba(batch_X)
                all_probabilities.append(batch_probabilities)
            
            if (i + 1) % 10 == 0 or i == n_batches - 1:
                logger.info(f"Processed batch {i + 1}/{n_batches}")
        
        # Combine results
        predictions = np.concatenate(all_predictions)
        probabilities = (
            np.concatenate(all_probabilities) 
            if all_probabilities else None
        )
        
        return predictions, probabilities
    
    def predict_single(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction for a single customer.
        
        Args:
            customer_data: Dictionary with customer features
            
        Returns:
            Single prediction result
        """
        # Convert to DataFrame
        df = pd.DataFrame([customer_data])
        
        # Make prediction
        results = self.predict(df, return_probabilities=True)
        
        # Extract single result
        single_result = {
            'prediction': results['predictions'][0],
            'churn_probability': results['probabilities']['class_1'][0],
            'confidence': results['confidence_scores'][0],
            'prediction_label': 'Churn' if results['predictions'][0] == 1 else 'No Churn',
            'metadata': results['metadata']
        }
        
        return single_result
    
    def save_predictions(self, 
                        results: Dict[str, Any],
                        output_file: Optional[str] = None,
                        include_input_data: bool = False,
                        input_data: Optional[pd.DataFrame] = None) -> str:
        """
        Save predictions to file.
        
        Args:
            results: Prediction results from predict()
            output_file: Output file path
            include_input_data: Whether to include original input data
            input_data: Original input data (if include_input_data=True)
            
        Returns:
            Path to saved file
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = PREDICTIONS_PATH / f"predictions_{timestamp}.csv"
        else:
            output_file = Path(output_file)
        
        # Create predictions DataFrame
        pred_df = pd.DataFrame({
            'prediction': results['predictions'],
            'churn_probability': (
                results['probabilities']['class_1'] 
                if 'probabilities' in results else None
            ),
            'confidence_score': (
                results['confidence_scores']
                if 'confidence_scores' in results else None
            )
        })
        
        # Add prediction labels
        pred_df['prediction_label'] = pred_df['prediction'].map({
            0: 'No Churn', 1: 'Churn'
        })
        
        # Include input data if requested
        if include_input_data and input_data is not None:
            # Reset index to ensure alignment
            input_data_reset = input_data.reset_index(drop=True)
            pred_df_reset = pred_df.reset_index(drop=True)
            
            # Combine DataFrames
            final_df = pd.concat([input_data_reset, pred_df_reset], axis=1)
        else:
            final_df = pred_df
        
        # Save to file
        final_df.to_csv(output_file, index=False)
        
        # Save metadata
        metadata_file = output_file.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            json.dump(results['metadata'], f, indent=2)
        
        logger.info(f"Predictions saved to {output_file}")
        logger.info(f"Metadata saved to {metadata_file}")
        
        return str(output_file)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        info = {
            'model_info': self.model_info,
            'model_type': type(self.model).__name__,
            'preprocessor_loaded': self.preprocessor is not None,
        }
        
        # Add model-specific info if available
        if hasattr(self.model, 'feature_importances_'):
            info['has_feature_importance'] = True
            info['n_features'] = len(self.model.feature_importances_)
        elif hasattr(self.model, 'coef_'):
            info['has_coefficients'] = True
            info['n_features'] = len(self.model.coef_[0])
        
        return info
    
    def explain_prediction(self, 
                          customer_data: Union[Dict[str, Any], pd.DataFrame],
                          top_n_features: int = 10) -> Dict[str, Any]:
        """
        Provide explanation for prediction (feature importance based).
        
        Args:
            customer_data: Customer data for explanation
            top_n_features: Number of top features to show
            
        Returns:
            Explanation results
        """
        if isinstance(customer_data, dict):
            df = pd.DataFrame([customer_data])
        else:
            df = customer_data.copy()
        
        # Make prediction
        prediction_result = self.predict(df, return_probabilities=True)
        
        explanation = {
            'prediction': prediction_result['predictions'][0],
            'churn_probability': prediction_result['probabilities']['class_1'][0],
            'features_explanation': None
        }
        
        # Get feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            feature_names = self.preprocessor.get_feature_names()
            if feature_names:
                # Get feature importance
                importance = self.model.feature_importances_
                
                # Create feature importance DataFrame
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                
                # Get top features
                top_features = importance_df.head(top_n_features)
                
                explanation['features_explanation'] = {
                    'top_features': top_features.to_dict('records'),
                    'explanation_method': 'feature_importance'
                }
        
        elif hasattr(self.model, 'coef_'):
            feature_names = self.preprocessor.get_feature_names()
            if feature_names:
                # Get coefficients (for logistic regression)
                coefficients = np.abs(self.model.coef_[0])
                
                # Create coefficients DataFrame
                coef_df = pd.DataFrame({
                    'feature': feature_names,
                    'coefficient_abs': coefficients
                }).sort_values('coefficient_abs', ascending=False)
                
                # Get top features
                top_features = coef_df.head(top_n_features)
                
                explanation['features_explanation'] = {
                    'top_features': top_features.to_dict('records'),
                    'explanation_method': 'coefficients'
                }
        
        return explanation


class BatchInferencePipeline:
    """Pipeline for batch inference on large datasets."""
    
    def __init__(self, inference_pipeline: InferencePipeline):
        self.inference_pipeline = inference_pipeline
    
    def process_file(self, 
                    input_file: str,
                    output_file: str,
                    batch_size: int = 1000,
                    chunk_size: int = 10000) -> Dict[str, Any]:
        """
        Process large CSV file in chunks.
        
        Args:
            input_file: Input CSV file path
            output_file: Output CSV file path
            batch_size: Batch size for predictions
            chunk_size: Chunk size for reading CSV
            
        Returns:
            Processing summary
        """
        logger.info(f"Processing file: {input_file}")
        
        # Initialize counters
        total_processed = 0
        total_churn_predicted = 0
        
        # Process file in chunks
        chunk_iter = pd.read_csv(input_file, chunksize=chunk_size)
        
        first_chunk = True
        
        for i, chunk in enumerate(chunk_iter):
            logger.info(f"Processing chunk {i + 1} (size: {len(chunk)})")
            
            # Make predictions for chunk
            results = self.inference_pipeline.predict(
                chunk, 
                return_probabilities=True,
                batch_size=batch_size
            )
            
            # Create results DataFrame
            pred_df = pd.DataFrame({
                'prediction': results['predictions'],
                'churn_probability': results['probabilities']['class_1'],
                'confidence_score': results['confidence_scores']
            })
            
            # Combine with input data
            combined_df = pd.concat([
                chunk.reset_index(drop=True),
                pred_df.reset_index(drop=True)
            ], axis=1)
            
            # Write to output file
            combined_df.to_csv(
                output_file,
                mode='w' if first_chunk else 'a',
                header=first_chunk,
                index=False
            )
            
            # Update counters
            total_processed += len(chunk)
            total_churn_predicted += sum(results['predictions'])
            
            first_chunk = False
        
        summary = {
            'total_processed': total_processed,
            'total_churn_predicted': total_churn_predicted,
            'churn_rate': total_churn_predicted / total_processed,
            'output_file': output_file
        }
        
        logger.info(f"Batch processing completed: {summary}")
        return summary


def main():
    """Main function for inference pipeline."""
    parser = argparse.ArgumentParser(description="Telco Churn Inference Pipeline")
    
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='Input CSV file for predictions'
    )
    
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Output file for predictions'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to model file'
    )
    
    parser.add_argument(
        '--preprocessor_path',
        type=str,
        default=None,
        help='Path to preprocessor file'
    )
    
    parser.add_argument(
        '--model_name',
        type=str,
        default=None,
        help='MLflow model registry name'
    )
    
    parser.add_argument(
        '--model_version',
        type=str,
        default=None,
        help='MLflow model version'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1000,
        help='Batch size for predictions'
    )
    
    parser.add_argument(
        '--include_input',
        action='store_true',
        help='Include input data in output file'
    )
    
    args = parser.parse_args()
    
    # Create directories
    create_directories()
    
    try:
        # Initialize inference pipeline
        pipeline = InferencePipeline(
            model_path=args.model_path,
            preprocessor_path=args.preprocessor_path,
            model_name=args.model_name,
            model_version=args.model_version
        )
        
        # Load input data
        input_data = pd.read_csv(args.input_file)
        logger.info(f"Loaded input data: {input_data.shape}")
        
        # Make predictions
        results = pipeline.predict(
            input_data,
            return_probabilities=True,
            batch_size=args.batch_size
        )
        
        # Save predictions
        output_file = pipeline.save_predictions(
            results,
            output_file=args.output_file,
            include_input_data=args.include_input,
            input_data=input_data if args.include_input else None
        )
        
        print("\n" + "="*50)
        print("INFERENCE COMPLETED")
        print("="*50)
        print(f"Input file: {args.input_file}")
        print(f"Output file: {output_file}")
        print(f"Total predictions: {results['statistics']['total_predictions']}")
        print(f"Predicted churn rate: {results['statistics']['predicted_churn_rate']:.2%}")
        print(f"Model info: {pipeline.get_model_info()}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()