"""Training pipeline for Telco Churn Prediction with MLflow integration."""

import os
import sys
import argparse
import logging
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, Any, Tuple, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_loader import DataLoader
from src.preprocessing.preprocessor import TelcoPreprocessor
from src.models.sklearn_model import TelcoChurnClassifier, create_model_comparison
from config.config import (
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, 
    create_directories, MODEL_DIR, LOGS_DIR
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'training_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Complete training pipeline with MLflow integration."""
    
    def __init__(self, 
                 experiment_name: str = MLFLOW_EXPERIMENT_NAME,
                 test_size: float = 0.2,
                 random_state: int = 42):
        """
        Initialize training pipeline.
        
        Args:
            experiment_name: MLflow experiment name
            test_size: Test set size ratio
            random_state: Random state for reproducibility
        """
        self.experiment_name = experiment_name
        self.test_size = test_size
        self.random_state = random_state
        self.data_loader = DataLoader()
        self.preprocessor = TelcoPreprocessor()
        self.models_trained = {}
        
        # Setup MLflow
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Setup MLflow tracking."""
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Create or get experiment
        try:
            experiment_id = mlflow.create_experiment(self.experiment_name)
            logger.info(f"Created new experiment: {self.experiment_name}")
        except mlflow.exceptions.MlflowException:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: {self.experiment_name}")
        
        mlflow.set_experiment(self.experiment_name)
    
    def load_and_prepare_data(self, data_file: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and prepare data for training.
        
        Args:
            data_file: Optional custom data file path
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info("Loading and preparing data...")
        
        with mlflow.start_run(run_name="data_preparation", nested=True):
            # Load raw data
            if data_file:
                df = pd.read_csv(data_file)
                mlflow.log_param("data_source", data_file)
            else:
                df = self.data_loader.load_raw_data()
                mlflow.log_param("data_source", "kaggle_telco_churn")
            
            # Log data info
            mlflow.log_param("raw_data_shape", df.shape)
            mlflow.log_param("raw_data_columns", len(df.columns))
            
            # Validate data
            validation_results = self.data_loader.validate_data(df)
            mlflow.log_param("data_validation_passed", validation_results['is_valid'])
            
            if not validation_results['is_valid']:
                for issue in validation_results['issues']:
                    logger.warning(f"Data validation issue: {issue}")
            
            # Split features and target
            X, y = self.data_loader.split_features_target(df)
            
            # Log class distribution
            class_dist = pd.Series(y).value_counts(normalize=True).to_dict()
            mlflow.log_params({f"class_distribution_{k}": v for k, v in class_dist.items()})
            
            # Preprocessing
            logger.info("Applying preprocessing...")
            X_processed = self.preprocessor.fit_transform(X)
            
            # Log preprocessing info
            mlflow.log_param("preprocessing_scaling", self.preprocessor.scaling_method)
            mlflow.log_param("preprocessing_encoding", self.preprocessor.encoding_method)
            mlflow.log_param("processed_features_count", X_processed.shape[1])
            
            # Save preprocessor
            preprocessor_path = MODEL_DIR / "preprocessor.pkl"
            self.preprocessor.save_pipeline(str(preprocessor_path))
            mlflow.log_artifact(str(preprocessor_path))
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, 
                test_size=self.test_size,
                stratify=y,
                random_state=self.random_state
            )
            
            # Log split info
            mlflow.log_param("test_size", self.test_size)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            
            logger.info(f"Data prepared: Train shape {X_train.shape}, Test shape {X_test.shape}")
            
            return X_train, X_test, y_train, y_test
    
    def train_single_model(self, 
                          model_type: str,
                          X_train: np.ndarray,
                          y_train: np.ndarray,
                          X_test: np.ndarray,
                          y_test: np.ndarray,
                          hyperparameter_tuning: bool = True) -> TelcoChurnClassifier:
        """
        Train a single model with MLflow tracking.
        
        Args:
            model_type: Type of model to train
            X_train, y_train: Training data
            X_test, y_test: Test data
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            Trained classifier
        """
        logger.info(f"Training {model_type} model...")
        
        with mlflow.start_run(run_name=f"train_{model_type}", nested=True):
            # Initialize model
            classifier = TelcoChurnClassifier(
                model_type=model_type,
                hyperparameter_tuning=hyperparameter_tuning,
                random_state=self.random_state
            )
            
            # Log model parameters
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("hyperparameter_tuning", hyperparameter_tuning)
            mlflow.log_param("cv_folds", classifier.cv_folds)
            mlflow.log_param("random_state", self.random_state)
            
            # Train model
            classifier.fit(X_train, y_train)
            
            # Log best hyperparameters if tuning was performed
            if classifier.best_params:
                mlflow.log_params({f"best_{k}": v for k, v in classifier.best_params.items()})
            
            # Log cross-validation scores
            if classifier.cv_scores:
                for metric, scores in classifier.cv_scores.items():
                    mlflow.log_metric(f"cv_{metric}_mean", scores['mean'])
                    mlflow.log_metric(f"cv_{metric}_std", scores['std'])
            
            # Evaluate on test set
            test_metrics = classifier.evaluate(X_test, y_test)
            
            # Log test metrics
            for metric, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric}", value)
            
            # Generate and log visualizations
            self._log_model_visualizations(classifier, X_test, y_test, model_type)
            
            # Save and log model
            model_path = MODEL_DIR / f"{model_type}_model.pkl"
            classifier.save_model(str(model_path))
            mlflow.log_artifact(str(model_path))
            
            # Log sklearn model for MLflow model registry
            mlflow.sklearn.log_model(
                sk_model=classifier.model,
                artifact_path=f"sklearn_{model_type}_model",
                registered_model_name=f"telco_churn_{model_type}"
            )
            
            # Store trained model
            self.models_trained[model_type] = classifier
            
            logger.info(f"{model_type} training completed. Test AUC: {test_metrics['roc_auc']:.4f}")
            
            return classifier
    
    def _log_model_visualizations(self, 
                                 classifier: TelcoChurnClassifier,
                                 X_test: np.ndarray,
                                 y_test: np.ndarray,
                                 model_type: str):
        """Generate and log model visualizations."""
        try:
            # Feature importance plot
            if classifier.feature_importance is not None:
                feature_names = self.preprocessor.get_feature_names()
                fig = classifier.plot_feature_importance(
                    feature_names=feature_names,
                    top_n=20
                )
                importance_path = LOGS_DIR / f"{model_type}_feature_importance.png"
                fig.savefig(importance_path, dpi=300, bbox_inches='tight')
                mlflow.log_artifact(str(importance_path))
                plt.close(fig)
            
            # Confusion matrix
            fig = classifier.plot_confusion_matrix(X_test, y_test)
            cm_path = LOGS_DIR / f"{model_type}_confusion_matrix.png"
            fig.savefig(cm_path, dpi=300, bbox_inches='tight')
            mlflow.log_artifact(str(cm_path))
            plt.close(fig)
            
            # Classification report
            report = classifier.get_classification_report(X_test, y_test)
            report_path = LOGS_DIR / f"{model_type}_classification_report.txt"
            with open(report_path, 'w') as f:
                f.write(report)
            mlflow.log_artifact(str(report_path))
            
            logger.info(f"Visualizations logged for {model_type}")
            
        except Exception as e:
            logger.warning(f"Could not generate visualizations for {model_type}: {e}")
    
    def train_all_models(self,
                        X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_test: np.ndarray,
                        y_test: np.ndarray,
                        model_types: Optional[list] = None) -> Dict[str, TelcoChurnClassifier]:
        """
        Train multiple models and compare performance.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            model_types: List of model types to train
            
        Returns:
            Dictionary of trained models
        """
        if model_types is None:
            model_types = ['logistic_regression', 'random_forest', 'gradient_boosting']
        
        logger.info(f"Training models: {model_types}")
        
        with mlflow.start_run(run_name="model_comparison", nested=True):
            models = {}
            results = []
            
            # Train each model
            for model_type in model_types:
                classifier = self.train_single_model(
                    model_type, X_train, y_train, X_test, y_test
                )
                models[model_type] = classifier
                
                # Collect results
                test_metrics = classifier.evaluate(X_test, y_test)
                result = {'model_type': model_type, **test_metrics}
                
                if classifier.cv_scores:
                    for metric, scores in classifier.cv_scores.items():
                        result[f'cv_{metric}_mean'] = scores['mean']
                        result[f'cv_{metric}_std'] = scores['std']
                
                results.append(result)
            
            # Create comparison DataFrame
            comparison_df = pd.DataFrame(results)
            
            # Log comparison results
            comparison_path = LOGS_DIR / "model_comparison.csv"
            comparison_df.to_csv(comparison_path, index=False)
            mlflow.log_artifact(str(comparison_path))
            
            # Create comparison visualization
            self._create_model_comparison_plot(comparison_df)
            
            # Log best model info
            best_model_type = comparison_df.loc[comparison_df['roc_auc'].idxmax(), 'model_type']
            mlflow.log_param("best_model", best_model_type)
            mlflow.log_metric("best_model_auc", comparison_df['roc_auc'].max())
            
            logger.info(f"Best model: {best_model_type} (AUC: {comparison_df['roc_auc'].max():.4f})")
            
            return models
    
    def _create_model_comparison_plot(self, comparison_df: pd.DataFrame):
        """Create model comparison visualization."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            metrics = ['accuracy', 'precision', 'recall', 'roc_auc']
            
            for i, metric in enumerate(metrics):
                ax = axes[i // 2, i % 2]
                
                # Plot test scores
                bars = ax.bar(comparison_df['model_type'], comparison_df[metric], 
                             alpha=0.7, label='Test Score')
                
                # Add CV scores if available
                cv_metric = f'cv_{metric}_mean'
                if cv_metric in comparison_df.columns:
                    ax.errorbar(comparison_df['model_type'], comparison_df[cv_metric],
                               yerr=comparison_df.get(f'cv_{metric}_std', 0),
                               fmt='ro', capsize=5, label='CV Score')
                
                ax.set_title(f'Model Comparison - {metric.upper()}')
                ax.set_ylabel(metric.capitalize())
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            comparison_plot_path = LOGS_DIR / "model_comparison.png"
            fig.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
            mlflow.log_artifact(str(comparison_plot_path))
            plt.close(fig)
            
            logger.info("Model comparison plot created and logged")
            
        except Exception as e:
            logger.warning(f"Could not create model comparison plot: {e}")
    
    def run_full_pipeline(self, 
                         data_file: Optional[str] = None,
                         model_types: Optional[list] = None) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Args:
            data_file: Optional custom data file
            model_types: List of model types to train
            
        Returns:
            Pipeline results summary
        """
        logger.info("Starting full training pipeline...")
        
        with mlflow.start_run(run_name="full_training_pipeline"):
            # Log pipeline parameters
            mlflow.log_param("pipeline_version", "1.0")
            mlflow.log_param("test_size", self.test_size)
            mlflow.log_param("random_state", self.random_state)
            
            try:
                # Load and prepare data
                X_train, X_test, y_train, y_test = self.load_and_prepare_data(data_file)
                
                # Train models
                models = self.train_all_models(X_train, y_train, X_test, y_test, model_types)
                
                # Create summary
                summary = {
                    'status': 'success',
                    'models_trained': list(models.keys()),
                    'data_shape': {
                        'train': X_train.shape,
                        'test': X_test.shape
                    },
                    'best_model': None,
                    'best_auc': 0
                }
                
                # Find best model
                best_auc = 0
                for name, model in models.items():
                    test_metrics = model.evaluate(X_test, y_test)
                    if test_metrics['roc_auc'] > best_auc:
                        best_auc = test_metrics['roc_auc']
                        summary['best_model'] = name
                        summary['best_auc'] = best_auc
                
                # Log summary
                mlflow.log_param("pipeline_status", summary['status'])
                mlflow.log_param("models_count", len(models))
                mlflow.log_metric("pipeline_best_auc", summary['best_auc'])
                
                # Save summary
                summary_path = LOGS_DIR / "pipeline_summary.json"
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2, default=str)
                mlflow.log_artifact(str(summary_path))
                
                logger.info("Training pipeline completed successfully!")
                logger.info(f"Best model: {summary['best_model']} (AUC: {summary['best_auc']:.4f})")
                
                return summary
                
            except Exception as e:
                logger.error(f"Pipeline failed: {e}")
                mlflow.log_param("pipeline_status", "failed")
                mlflow.log_param("error_message", str(e))
                raise


def main():
    """Main function to run training pipeline."""
    parser = argparse.ArgumentParser(description="Telco Churn Training Pipeline")
    
    parser.add_argument(
        '--experiment_name',
        type=str,
        default=MLFLOW_EXPERIMENT_NAME,
        help='MLflow experiment name'
    )
    
    parser.add_argument(
        '--data_file',
        type=str,
        default=None,
        help='Path to custom data file'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        default=['logistic_regression', 'random_forest', 'gradient_boosting'],
        help='Models to train'
    )
    
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Test set size ratio'
    )
    
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random state for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Create directories
    create_directories()
    
    # Initialize and run pipeline
    pipeline = TrainingPipeline(
        experiment_name=args.experiment_name,
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    try:
        results = pipeline.run_full_pipeline(
            data_file=args.data_file,
            model_types=args.models
        )
        
        print("\n" + "="*50)
        print("TRAINING PIPELINE COMPLETED")
        print("="*50)
        print(f"Experiment: {args.experiment_name}")
        print(f"Models trained: {results['models_trained']}")
        print(f"Best model: {results['best_model']} (AUC: {results['best_auc']:.4f})")
        print(f"MLflow UI: {MLFLOW_TRACKING_URI}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()