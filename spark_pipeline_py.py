"""PySpark MLlib pipeline for Telco Churn Prediction."""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json

# PySpark imports
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, when, isnan, isnull
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler,
    StandardScaler, Imputer
)
from pyspark.ml.classification import (
    LogisticRegression, RandomForestClassifier, GBTClassifier
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.param.shared import HasSeed

# MLflow for tracking
import mlflow
import mlflow.spark

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.config import (
    SPARK_CONFIG, CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COLUMN,
    MLFLOW_TRACKING_URI, create_directories, MODEL_DIR, LOGS_DIR,
    SPARK_MODELS_CONFIG
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'spark_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SparkChurnPipeline:
    """Complete Spark MLlib pipeline for Telco Churn Prediction."""
    
    def __init__(self, 
                 app_name: str = "TelcoChurnPrediction",
                 master: str = "local[*]",
                 experiment_name: str = "telco_churn_spark"):
        """
        Initialize Spark pipeline.
        
        Args:
            app_name: Spark application name
            master: Spark master URL
            experiment_name: MLflow experiment name
        """
        self.app_name = app_name
        self.master = master
        self.experiment_name = experiment_name
        self.spark = None
        self.models = {}
        
        # Initialize Spark session
        self._init_spark()
        
        # Setup MLflow
        self._setup_mlflow()
    
    def _init_spark(self):
        """Initialize Spark session with optimized configuration."""
        logger.info("Initializing Spark session...")
        
        builder = SparkSession.builder.appName(self.app_name)
        
        # Apply configuration
        for key, value in SPARK_CONFIG.items():
            if key != "spark.app.name":  # Already set
                builder = builder.config(key, value)
        
        # Set master if not in config
        if "spark.master" not in SPARK_CONFIG:
            builder = builder.master(self.master)
        
        self.spark = builder.getOrCreate()
        
        # Set log level to reduce verbosity
        self.spark.sparkContext.setLogLevel("WARN")
        
        logger.info(f"Spark session initialized: {self.spark.version}")
        logger.info(f"Spark UI: {self.spark.sparkContext.uiWebUrl}")
    
    def _setup_mlflow(self):
        """Setup MLflow for experiment tracking."""
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        try:
            mlflow.create_experiment(self.experiment_name)
            logger.info(f"Created MLflow experiment: {self.experiment_name}")
        except mlflow.exceptions.MlflowException:
            logger.info(f"Using existing MLflow experiment: {self.experiment_name}")
        
        mlflow.set_experiment(self.experiment_name)
    
    def load_data(self, data_path: str) -> DataFrame:
        """
        Load data from CSV file.
        
        Args:
            data_path: Path to CSV file
            
        Returns:
            Spark DataFrame
        """
        logger.info(f"Loading data from {data_path}")
        
        df = self.spark.read.csv(
            data_path,
            header=True,
            inferSchema=True
        )
        
        logger.info(f"Loaded data: {df.count()} rows, {len(df.columns)} columns")
        
        return df
    
    def preprocess_data(self, df: DataFrame) -> DataFrame:
        """
        Preprocess data for ML pipeline.
        
        Args:
            df: Raw Spark DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Preprocessing data...")
        
        # Handle TotalCharges - convert to numeric
        df = df.withColumn(
            "TotalCharges",
            when(col("TotalCharges") == " ", None)
            .otherwise(col("TotalCharges").cast(DoubleType()))
        )
        
        # Fill missing TotalCharges with tenure * MonthlyCharges
        df = df.withColumn(
            "TotalCharges",
            when(col("TotalCharges").isNull(), 
                 col("tenure") * col("MonthlyCharges"))
            .otherwise(col("TotalCharges"))
        )
        
        # Convert target to numeric
        df = df.withColumn(
            TARGET_COLUMN,
            when(col(TARGET_COLUMN) == "Yes", 1.0).otherwise(0.0)
        )
        
        # Convert SeniorCitizen to string for consistent categorical handling
        df = df.withColumn("SeniorCitizen", col("SeniorCitizen").cast("string"))
        
        # Feature engineering
        df = self._create_features(df)
        
        logger.info("Data preprocessing completed")
        
        return df
    
    def _create_features(self, df: DataFrame) -> DataFrame:
        """Create additional features."""
        # Average monthly charges
        df = df.withColumn(
            "avg_monthly_charges",
            when(col("tenure") > 0, col("TotalCharges") / col("tenure"))
            .otherwise(col("MonthlyCharges"))
        )
        
        # Tenure groups
        df = df.withColumn(
            "tenure_group",
            when(col("tenure") <= 12, "0-12 months")
            .when(col("tenure") <= 24, "12-24 months")  
            .when(col("tenure") <= 48, "24-48 months")
            .otherwise("48+ months")
        )
        
        return df
    
    def create_ml_pipeline(self, model_type: str = "logistic_regression") -> Pipeline:
        """
        Create ML pipeline with preprocessing and model.
        
        Args:
            model_type: Type of model to use
            
        Returns:
            ML Pipeline
        """
        logger.info(f"Creating ML pipeline with {model_type}")
        
        stages = []
        
        # String indexing for categorical features
        categorical_cols = [col for col in CATEGORICAL_FEATURES if col != "SeniorCitizen"] + ["SeniorCitizen", "tenure_group"]
        indexed_cols = []
        
        for col_name in categorical_cols:
            indexer = StringIndexer(
                inputCol=col_name,
                outputCol=f"{col_name}_indexed",
                handleInvalid="keep"
            )
            stages.append(indexer)
            indexed_cols.append(f"{col_name}_indexed")
        
        # One-hot encoding
        encoder = OneHotEncoder(
            inputCols=indexed_cols,
            outputCols=[f"{col}_encoded" for col in indexed_cols]
        )
        stages.append(encoder)
        
        # Vector assembling
        feature_cols = NUMERICAL_FEATURES + ["avg_monthly_charges"] + [f"{col}_encoded" for col in indexed_cols]
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features_unscaled",
            handleInvalid="skip"
        )
        stages.append(assembler)
        
        # Feature scaling
        scaler = StandardScaler(
            inputCol="features_unscaled",
            outputCol="features",
            withStd=True,
            withMean=True
        )
        stages.append(scaler)
        
        # Add classifier
        classifier = self._get_classifier(model_type)
        stages.append(classifier)
        
        pipeline = Pipeline(stages=stages)
        
        return pipeline
    
    def _get_classifier(self, model_type: str):
        """Get classifier based on model type."""
        classifiers = {
            "logistic_regression": LogisticRegression(
                featuresCol="features",
                labelCol=TARGET_COLUMN,
                seed=42
            ),
            "random_forest": RandomForestClassifier(
                featuresCol="features",
                labelCol=TARGET_COLUMN,
                seed=42
            ),
            "gradient_boosted_trees": GBTClassifier(
                featuresCol="features",
                labelCol=TARGET_COLUMN,
                seed=42
            )
        }
        
        if model_type not in classifiers:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return classifiers[model_type]
    
    def train_model_with_cv(self, 
                           df: DataFrame,
                           model_type: str = "logistic_regression",
                           test_ratio: float = 0.2) -> Dict[str, Any]:
        """
        Train model with cross-validation and hyperparameter tuning.
        
        Args:
            df: Preprocessed DataFrame
            model_type: Type of model to train
            test_ratio: Test set ratio
            
        Returns:
            Training results and model
        """
        logger.info(f"Training {model_type} with cross-validation...")
        
        with mlflow.start_run(run_name=f"spark_{model_type}"):
            start_time = time.time()
            
            # Log parameters
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("framework", "spark")
            mlflow.log_param("test_ratio", test_ratio)
            mlflow.log_param("data_count", df.count())
            
            # Split data
            train_df, test_df = df.randomSplit([1 - test_ratio, test_ratio], seed=42)
            
            train_count = train_df.count()
            test_count = test_df.count()
            
            mlflow.log_param("train_count", train_count)
            mlflow.log_param("test_count", test_count)
            
            logger.info(f"Data split: Train {train_count}, Test {test_count}")
            
            # Create pipeline
            pipeline = self.create_ml_pipeline(model_type)
            
            # Setup parameter grid for hyperparameter tuning
            param_grid = self._create_param_grid(pipeline, model_type)
            
            # Setup evaluators
            binary_evaluator = BinaryClassificationEvaluator(
                labelCol=TARGET_COLUMN,
                metricName="areaUnderROC"
            )
            
            # Cross validator
            cv = CrossValidator(
                estimator=pipeline,
                estimatorParamMaps=param_grid,
                evaluator=binary_evaluator,
                numFolds=3,
                seed=42
            )
            
            # Fit model
            logger.info("Fitting model with cross-validation...")
            cv_model = cv.fit(train_df)
            
            training_time = time.time() - start_time
            mlflow.log_metric("training_time_seconds", training_time)
            
            # Get best model
            best_model = cv_model.bestModel
            
            # Log best parameters
            self._log_best_params(best_model, model_type)
            
            # Evaluate on test set
            test_predictions = best_model.transform(test_df)
            
            # Calculate metrics
            metrics = self._calculate_metrics(test_predictions)
            
            # Log metrics
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"test_{metric_name}", value)
            
            # Log model
            mlflow.spark.log_model(
                spark_model=best_model,
                artifact_path=f"spark_{model_type}_model",
                registered_model_name=f"telco_churn_spark_{model_type}"
            )
            
            # Save model locally
            model_path = MODEL_DIR / f"spark_{model_type}_model"
            best_model.write().overwrite().save(str(model_path))
            
            results = {
                'model': best_model,
                'metrics': metrics,
                'training_time': training_time,
                'model_path': str(model_path),
                'test_predictions': test_predictions
            }
            
            self.models[model_type] = results
            
            logger.info(f"{model_type} training completed. AUC: {metrics['auc']:.4f}")
            
            return results
    
    def _create_param_grid(self, pipeline: Pipeline, model_type: str) -> List[Dict]:
        """Create parameter grid for hyperparameter tuning."""
        param_config = SPARK_MODELS_CONFIG.get(model_type, {})
        
        if not param_config:
            return [{}]  # Empty grid if no parameters defined
        
        builder = ParamGridBuilder()
        
        # Get the classifier stage (last stage in pipeline)
        classifier = pipeline.getStages()[-1]
        
        # Add parameters based on model type
        if model_type == "logistic_regression":
            if "regParam" in param_config:
                builder = builder.addGrid(classifier.regParam, param_config["regParam"])
            if "elasticNetParam" in param_config:
                builder = builder.addGrid(classifier.elasticNetParam, param_config["elasticNetParam"])
            if "maxIter" in param_config:
                builder = builder.addGrid(classifier.maxIter, param_config["maxIter"])
        
        elif model_type == "random_forest":
            if "numTrees" in param_config:
                builder = builder.addGrid(classifier.numTrees, param_config["numTrees"])
            if "maxDepth" in param_config:
                builder = builder.addGrid(classifier.maxDepth, param_config["maxDepth"])
            if "minInstancesPerNode" in param_config:
                builder = builder.addGrid(classifier.minInstancesPerNode, param_config["minInstancesPerNode"])
        
        elif model_type == "gradient_boosted_trees":
            if "maxIter" in param_config:
                builder = builder.addGrid(classifier.maxIter, param_config["maxIter"])
            if "stepSize" in param_config:
                builder = builder.addGrid(classifier.stepSize, param_config["stepSize"])
            if "maxDepth" in param_config:
                builder = builder.addGrid(classifier.maxDepth, param_config["maxDepth"])
        
        return builder.build()
    
    def _log_best_params(self, best_model: Pipeline, model_type: str):
        """Log best parameters from cross-validation."""
        try:
            # Get the classifier stage (last stage)
            classifier = best_model.stages[-1]
            
            # Extract parameters based on model type
            if model_type == "logistic_regression":
                mlflow.log_param("best_regParam", classifier.getRegParam())
                mlflow.log_param("best_elasticNetParam", classifier.getElasticNetParam())
                mlflow.log_param("best_maxIter", classifier.getMaxIter())
            
            elif model_type == "random_forest":
                mlflow.log_param("best_numTrees", classifier.getNumTrees())
                mlflow.log_param("best_maxDepth", classifier.getMaxDepth())
                mlflow.log_param("best_minInstancesPerNode", classifier.getMinInstancesPerNode())
            
            elif model_type == "gradient_boosted_trees":
                mlflow.log_param("best_maxIter", classifier.getMaxIter())
                mlflow.log_param("best_stepSize", classifier.getStepSize())
                mlflow.log_param("best_maxDepth", classifier.getMaxDepth())
        
        except Exception as e:
            logger.warning(f"Could not log best parameters: {e}")
    
    def _calculate_metrics(self, predictions: DataFrame) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        # Binary classification evaluator
        binary_evaluator = BinaryClassificationEvaluator(
            labelCol=TARGET_COLUMN,
            metricName="areaUnderROC"
        )
        auc = binary_evaluator.evaluate(predictions)
        
        binary_evaluator_pr = BinaryClassificationEvaluator(
            labelCol=TARGET_COLUMN,
            metricName="areaUnderPR"
        )
        auc_pr = binary_evaluator_pr.evaluate(predictions)
        
        # Multiclass evaluator for additional metrics
        multiclass_evaluator = MulticlassClassificationEvaluator(
            labelCol=TARGET_COLUMN,
            predictionCol="prediction"
        )
        
        accuracy = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "accuracy"})
        precision = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "weightedPrecision"})
        recall = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "weightedRecall"})
        f1 = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "f1"})
        
        return {
            'auc': auc,
            'auc_pr': auc_pr,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def compare_models(self, df: DataFrame, model_types: List[str] = None) -> DataFrame:
        """
        Compare multiple models and return results.
        
        Args:
            df: Preprocessed DataFrame
            model_types: List of model types to compare
            
        Returns:
            Comparison results as Spark DataFrame
        """
        if model_types is None:
            model_types = ["logistic_regression", "random_forest", "gradient_boosted_trees"]
        
        logger.info(f"Comparing models: {model_types}")
        
        with mlflow.start_run(run_name="spark_model_comparison"):
            results = []
            
            for model_type in model_types:
                logger.info(f"Training {model_type} for comparison...")
                
                model_results = self.train_model_with_cv(df, model_type)
                
                result_row = {
                    'model_type': model_type,
                    'framework': 'spark',
                    'training_time': model_results['training_time'],
                    **model_results['metrics']
                }
                
                results.append(result_row)
            
            # Create comparison DataFrame
            comparison_df = self.spark.createDataFrame(results)
            
            # Show results
            logger.info("Model comparison results:")
            comparison_df.show()
            
            # Save results
            comparison_path = LOGS_DIR / "spark_model_comparison.json"
            with open(comparison_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            mlflow.log_artifact(str(comparison_path))
            
            # Log best model
            best_result = max(results, key=lambda x: x['auc'])
            mlflow.log_param("best_model", best_result['model_type'])
            mlflow.log_metric("best_auc", best_result['auc'])
            
            logger.info(f"Best model: {best_result['model_type']} (AUC: {best_result['auc']:.4f})")
            
            return comparison_df
    
    def predict_batch(self, 
                     model_path: str,
                     input_data: DataFrame,
                     output_path: str = None) -> DataFrame:
        """
        Make batch predictions using saved model.
        
        Args:
            model_path: Path to saved Spark model
            input_data: Input DataFrame
            output_path: Optional output path to save predictions
            
        Returns:
            DataFrame with predictions
        """
        logger.info(f"Loading model from {model_path}")
        
        # Load model
        from pyspark.ml import PipelineModel
        model = PipelineModel.load(model_path)
        
        # Make predictions
        predictions = model.transform(input_data)
        
        # Select relevant columns
        result_df = predictions.select(
            "*",  # All original columns
            "prediction",
            "probability"
        )
        
        # Save if output path provided
        if output_path:
            logger.info(f"Saving predictions to {output_path}")
            result_df.write.mode("overwrite").csv(output_path, header=True)
        
        return result_df
    
    def performance_comparison_with_sklearn(self, 
                                          sklearn_results: Dict[str, Any],
                                          spark_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare Spark and sklearn performance.
        
        Args:
            sklearn_results: Results from sklearn models
            spark_results: Results from Spark models
            
        Returns:
            Comparison summary
        """
        logger.info("Comparing Spark vs Sklearn performance...")
        
        comparison = {
            'framework_comparison': {},
            'model_comparison': {}
        }
        
        # Framework-level comparison
        sklearn_times = [result.get('training_time', 0) for result in sklearn_results.values()]
        spark_times = [result.get('training_time', 0) for result in spark_results.values()]
        
        comparison['framework_comparison'] = {
            'sklearn_avg_time': sum(sklearn_times) / len(sklearn_times) if sklearn_times else 0,
            'spark_avg_time': sum(spark_times) / len(spark_times) if spark_times else 0,
            'spark_speedup': (sum(sklearn_times) / sum(spark_times)) if sum(spark_times) > 0 else 1
        }
        
        # Model-level comparison
        for model_type in set(sklearn_results.keys()) & set(spark_results.keys()):
            sklearn_auc = sklearn_results[model_type]['metrics'].get('auc', 0)
            spark_auc = spark_results[model_type]['metrics'].get('auc', 0)
            
            sklearn_time = sklearn_results[model_type].get('training_time', 0)
            spark_time = spark_results[model_type].get('training_time', 0)
            
            comparison['model_comparison'][model_type] = {
                'sklearn_auc': sklearn_auc,
                'spark_auc': spark_auc,
                'auc_difference': spark_auc - sklearn_auc,
                'sklearn_time': sklearn_time,
                'spark_time': spark_time,
                'time_ratio': sklearn_time / spark_time if spark_time > 0 else 1
            }
        
        # Log comparison
        comparison_path = LOGS_DIR / "spark_sklearn_comparison.json"
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info(f"Performance comparison saved to {comparison_path}")
        
        return comparison
    
    def cleanup(self):
        """Cleanup Spark session."""
        if self.spark:
            logger.info("Stopping Spark session...")
            self.spark.stop()
    
    def __del__(self):
        """Destructor to cleanup Spark session."""
        self.cleanup()


def main():
    """Main function for Spark pipeline."""
    parser = argparse.ArgumentParser(description="Telco Churn Spark Pipeline")
    
    parser.add_argument(
        '--mode',
        choices=['train', 'predict', 'compare'],
        required=True,
        help='Pipeline mode'
    )
    
    parser.add_argument(
        '--data_file',
        type=str,
        default="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        help='Input data file'
    )
    
    parser.add_argument(
        '--model_type',
        type=str,
        choices=['logistic_regression', 'random_forest', 'gradient_boosted_trees'],
        default='random_forest',
        help='Model type to train'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to saved model (for prediction mode)'
    )
    
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Output path for predictions'
    )
    
    parser.add_argument(
        '--master',
        type=str,
        default="local[*]",
        help='Spark master URL'
    )
    
    parser.add_argument(
        '--experiment_name',
        type=str,
        default="telco_churn_spark",
        help='MLflow experiment name'
    )
    
    args = parser.parse_args()
    
    # Create directories
    create_directories()
    
    # Initialize pipeline
    pipeline = SparkChurnPipeline(
        master=args.master,
        experiment_name=args.experiment_name
    )
    
    try:
        if args.mode == 'train':
            # Load and preprocess data
            df = pipeline.load_data(args.data_file)
            df_processed = pipeline.preprocess_data(df)
            
            # Train model
            results = pipeline.train_model_with_cv(df_processed, args.model_type)
            
            print("\n" + "="*50)
            print("SPARK TRAINING COMPLETED")
            print("="*50)
            print(f"Model type: {args.model_type}")
            print(f"Training time: {results['training_time']:.2f} seconds")
            print(f"Test AUC: {results['metrics']['auc']:.4f}")
            print(f"Test Accuracy: {results['metrics']['accuracy']:.4f}")
            print(f"Model saved to: {results['model_path']}")
            print("="*50)
        
        elif args.mode == 'compare':
            # Load and preprocess data
            df = pipeline.load_data(args.data_file)
            df_processed = pipeline.preprocess_data(df)
            
            # Compare all models
            comparison_df = pipeline.compare_models(df_processed)
            
            print("\n" + "="*50)
            print("SPARK MODEL COMPARISON COMPLETED")
            print("="*50)
            print("Results saved to logs/spark_model_comparison.json")
            print("="*50)
        
        elif args.mode == 'predict':
            if not args.model_path:
                # Try to find latest model
                model_files = list(MODEL_DIR.glob(f"spark_{args.model_type}_model"))
                if not model_files:
                    raise ValueError(f"No model found for {args.model_type}")
                args.model_path = str(model_files[0])
            
            # Load input data
            df = pipeline.load_data(args.data_file)
            df_processed = pipeline.preprocess_data(df)
            
            # Make predictions
            predictions = pipeline.predict_batch(
                args.model_path,
                df_processed,
                args.output_path
            )
            
            # Show sample predictions
            print("\n" + "="*50)
            print("SPARK PREDICTIONS COMPLETED")
            print("="*50)
            print("Sample predictions:")
            predictions.select("customerID", "prediction", "probability").show(10)
            
            if args.output_path:
                print(f"Full predictions saved to: {args.output_path}")
            print("="*50)
    
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)
    
    finally:
        # Cleanup
        pipeline.cleanup()


if __name__ == "__main__":
    main()