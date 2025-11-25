"""Airflow DAG for Telco Churn Prediction ML Pipeline."""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.config import AIRFLOW_CONFIG, create_directories
from src.data.data_loader import DataLoader
from src.preprocessing.preprocessor import TelcoPreprocessor
from src.models.sklearn_model import create_model_comparison
from pipelines.training_pipeline import TrainingPipeline
from pipelines.inference_pipeline import InferencePipeline

# Default arguments
default_args = {
    'owner': AIRFLOW_CONFIG['owner'],
    'depends_on_past': AIRFLOW_CONFIG['depends_on_past'],
    'start_date': days_ago(1),
    'email_on_failure': AIRFLOW_CONFIG['email_on_failure'],
    'email_on_retry': AIRFLOW_CONFIG['email_on_retry'],
    'retries': AIRFLOW_CONFIG['retries'],
    'retry_delay': timedelta(minutes=5),
    'catchup': AIRFLOW_CONFIG['catchup']
}

# DAG definition
dag = DAG(
    'telco_churn_ml_pipeline',
    default_args=default_args,
    description='Complete ML pipeline for Telco Churn Prediction',
    schedule_interval='@weekly',  # Run weekly
    max_active_runs=1,
    tags=['machine-learning', 'telco', 'churn', 'mlflow'],
)

# Task functions
def setup_environment(**context):
    """Setup environment and create necessary directories."""
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Setting up environment...")
    create_directories()
    
    # Log environment info
    logger.info(f"Task instance: {context['task_instance']}")
    logger.info(f"Execution date: {context['execution_date']}")
    
    return "Environment setup completed"


def validate_data(**context):
    """Validate input data quality and structure."""
    import logging
    import pandas as pd
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Validating input data...")
    
    # Load data
    data_loader = DataLoader()
    
    try:
        df = data_loader.load_raw_data()
        
        # Validate data
        validation_results = data_loader.validate_data(df)
        
        # Log validation results
        logger.info(f"Data validation results: {validation_results}")
        
        if not validation_results['is_valid']:
            raise ValueError(f"Data validation failed: {validation_results['issues']}")
        
        # Store validation results in XCom
        context['task_instance'].xcom_push(
            key='validation_results',
            value=validation_results
        )
        
        # Store data summary
        summary = data_loader.get_data_summary(df)
        context['task_instance'].xcom_push(
            key='data_summary',
            value=summary
        )
        
        logger.info("Data validation completed successfully")
        return validation_results
        
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        raise


def preprocess_data(**context):
    """Preprocess data for model training."""
    import logging
    import joblib
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting data preprocessing...")
    
    # Load data
    data_loader = DataLoader()
    df = data_loader.load_raw_data()
    X, y = data_loader.split_features_target(df)
    
    # Create and fit preprocessor
    preprocessor = TelcoPreprocessor(
        scaling_method='standard',
        encoding_method='onehot'
    )
    
    X_processed = preprocessor.fit_transform(X)
    
    # Save preprocessor and processed data
    from config.config import MODEL_DIR, PROCESSED_DATA_PATH
    
    preprocessor_path = MODEL_DIR / "airflow_preprocessor.pkl"
    preprocessor.save_pipeline(str(preprocessor_path))
    
    # Save processed data
    processed_data_path = PROCESSED_DATA_PATH / "airflow_processed_data.pkl"
    joblib.dump({
        'X_processed': X_processed,
        'y': y,
        'feature_names': preprocessor.get_feature_names()
    }, processed_data_path)
    
    # Store paths in XCom
    context['task_instance'].xcom_push(
        key='preprocessor_path',
        value=str(preprocessor_path)
    )
    context['task_instance'].xcom_push(
        key='processed_data_path',
        value=str(processed_data_path)
    )
    
    logger.info(f"Preprocessing completed. Shape: {X_processed.shape}")
    return str(processed_data_path)


def train_sklearn_models(**context):
    """Train sklearn models with hyperparameter tuning."""
    import logging
    import joblib
    from sklearn.model_selection import train_test_split
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Training sklearn models...")
    
    # Get processed data path from XCom
    processed_data_path = context['task_instance'].xcom_pull(
        task_ids='preprocess_data',
        key='processed_data_path'
    )
    
    # Load processed data
    data = joblib.load(processed_data_path)
    X_processed = data['X_processed']
    y = data['y']
    feature_names = data['feature_names']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Train models using comparison function
    results_df = create_model_comparison(
        X_train, y_train, X_test, y_test, feature_names
    )
    
    # Save results
    from config.config import LOGS_DIR
    results_path = LOGS_DIR / "airflow_model_comparison.csv"
    results_df.to_csv(results_path, index=False)
    
    # Find best model
    best_model_type = results_df.loc[results_df['roc_auc'].idxmax(), 'model_type']
    best_auc = results_df['roc_auc'].max()
    
    # Store results in XCom
    context['task_instance'].xcom_push(
        key='model_results',
        value=results_df.to_dict('records')
    )
    context['task_instance'].xcom_push(
        key='best_model_type',
        value=best_model_type
    )
    context['task_instance'].xcom_push(
        key='best_auc',
        value=float(best_auc)
    )
    
    logger.info(f"Model training completed. Best model: {best_model_type} (AUC: {best_auc:.4f})")
    return results_path


def train_spark_models(**context):
    """Train Spark models for comparison."""
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Training Spark models...")
    
    try:
        from pipelines.spark_pipeline import SparkChurnPipeline
        
        # Initialize Spark pipeline
        pipeline = SparkChurnPipeline(
            experiment_name="telco_churn_airflow_spark"
        )
        
        # Load and preprocess data
        df = pipeline.load_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
        df_processed = pipeline.preprocess_data(df)
        
        # Compare models
        comparison_df = pipeline.compare_models(df_processed)
        
        # Convert to pandas for XCom storage
        comparison_results = comparison_df.toPandas().to_dict('records')
        
        # Store results
        context['task_instance'].xcom_push(
            key='spark_results',
            value=comparison_results
        )
        
        logger.info("Spark model training completed")
        return comparison_results
        
    except Exception as e:
        logger.error(f"Spark training failed: {e}")
        # Don't fail the entire pipeline if Spark fails
        logger.warning("Continuing pipeline without Spark results")
        return None
    
    finally:
        # Cleanup Spark session
        try:
            pipeline.cleanup()
        except:
            pass


def evaluate_models(**context):
    """Evaluate and compare all trained models."""
    import logging
    import json
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Evaluating models...")
    
    # Get results from XCom
    sklearn_results = context['task_instance'].xcom_pull(
        task_ids='train_sklearn_models',
        key='model_results'
    )
    
    spark_results = context['task_instance'].xcom_pull(
        task_ids='train_spark_models',
        key='spark_results'
    )
    
    best_model_type = context['task_instance'].xcom_pull(
        task_ids='train_sklearn_models',
        key='best_model_type'
    )
    
    best_auc = context['task_instance'].xcom_pull(
        task_ids='train_sklearn_models',
        key='best_auc'
    )
    
    # Create evaluation summary
    evaluation = {
        'evaluation_date': datetime.now().isoformat(),
        'sklearn_results': sklearn_results,
        'spark_results': spark_results,
        'best_model': {
            'type': best_model_type,
            'framework': 'sklearn',
            'auc': best_auc
        },
        'recommendations': []
    }
    
    # Add recommendations based on results
    if best_auc > 0.85:
        evaluation['recommendations'].append("Model performance is excellent (AUC > 0.85)")
    elif best_auc > 0.75:
        evaluation['recommendations'].append("Model performance is good (AUC > 0.75)")
    else:
        evaluation['recommendations'].append("Model performance needs improvement (AUC < 0.75)")
    
    if spark_results:
        evaluation['recommendations'].append("Spark models trained successfully for scalability")
    
    # Save evaluation
    from config.config import LOGS_DIR
    eval_path = LOGS_DIR / "airflow_model_evaluation.json"
    with open(eval_path, 'w') as f:
        json.dump(evaluation, f, indent=2)
    
    # Store in XCom
    context['task_instance'].xcom_push(
        key='evaluation_summary',
        value=evaluation
    )
    
    logger.info(f"Model evaluation completed. Best model: {best_model_type} (AUC: {best_auc:.4f})")
    return evaluation


def deploy_model(**context):
    """Deploy the best model for inference."""
    import logging
    import shutil
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Deploying best model...")
    
    # Get best model info
    best_model_type = context['task_instance'].xcom_pull(
        task_ids='train_sklearn_models',
        key='best_model_type'
    )
    
    # Copy best model to production location
    from config.config import MODEL_DIR
    
    source_path = MODEL_DIR / f"{best_model_type}_model.pkl"
    production_path = MODEL_DIR / "production_model.pkl"
    
    if source_path.exists():
        shutil.copy2(source_path, production_path)
        logger.info(f"Model {best_model_type} deployed to {production_path}")
        
        # Also copy preprocessor
        preprocessor_source = MODEL_DIR / "airflow_preprocessor.pkl"
        preprocessor_prod = MODEL_DIR / "production_preprocessor.pkl"
        
        if preprocessor_source.exists():
            shutil.copy2(preprocessor_source, preprocessor_prod)
            logger.info("Preprocessor deployed")
        
        # Create deployment info
        deployment_info = {
            'model_type': best_model_type,
            'deployment_date': datetime.now().isoformat(),
            'model_path': str(production_path),
            'preprocessor_path': str(preprocessor_prod)
        }
        
        # Save deployment info
        from config.config import LOGS_DIR
        import json
        
        deploy_info_path = LOGS_DIR / "deployment_info.json"
        with open(deploy_info_path, 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        context['task_instance'].xcom_push(
            key='deployment_info',
            value=deployment_info
        )
        
        return deployment_info
    
    else:
        raise FileNotFoundError(f"Best model not found: {source_path}")


def test_inference(**context):
    """Test inference pipeline with deployed model."""
    import logging
    import pandas as pd
    import numpy as np
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Testing inference pipeline...")
    
    # Get deployment info
    deployment_info = context['task_instance'].xcom_pull(
        task_ids='deploy_model',
        key='deployment_info'
    )
    
    # Create test data (sample from original data)
    data_loader = DataLoader()
    df = data_loader.load_raw_data()
    
    # Take a small sample for testing
    test_sample = df.sample(n=10, random_state=42)
    
    # Remove target column for inference
    if 'Churn' in test_sample.columns:
        test_sample = test_sample.drop(columns=['Churn'])
    
    # Initialize inference pipeline
    inference_pipeline = InferencePipeline(
        model_path=deployment_info['model_path'],
        preprocessor_path=deployment_info['preprocessor_path']
    )
    
    # Make predictions
    results = inference_pipeline.predict(
        test_sample,
        return_probabilities=True
    )
    
    # Log results
    logger.info(f"Inference test completed:")
    logger.info(f"Predictions: {results['predictions']}")
    logger.info(f"Predicted churn rate: {results['statistics']['predicted_churn_rate']:.2%}")
    
    # Store test results
    context['task_instance'].xcom_push(
        key='inference_test_results',
        value=results['statistics']
    )
    
    return results['statistics']


def send_notification(**context):
    """Send notification about pipeline completion."""
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Sending pipeline completion notification...")
    
    # Get pipeline results
    evaluation = context['task_instance'].xcom_pull(
        task_ids='evaluate_models',
        key='evaluation_summary'
    )
    
    deployment_info = context['task_instance'].xcom_pull(
        task_ids='deploy_model',
        key='deployment_info'
    )
    
    inference_test = context['task_instance'].xcom_pull(
        task_ids='test_inference',
        key='inference_test_results'
    )
    
    # Create notification message
    message = f"""
    Telco Churn ML Pipeline Completed Successfully!
    
    Execution Date: {context['execution_date']}
    Best Model: {evaluation['best_model']['type']} (AUC: {evaluation['best_model']['auc']:.4f})
    Deployment Date: {deployment_info['deployment_date']}
    
    Inference Test Results:
    - Predicted Churn Rate: {inference_test['predicted_churn_rate']:.2%}
    - Total Predictions: {inference_test['total_predictions']}
    
    Pipeline Status: SUCCESS
    """
    
    logger.info(message)
    
    # Here you could add actual notification logic:
    # - Send email
    # - Post to Slack
    # - Update dashboard
    # - etc.
    
    return "Notification sent successfully"


# Define tasks
start_task = DummyOperator(
    task_id='start_pipeline',
    dag=dag
)

setup_task = PythonOperator(
    task_id='setup_environment',
    python_callable=setup_environment,
    dag=dag
)

# Data validation sensor (wait for data file)
data_sensor = FileSensor(
    task_id='wait_for_data',
    filepath='data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv',
    fs_conn_id='fs_default',
    poke_interval=30,
    timeout=300,
    dag=dag
)

validate_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    dag=dag
)

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag
)

# Model training task group
with TaskGroup('model_training', dag=dag) as training_group:
    sklearn_task = PythonOperator(
        task_id='train_sklearn_models',
        python_callable=train_sklearn_models
    )
    
    spark_task = PythonOperator(
        task_id='train_spark_models',
        python_callable=train_spark_models
    )

evaluate_task = PythonOperator(
    task_id='evaluate_models',
    python_callable=evaluate_models,
    dag=dag
)

deploy_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag
)

test_task = PythonOperator(
    task_id='test_inference',
    python_callable=test_inference,
    dag=dag
)

notify_task = PythonOperator(
    task_id='send_notification',
    python_callable=send_notification,
    dag=dag
)

end_task = DummyOperator(
    task_id='end_pipeline',
    dag=dag
)

# Define task dependencies
start_task >> setup_task >> data_sensor >> validate_task >> preprocess_task
preprocess_task >> training_group >> evaluate_task >> deploy_task >> test_task >> notify_task >> end_task

# Alternative failure handling
failure_task = BashOperator(
    task_id='handle_failure',
    bash_command='echo "Pipeline failed. Check logs for details."',
    trigger_rule='one_failed',
    dag=dag
)

# Connect failure handling
[validate_task, preprocess_task, training_group, evaluate_task, deploy_task, test_task] >> failure_task