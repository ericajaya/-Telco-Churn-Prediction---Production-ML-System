# -Telco-Churn-Prediction---Production-ML-System
A comprehensive production-ready machine learning system for predicting customer churn in the telecommunications industry. This project demonstrates advanced MLOps practices with complete model pipelines, experiment tracking, distributed computing, and workflow orchestration.
  Telco Churn Prediction - Production ML System

A comprehensive production-ready machine learning system for predicting customer churn in the telecommunications industry. This project demonstrates advanced MLOps practices with complete model pipelines, experiment tracking, distributed computing, and workflow orchestration.

    Project Overview

This project transforms a basic churn prediction model into a production-ready ML system featuring:

    Architecture

```
telco_churn_ml_system/
├── 📁 config/                   Configuration management
├── 📁 src/                      Core source code
│   ├── 📁 data/                Data loading and validation
│   ├── 📁 preprocessing/       Feature engineering and preprocessing
│   ├── 📁 models/             Model implementations
│   └── 📁 utils/              Utility functions
├── 📁 pipelines/               ML pipeline implementations
│   ├── training_pipeline.py       Training with MLflow tracking
│   ├── inference_pipeline.py      Production inference system
│   └── spark_pipeline.py          Distributed computing pipeline
├── 📁 dags/                    Airflow workflow definitions
├── 📁 tests/                   Comprehensive test suite
├── 📁 notebooks/               Jupyter notebooks for exploration
├── 📁 models/                  Saved model artifacts
└── 📁 data/                    Data storage (raw, processed, predictions)
```


   Setup and Installation

  Prerequisites
- Python 3.8+
- Apache Spark 3.4+
- Docker (optional, for MLflow and Airflow)

  1. Environment Setup
```bash
  Create virtual environment
python -m venv telco_churn_env
source telco_churn_env/bin/activate    On Windows: telco_churn_env\Scripts\activate

  Install dependencies
pip install -r requirements.txt
```

  2. Data Setup
```bash
  Create directory structure
python -c "from config.config import create_directories; create_directories()"

  Download dataset from Kaggle
  Place WA_Fn-UseC_-Telco-Customer-Churn.csv in data/raw/
```

  3. MLflow Setup
```bash
  Start MLflow tracking server
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000
```

  4. Airflow Setup (Optional)
```bash
  Initialize Airflow
export AIRFLOW_HOME=$(pwd)/airflow
airflow db init
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

  Start Airflow
airflow webserver --port 8080 &
airflow scheduler
```

   

    Configuration

  Environment Variables
```bash
  MLflow configuration
export MLFLOW_TRACKING_URI="http://localhost:5000"
export MLFLOW_EXPERIMENT_NAME="telco_churn_prediction"

  Spark configuration
export SPARK_HOME="/path/to/spark"
export PYSPARK_PYTHON="python"
```

  Model Configuration
Edit `config/config.py` to customize:
- Hyperparameter grids for model tuning
- Feature engineering parameters
- Data validation rules
- Spark cluster settings

    Testing Strategy

  Running Specific Test Types
```bash
  Unit tests only
python -m pytest tests/ -m "not integration and not performance"

  Integration tests
python -m pytest tests/ -m "integration"

  Performance tests
python -m pytest tests/ -m "performance"
```

   

    Deployment Options

  Local Development
```bash
  Start all services locally
docker-compose up -d    MLflow, Spark, Airflow (if using Docker)
```

  Production Deployment



 

    Requirements

  Core Dependencies
-   scikit-learn  : Machine learning algorithms
-   pandas  : Data manipulation and analysis
-   numpy  : Numerical computing
-   mlflow  : Experiment tracking and model registry
-   pyspark  : Distributed computing
-   apache-airflow  : Workflow orchestration

  Development Dependencies
-   pytest  : Testing framework
-   black  : Code formatting
-   flake8  : Code linting
-   jupyter  : Interactive development

    

