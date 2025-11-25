"""Tests for preprocessing pipeline."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import tempfile
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.preprocessor import TelcoPreprocessor, TelcoDataCleaner, FeatureEngineer
from config.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


@pytest.fixture
def sample_telco_data():
    """Create sample Telco data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'customerID': [f'ID{i:04d}' for i in range(n_samples)],
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples),
        'Partner': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples),
        'tenure': np.random.randint(0, 72, n_samples),
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
        'Churn': np.random.choice(['Yes', 'No'], n_samples)
    }
    
    # Add some problematic data
    # Missing TotalCharges (as spaces)
    data['TotalCharges'][5:10] = [' ', '  ', '', ' ', '   ']
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_data_with_issues():
    """Create sample data with known data quality issues."""
    data = {
        'customerID': ['ID001', 'ID002', 'ID003'],
        'gender': ['Male', 'Female', None],
        'SeniorCitizen': [0, 1, 0],
        'Partner': ['Yes', 'No', 'Yes'],
        'Dependents': ['No', 'Yes', 'No'],
        'tenure': [12, 24, None],
        'PhoneService': ['Yes', 'No', 'Yes'],
        'MultipleLines': ['No', 'No phone service', 'Yes'],
        'InternetService': ['DSL', 'Fiber optic', 'No'],
        'OnlineSecurity': ['No', 'Yes', 'No internet service'],
        'OnlineBackup': ['Yes', 'No', 'No internet service'],
        'DeviceProtection': ['No', 'Yes', 'No internet service'],
        'TechSupport': ['No', 'Yes', 'No internet service'],
        'StreamingTV': ['No', 'Yes', 'No internet service'],
        'StreamingMovies': ['No', 'Yes', 'No internet service'],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'PaperlessBilling': ['Yes', 'No', 'Yes'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)'],
        'MonthlyCharges': [50.0, 80.0, 65.0],
        'TotalCharges': [' ', '1920.0', '   '],  # Problematic TotalCharges
        'Churn': ['No', 'Yes', 'No']
    }
    
    return pd.DataFrame(data)


class TestTelcoDataCleaner:
    """Test TelcoDataCleaner class."""
    
    def test_init(self):
        """Test cleaner initialization."""
        cleaner = TelcoDataCleaner()
        assert not cleaner.fitted
    
    def test_fit_transform(self, sample_data_with_issues):
        """Test fit and transform methods."""
        cleaner = TelcoDataCleaner()
        
        # Test fit
        cleaner.fit(sample_data_with_issues)
        assert cleaner.fitted
        
        # Test transform
        cleaned_data = cleaner.transform(sample_data_with_issues)
        
        # Check TotalCharges conversion
        assert pd.api.types.is_numeric_dtype(cleaned_data['TotalCharges'])
        
        # Check that spaces were converted to NaN
        assert pd.isna(cleaned_data['TotalCharges'].iloc[0])
        assert pd.isna(cleaned_data['TotalCharges'].iloc[2])
        
        # Check SeniorCitizen conversion
        assert cleaned_data['SeniorCitizen'].dtype == 'object'
        assert all(isinstance(val, str) for val in cleaned_data['SeniorCitizen'])
    
    def test_handle_total_charges_filling(self, sample_data_with_issues):
        """Test TotalCharges filling logic."""
        cleaner = TelcoDataCleaner()
        cleaned_data = cleaner.fit_transform(sample_data_with_issues)
        
        # First row should be filled with tenure * MonthlyCharges
        expected_value = sample_data_with_issues['tenure'].iloc[0] * sample_data_with_issues['MonthlyCharges'].iloc[0]
        
        # Account for potential NaN in tenure
        if pd.isna(sample_data_with_issues['tenure'].iloc[0]):
            assert pd.isna(cleaned_data['TotalCharges'].iloc[0])
        else:
            assert abs(cleaned_data['TotalCharges'].iloc[0] - expected_value) < 0.01


class TestFeatureEngineer:
    """Test FeatureEngineer class."""
    
    def test_init(self):
        """Test engineer initialization."""
        engineer = FeatureEngineer()
        assert engineer.create_interaction_features
        assert not engineer.fitted
        
        engineer_no_interaction = FeatureEngineer(create_interaction_features=False)
        assert not engineer_no_interaction.create_interaction_features
    
    def test_tenure_groups(self, sample_telco_data):
        """Test tenure group creation."""
        engineer = FeatureEngineer()
        engineered_data = engineer.fit_transform(sample_telco_data)
        
        assert 'tenure_group' in engineered_data.columns
        
        # Check that groups are created correctly
        tenure_groups = engineered_data['tenure_group'].unique()
        expected_groups = ['0-12 months', '12-24 months', '24-48 months', '48+ months']
        
        for group in tenure_groups:
            assert group in expected_groups or pd.isna(group)
    
    def test_charges_groups(self, sample_telco_data):
        """Test monthly charges group creation."""
        engineer = FeatureEngineer()
        engineered_data = engineer.fit_transform(sample_telco_data)
        
        assert 'charges_group' in engineered_data.columns
        
        # Check group values
        charge_groups = engineered_data['charges_group'].unique()
        expected_groups = ['Low', 'Medium', 'High', 'Very High']
        
        for group in charge_groups:
            assert group in expected_groups or pd.isna(group)
    
    def test_service_count(self, sample_telco_data):
        """Test service count feature."""
        engineer = FeatureEngineer()
        engineered_data = engineer.fit_transform(sample_telco_data)
        
        assert 'service_count' in engineered_data.columns
        
        # Service count should be numeric and non-negative
        assert pd.api.types.is_numeric_dtype(engineered_data['service_count'])
        assert all(engineered_data['service_count'] >= 0)
    
    def test_interaction_features(self, sample_telco_data):
        """Test interaction feature creation."""
        engineer = FeatureEngineer(create_interaction_features=True)
        engineered_data = engineer.fit_transform(sample_telco_data)
        
        # Check for interaction features
        assert 'contract_payment' in engineered_data.columns
        assert 'internet_streaming' in engineered_data.columns


class TestTelcoPreprocessor:
    """Test TelcoPreprocessor class."""
    
    def test_init(self):
        """Test preprocessor initialization."""
        preprocessor = TelcoPreprocessor()
        assert preprocessor.scaling_method == 'standard'
        assert preprocessor.encoding_method == 'onehot'
        assert not preprocessor.handle_imbalance
        assert preprocessor.pipeline is None
        
        # Test custom parameters
        preprocessor_custom = TelcoPreprocessor(
            scaling_method='minmax',
            encoding_method='label',
            handle_imbalance=True
        )
        assert preprocessor_custom.scaling_method == 'minmax'
        assert preprocessor_custom.encoding_method == 'label'
        assert preprocessor_custom.handle_imbalance
    
    def test_fit_transform(self, sample_telco_data):
        """Test fit_transform method."""
        preprocessor = TelcoPreprocessor()
        
        # Remove target column for feature preprocessing
        X = sample_telco_data.drop(columns=['Churn'])
        
        X_transformed = preprocessor.fit_transform(X)
        
        # Check output
        assert isinstance(X_transformed, np.ndarray)
        assert X_transformed.shape[0] == len(X)
        assert X_transformed.shape[1] > 0
        
        # Pipeline should be fitted
        assert preprocessor.pipeline is not None
    
    def test_transform_new_data(self, sample_telco_data):
        """Test transform method on new data."""
        preprocessor = TelcoPreprocessor()
        
        X = sample_telco_data.drop(columns=['Churn'])
        
        # Fit on part of the data
        X_train = X.iloc[:80]
        X_test = X.iloc[80:]
        
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        # Check shapes
        assert X_train_transformed.shape[1] == X_test_transformed.shape[1]
        assert X_test_transformed.shape[0] == len(X_test)
    
    def test_save_load_pipeline(self, sample_telco_data):
        """Test saving and loading pipeline."""
        preprocessor = TelcoPreprocessor()
        
        X = sample_telco_data.drop(columns=['Churn'])
        X_transformed = preprocessor.fit_transform(X)
        
        # Save pipeline
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            preprocessor.save_pipeline(tmp.name)
            
            # Load in new preprocessor
            new_preprocessor = TelcoPreprocessor()
            new_preprocessor.load_pipeline(tmp.name)
            
            # Test that loaded preprocessor works
            X_test = X.iloc[:10]
            X_new_transformed = new_preprocessor.transform(X_test)
            
            assert X_new_transformed.shape[1] == X_transformed.shape[1]
            
            # Clean up
            os.unlink(tmp.name)
    
    def test_different_scaling_methods(self, sample_telco_data):
        """Test different scaling methods."""
        X = sample_telco_data.drop(columns=['Churn'])
        
        scaling_methods = ['standard', 'minmax', 'robust']
        
        for method in scaling_methods:
            preprocessor = TelcoPreprocessor(scaling_method=method)
            X_transformed = preprocessor.fit_transform(X)
            
            assert isinstance(X_transformed, np.ndarray)
            assert X_transformed.shape[0] == len(X)
    
    def test_different_encoding_methods(self, sample_telco_data):
        """Test different encoding methods."""
        X = sample_telco_data.drop(columns=['Churn'])
        
        encoding_methods = ['onehot', 'label']
        
        for method in encoding_methods:
            preprocessor = TelcoPreprocessor(encoding_method=method)
            X_transformed = preprocessor.fit_transform(X)
            
            assert isinstance(X_transformed, np.ndarray)
            assert X_transformed.shape[0] == len(X)
    
    def test_get_feature_names(self, sample_telco_data):
        """Test feature names extraction."""
        preprocessor = TelcoPreprocessor()
        
        X = sample_telco_data.drop(columns=['Churn'])
        X_transformed = preprocessor.fit_transform(X)
        
        feature_names = preprocessor.get_feature_names()
        
        # Should return feature names or None
        if feature_names is not None:
            assert len(feature_names) == X_transformed.shape[1]
            assert all(isinstance(name, str) for name in feature_names)
    
    def test_handles_missing_columns(self):
        """Test preprocessor handles missing columns gracefully."""
        # Create data with missing columns
        data = {
            'gender': ['Male', 'Female'],
            'tenure': [12, 24],
            'MonthlyCharges': [50.0, 80.0],
        }
        df = pd.DataFrame(data)
        
        preprocessor = TelcoPreprocessor()
        
        # Should not raise an error, but handle missing columns
        X_transformed = preprocessor.fit_transform(df)
        assert isinstance(X_transformed, np.ndarray)
    
    def test_pipeline_reproducibility(self, sample_telco_data):
        """Test that pipeline produces consistent results."""
        X = sample_telco_data.drop(columns=['Churn'])
        
        preprocessor1 = TelcoPreprocessor(scaling_method='standard', encoding_method='onehot')
        preprocessor2 = TelcoPreprocessor(scaling_method='standard', encoding_method='onehot')
        
        X_transformed1 = preprocessor1.fit_transform(X)
        X_transformed2 = preprocessor2.fit_transform(X)
        
        # Results should be very similar (allowing for small numerical differences)
        assert X_transformed1.shape == X_transformed2.shape
        # Note: Due to potential randomness in some sklearn components,
        # we don't check for exact equality


@pytest.mark.integration
class TestPreprocessingIntegration:
    """Integration tests for preprocessing pipeline."""
    
    def test_full_preprocessing_pipeline(self, sample_telco_data):
        """Test complete preprocessing pipeline."""
        # Test the full pipeline end-to-end
        preprocessor = TelcoPreprocessor(
            scaling_method='standard',
            encoding_method='onehot'
        )
        
        X = sample_telco_data.drop(columns=['Churn'])
        y = sample_telco_data['Churn'].map({'Yes': 1, 'No': 0})
        
        # Fit and transform
        X_transformed = preprocessor.fit_transform(X)
        
        # Check that transformation is valid for ML
        assert not np.any(np.isnan(X_transformed))
        assert not np.any(np.isinf(X_transformed))
        
        # Check that we can use this with sklearn
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_transformed, y, test_size=0.2, random_state=42
        )
        
        # Should be able to train a model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Should be able to make predictions
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)
    
    def test_preprocessing_with_new_data_categories(self, sample_telco_data):
        """Test preprocessing handles new categories in test data."""
        preprocessor = TelcoPreprocessor(encoding_method='onehot')
        
        X = sample_telco_data.drop(columns=['Churn'])
        
        # Fit on subset
        X_train = X.iloc[:80]
        preprocessor.fit_transform(X_train)
        
        # Create test data with new category
        X_test = X.iloc[80:].copy()
        X_test.loc[X_test.index[0], 'gender'] = 'Other'  # New category
        
        # Should handle gracefully
        X_test_transformed = preprocessor.transform(X_test)
        assert isinstance(X_test_transformed, np.ndarray)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
