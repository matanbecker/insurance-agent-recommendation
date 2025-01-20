import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from model.app import preprocessor, renewal_model

@pytest.fixture
def sample_data():
    # Load a subset of synthetic data for testing
    return pd.read_csv('data_creation/synthetic_renewal_data.csv').sample(500)

def test_pipeline_fit(sample_data):
    num_features = [
        'Premium_Amount', 'Policy_Duration_Years', 'Num_Contacts',
        'Call_Duration_Minutes', 'Claims_Filed', 'Customer_Satisfaction_Score',
        'Discount_Availed', 'Economic_Trend_Index', 'Market_Competitiveness', 'Time_to_Renewal'
    ]
    cat_features = ['Industry', 'Location', 'Size', 'Policy_Type']
    
    X = sample_data[num_features + cat_features]
    y = sample_data['Renewal_Status']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    renewal_model.fit(X_train, y_train)
    
    # Test model
    score = renewal_model.score(X_test, y_test)
    assert score > 0.7, "The model's score is too low. Consider tuning hyperparameters."
