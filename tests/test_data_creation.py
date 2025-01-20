import pandas as pd
import pytest
from data_creation.generate_synthetic_data import generate_synthetic_data

def test_generate_synthetic_data():
    n_samples = 1000
    df = generate_synthetic_data(n_samples=n_samples)

    # Check the number of rows
    assert len(df) == n_samples, "The number of rows in the DataFrame is incorrect."

    # Check the column names
    expected_columns = [
        'Business_Name', 'Industry', 'Location', 'Size', 'Policy_Type',
        'Premium_Amount', 'Policy_Duration_Years', 'Renewal_Date', 
        'Num_Contacts', 'Call_Duration_Minutes', 'Last_Interaction_Outcome',
        'Claims_Filed', 'Customer_Satisfaction_Score', 'Discount_Availed',
        'Economic_Trend_Index', 'Market_Competitiveness', 'Renewal_Status'
    ]
    assert list(df.columns) == expected_columns, "The columns in the DataFrame do not match the expected schema."

    # Check data types
    assert pd.api.types.is_numeric_dtype(df['Premium_Amount']), "Premium_Amount should be numeric."
    assert pd.api.types.is_datetime64_any_dtype(df['Renewal_Date']), "Renewal_Date should be datetime."
    assert df['Renewal_Status'].isin([0, 1]).all(), "Renewal_Status should only contain 0 or 1."
    