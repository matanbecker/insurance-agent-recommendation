import pytest
from model.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_unified_endpoint(client):
    sample_input = {
        "Premium_Amount": 1500,
        "Policy_Duration_Years": 3,
        "Num_Contacts": 5,
        "Call_Duration_Minutes": 30,
        "Claims_Filed": 2,
        "Customer_Satisfaction_Score": 4.2,
        "Discount_Availed": 10,
        "Economic_Trend_Index": 100,
        "Market_Competitiveness": 8,
        "Time_to_Renewal": 45,
        "Industry": "Healthcare",
        "Location": "NY",
        "Size": "Medium",
        "Policy_Type": "General Liability"
    }

    response = client.post('/unified-process', json=sample_input)
    assert response.status_code == 200, "The endpoint did not return a success status."
    response_data = response.get_json()

    # Validate the structure of the response
    assert "renewal_probability" in response_data, "The response is missing 'renewal_probability'."
    assert "recommendations" in response_data, "The response is missing 'recommendations'."
