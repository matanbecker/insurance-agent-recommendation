import pandas as pd
import numpy as np


def generate_synthetic_data(n_samples=10000, seed=42):
    """
    Generates synthetic insurance renewal data.

    Args:
        n_samples (int): Number of samples to generate.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Generated synthetic dataset.
    """
    np.random.seed(seed)
    industries = ['Healthcare', 'Retail', 'Technology', 'Construction', 'Education']
    locations = ['NY', 'CA', 'TX', 'FL', 'IL']
    sizes = ['Small', 'Medium', 'Large']
    policy_types = ['General Liability', 'Property', 'Workers Compensation', 'Auto', 'Cyber']

    data = {
        'Business_Name': [f"Business_{i}" for i in range(n_samples)],
        'Industry': np.random.choice(industries, n_samples),
        'Location': np.random.choice(locations, n_samples),
        'Size': np.random.choice(sizes, n_samples),
        'Policy_Type': np.random.choice(policy_types, n_samples),
        'Premium_Amount': np.random.uniform(500, 50000, n_samples),
        'Policy_Duration_Years': np.random.randint(1, 5, n_samples),
        'Renewal_Date': pd.date_range(start='2023-01-01', periods=n_samples, freq='D'),
        'Num_Contacts': np.random.randint(1, 10, n_samples),
        'Call_Duration_Minutes': np.random.uniform(2, 50, n_samples),
        'Last_Interaction_Outcome': np.random.choice(['Positive', 'Neutral', 'Negative'], n_samples),
        'Claims_Filed': np.random.randint(0, 10, n_samples),
        'Customer_Satisfaction_Score': np.random.uniform(1, 5, n_samples),
        'Discount_Availed': np.random.choice([0, 5, 10, 15, 20], n_samples),
        'Economic_Trend_Index': np.random.uniform(80, 120, n_samples),
        'Market_Competitiveness': np.random.uniform(1, 10, n_samples),
        'Renewal_Status': np.random.choice([1, 0], n_samples, p=[0.7, 0.3])
    }
    return pd.DataFrame(data)


if __name__ == "__main__":
    df = generate_synthetic_data()
    df.to_csv('../data/synthetic_renewal_data.csv', index=False)
    print("Synthetic data generated and saved to data/synthetic_renewal_data.csv")
