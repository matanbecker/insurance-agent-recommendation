import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib


def train_model(input_path, output_path):
    """
    Trains a RandomForest model on renewal data.

    Args:
        input_path (str): Path to input CSV file.
        output_path (str): Path to save trained model.
    """
    df = pd.read_csv(input_path)
    # Derived feature: Calculate Time_to_Renewal in days
    df['Renewal_Date'] = pd.to_datetime(df['Renewal_Date'])
    df['Time_to_Renewal'] = (df['Renewal_Date'] - pd.Timestamp.now()).apply(lambda x: x.days)

    # Define features and target variable
    num_features = [
        'Premium_Amount', 'Policy_Duration_Years', 'Num_Contacts',
        'Call_Duration_Minutes', 'Claims_Filed', 'Customer_Satisfaction_Score',
        'Discount_Availed', 'Economic_Trend_Index', 'Market_Competitiveness', 'Time_to_Renewal'
    ]
    cat_features = ['Industry', 'Location', 'Size', 'Policy_Type']

    X = df[num_features + cat_features]
    y = df['Renewal_Status']

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ]
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Regressor for percentage prediction
    renewal_model = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    renewal_model.fit(X_train, y_train)

    joblib.dump(renewal_model, output_path)
    print(f"Model trained and saved to {output_path}")


if __name__ == "__main__":
    train_model('C:/Users/MATAN/git/renewals_engine/data/synthetic_renewal_data.csv', 'C:/Users/MATAN/git/renewals_engine/models/renewal_model.pkl')