# Updates include:
# - Renewal prediction as percentage
# - GPT prompt engineering for dynamic customer understanding
# - Consolidated prompts for efficient interactions

from flask import Flask, request, jsonify
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import numpy as np
from collections import OrderedDict
import joblib
import openai

openai.api_key = "INPUT OPEN API KEY HERE"

# Load data
df = pd.read_csv('synthetic_renewal_data.csv')

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

# Save the trained model
joblib.dump(renewal_model, 'renewal_model.pkl')

# Helper function to generate customer-specific GPT prompts and capture answers
def generate_customer_questions_and_answers(customer_info, renewal_probability):
    # List of concise questions
    questions = [
        f"1. What factors impact renewal in {customer_info['Industry']}?",
        f"2. How does recent claims history influence renewal likelihood?",
        f"3. Are there regional trends that impact renewal in {customer_info['Location']}?",
        f"4. How does policy duration affect renewal in {customer_info['Industry']}?",
        f"5. What renewal challenges are typical for customers of size {customer_info['Size']}?",
        f"6. How does customer satisfaction impact renewal decisions?",
        f"7. Are there upselling opportunities based on {customer_info['Premium_Amount']} premium?",
        f"8. How does market competitiveness affect retention in {customer_info['Location']}?",
        f"9. What strategies can increase renewal chances for this customer?",
        f"10. What actions can improve renewal probability in the next {customer_info['Time_to_Renewal']} days?"
    ]
    
    # Generate prompt for GPT to answer the questions
    question_list = "\n".join(questions)
    customer_profile = f"""
    Industry: {customer_info['Industry']}
    Location: {customer_info['Location']}
    Size: {customer_info['Size']}
    Policy Type: {customer_info['Policy_Type']}
    Premium Amount: {customer_info['Premium_Amount']}
    Claims Filed: {customer_info['Claims_Filed']}
    Satisfaction: {customer_info['Customer_Satisfaction_Score']}
    Time to Renewal: {customer_info['Time_to_Renewal']} days
    Renewal Probability: {renewal_probability:.2f}%
    """

    prompt = f"""
    Customer Profile:
    {customer_profile}

    Questions:
    {question_list}

    Please provide concise and actionable answers to the above questions.
    """
    
    return prompt, questions

# Flask app for the unified endpoint
app = Flask(__name__)

# Flask app for the unified endpoint
app = Flask(__name__)

@app.route('/unified-process', methods=['POST'])
def unified_endpoint():
    try:
        data = request.get_json()

        # Prepare input data
        input_features = {key: [data[key]] for key in num_features + cat_features}
        input_df = pd.DataFrame.from_dict(input_features)

        # Load model and predict renewal probability
        model = joblib.load('renewal_model.pkl')
        prediction_probability = model.predict(input_df)[0] * 100  # Convert to percentage

        # Generate GPT prompts for customer questions
        customer_info = {key: data[key] for key in num_features + cat_features}
        customer_questions_prompt, questions = generate_customer_questions_and_answers(customer_info, prediction_probability)

        # GPT-4 call for answering questions
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in customer renewals."},
                {"role": "user", "content": customer_questions_prompt}
            ]
        )
        answers = response['choices'][0]['message']['content']

        # GPT call for final recommendation based on answers and customer profile
        recommendation_prompt = f"""
        Answers to the Questions:
        {answers}

        Customer Profile:
        Industry: {customer_info['Industry']}
        Location: {customer_info['Location']}
        Size: {customer_info['Size']}
        Policy Type: {customer_info['Policy_Type']}
        Premium Amount: {customer_info['Premium_Amount']}
        Claims Filed: {customer_info['Claims_Filed']}
        Satisfaction: {customer_info['Customer_Satisfaction_Score']}
        Time to Renewal: {customer_info['Time_to_Renewal']} days
        Renewal Probability: {prediction_probability:.2f}%

        Provide concise, actionable strategies to maximize renewal likelihood for this customer. Format the recommendations as bullet points.
        """
        recommendation_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in customer retention."},
                {"role": "user", "content": recommendation_prompt}
            ]
        )
        recommendations = recommendation_response['choices'][0]['message']['content']

        structured_response = OrderedDict([
            ("message", [
                "This response contains details about the customer's profile, renewal probability, answers to key questions, and recommendations for improvement."
            ]),
            ("renewal_probability", [f"{prediction_probability:.2f}%"]),  # Wrapped in a list for uniform rendering
            ("customer_profile", [
                f"Industry: {customer_info.get('Industry', 'N/A')}",
                f"Location: {customer_info.get('Location', 'N/A')}",
                f"Size: {customer_info.get('Size', 'N/A')}",
                f"Policy Type: {customer_info.get('Policy_Type', 'N/A')}",
                f"Premium Amount: ${customer_info.get('Premium_Amount', 0):,}",
                f"Claims Filed: {customer_info.get('Claims_Filed', 'N/A')}",
                f"Customer Satisfaction Score: {customer_info.get('Customer_Satisfaction_Score', 'N/A')}",
                f"Time to Renewal (days): {customer_info.get('Time_to_Renewal', 'N/A')}"
            ]),
            ("answers_to_questions", [
                answer.strip() for answer in answers.split("\n") if answer.strip()
            ]),  # Split GPT answers into dynamic bullet points, ignoring empty lines
            ("recommendations", [
                recommendation.strip() for recommendation in recommendations.split("\n") if recommendation.strip()
            ])  # Split recommendations into dynamic bullet points, ignoring empty lines
        ])
        # Return structured JSON response
        return jsonify(structured_response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)





# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# from sklearn.metrics.pairwise import cosine_similarity
# import joblib
# import openai
# from flask import Flask, request, jsonify


# # Load data
# df = pd.read_csv('synthetic_renewal_data.csv')

# # Derived feature: Calculate Time_to_Renewal in days
# df['Renewal_Date'] = pd.to_datetime(df['Renewal_Date'])
# df['Time_to_Renewal'] = (df['Renewal_Date'] - pd.Timestamp.now()).apply(lambda x: x.days)

# # Define features and target variable
# num_features = [
#     'Premium_Amount', 'Policy_Duration_Years', 'Num_Contacts',
#     'Call_Duration_Minutes', 'Claims_Filed', 'Customer_Satisfaction_Score',
#     'Discount_Availed', 'Economic_Trend_Index', 'Market_Competitiveness', 'Time_to_Renewal'
# ]
# cat_features = ['Industry', 'Location', 'Size', 'Policy_Type']

# # Selecting only the necessary columns from the original dataframe
# X = df[num_features + cat_features]  # Features
# y = df['Renewal_Status']  # Target variable

# # Custom transformer for categorical features
# class CustomTransformer:
#     def __init__(self):
#         self.label_encoders = {}

#     def fit(self, X, y=None):
#         for col in cat_features:
#             self.label_encoders[col] = LabelEncoder().fit(X[col])
#         return self

#     def transform(self, X):
#         X_transformed = X.copy()
#         for col, encoder in self.label_encoders.items():
#             X_transformed[col] = encoder.transform(X[col])
#         return X_transformed

#     def fit_transform(self, X, y=None):
#         return self.fit(X, y).transform(X)

#     def get_params(self, deep=True):
#         return {}

#     def set_params(self, **params):
#         return self

# # Preprocessing pipeline
# preprocessor = ColumnTransformer(
#     transformers=[ 
#         ('num', StandardScaler(), num_features),
#         ('cat', CustomTransformer(), cat_features)
#     ]
# )

# # Split data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train a model to learn feature importance
# feature_model = RandomForestRegressor(n_estimators=100, random_state=42)
# X_train_transformed = preprocessor.fit_transform(X_train)
# feature_model.fit(X_train_transformed, y_train)

# # Get feature importances (used as dynamic weights)
# feature_importances = feature_model.feature_importances_

# # Normalize feature importances to get weights between 0 and 1
# normalized_weights = feature_importances / np.sum(feature_importances)

# # Create a mapping of features to their respective weights
# feature_weight_map = {feature: weight for feature, weight in zip(num_features + cat_features, normalized_weights)}

# # Function to recommend similar businesses with dynamic ML-based weighting
# def recommend_similar_businesses(business_id, df, top_n=5):
#     # Preprocess the features (numeric and categorical)
#     features = df[num_features + cat_features]
#     transformed_features = preprocessor.transform(features)
    
#     # Apply dynamic weights to the features based on the trained model
#     weighted_features = transformed_features * np.array([feature_weight_map[feature] for feature in num_features + cat_features])
    
#     # Compute cosine similarity between the target business and all other businesses
#     similarities = cosine_similarity(weighted_features)
    
#     # Sort the businesses by similarity, excluding the business itself
#     similar_indices = similarities[business_id].argsort()[-top_n-1:-1][::-1]
    
#     return df.iloc[similar_indices]

# # Unified function to process predictions, summaries, and recommendations
# def unified_process(data):
#     # Load model
#     model = joblib.load('renewal_model.pkl')
    
#     # Prepare input data
#     input_features = {key: [data[key]] for key in num_features + cat_features}
#     input_df = pd.DataFrame.from_dict(input_features)
    
#     # Generate prediction
#     prediction = int(model.predict(input_df)[0])  # Convert to Python int
    
#     # Generate top 5 similar businesses summary
#     business_id = data.get("business_id", 0)  # Default to 0 if not provided
#     top_n = data.get("top_n", 5)
#     similar_businesses = recommend_similar_businesses(business_id, df, top_n)
#     similar_businesses_summary = similar_businesses[[ 
#         'Industry', 'Location', 'Size', 'Policy_Type',
#         'Customer_Satisfaction_Score', 'Claims_Filed',
#         'Time_to_Renewal', 'Market_Competitiveness'
#     ]].to_dict(orient='records')

#     # Extract customer information to help GPT understand the customer context
#     customer_info = {key: data[key] for key in num_features + cat_features}
    
#     # Generate agent recommendations using LLM
#     prompt = f"""
#     You are an AI assistant helping an insurance agent optimize policy renewals. 
#     First, you need to understand the customer:
    
#     Customer Profile:
#     Industry: {customer_info['Industry']}
#     Location: {customer_info['Location']}
#     Size: {customer_info['Size']}
#     Policy Type: {customer_info['Policy_Type']}
#     Premium Amount: {customer_info['Premium_Amount']}
#     Claims Filed: {customer_info['Claims_Filed']}
#     Customer Satisfaction Score: {customer_info['Customer_Satisfaction_Score']}
#     Time to Renewal: {customer_info['Time_to_Renewal']}
#     Market Competitiveness: {customer_info['Market_Competitiveness']}

#     Now that you understand the customer, here are the top 5 similar businesses and their renewal prediction:
#     {similar_businesses_summary}
    
#     Renewal Prediction: {prediction} (1 = renewal, 0 = no renewal)

#     Based on this, provide actionable recommendations to the agent on how to close this customer. 
#     Focus on strategies that align with the customer's industry, satisfaction score, claims history, and time to renewal. 
#     Suggest specific actions to increase the likelihood of closing the deal.
#     """
    
#     response = openai.ChatCompletion.create(
#         model="gpt-4",
#         messages=[ 
#             {"role": "system", "content": "You are an expert assistant in insurance renewals."},
#             {"role": "user", "content": prompt}
#         ]
#     )
#     recommendations = response['choices'][0]['message']['content']
    
#     # Return results
#     return {
#         "renewal_prediction": prediction,
#         "top_similar_businesses": similar_businesses_summary,
#         "agent_recommendations": recommendations
#     }

# # Flask app for the unified endpoint
# app = Flask(__name__)

# @app.route('/unified-process', methods=['POST'])
# def unified_endpoint():
#     try:
#         data = request.get_json()
#         result = unified_process(data)
#         return jsonify(result), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True)


