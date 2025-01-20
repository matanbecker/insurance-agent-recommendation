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
import training
from training import train_model

openai.api_key = "INPUT OPEN API KEY HERE"

# Define features and target variable
num_features = [
    'Premium_Amount', 'Policy_Duration_Years', 'Num_Contacts',
    'Call_Duration_Minutes', 'Claims_Filed', 'Customer_Satisfaction_Score',
    'Discount_Availed', 'Economic_Trend_Index', 'Market_Competitiveness', 'Time_to_Renewal'
]
cat_features = ['Industry', 'Location', 'Size', 'Policy_Type']

train_model('C:/Users/MATAN/git/renewals_engine/data/synthetic_renewal_data.csv', 'C:/Users/MATAN/git/renewals_engine/models/renewal_model.pkl')

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

# # Flask app for the unified endpoint
# app = Flask(__name__)

@app.route('/unified-process', methods=['POST'])
def unified_endpoint():
    try:
        data = request.get_json()

        # Prepare input data
        input_features = {key: [data[key]] for key in num_features + cat_features}
        input_df = pd.DataFrame.from_dict(input_features)

        # Load model and predict renewal probability
        model = joblib.load('C:/Users/MATAN/git/renewals_engine/models/renewal_model.pkl')
        prediction_probability = model.predict(input_df)[0] * 100  # Convert to percentage

        # Generate GPT prompts for customer questions
        customer_info = {key: data[key] for key in num_features + cat_features}
        customer_questions_prompt, questions = generate_customer_questions_and_answers(customer_info, prediction_probability)

        # GPT-4 call for answering questions
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are an expert in customer renewals."},
                      {"role": "user", "content": customer_questions_prompt}]
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
            messages=[{"role": "system", "content": "You are an expert in customer retention."},
                      {"role": "user", "content": recommendation_prompt}]
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
            ("questions", questions),  # Included questions in the response
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