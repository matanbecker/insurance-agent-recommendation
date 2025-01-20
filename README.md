# Renewal Prediction API with GPT Integration

This repository provides an API that predicts customer renewal probability using machine learning models and enhances decision-making by using GPT to provide personalized strategies for improving renewal likelihood. The API is built using Flask and integrates a trained machine learning model for renewal prediction, along with GPT-4 for generating customer-specific prompts and recommendations.

## Features

- **Renewal Prediction**: The model predicts the probability of a customer renewing their policy based on input features like premium amount, policy duration, customer satisfaction, claims filed, and other relevant factors.
- **GPT Prompt Engineering**: The system dynamically generates questions related to customer renewal and uses GPT-4 to answer these questions.
- **Actionable Recommendations**: Based on the customer profile and GPT-generated answers, the system provides strategies to improve renewal chances.
- **Unified API**: A single endpoint integrates the model prediction and GPT answers, making it easy to access both insights and recommendations for a specific customer.

## Prerequisites
To run this API locally, you need to have the following installed:

- **Python 3.7+**
- **Flask**: A lightweight WSGI web application framework.
- **scikit-learn**: For the machine learning model.
- **openai**: For accessing GPT-4 API.
  
You can install the required Python packages using ```pip```:

```bash
pip install Flask scikit-learn openai pandas numpy joblib
```

Additionally, you will need access to GPT-4 via OpenAI's API. Make sure you have an API key from OpenAI.

## Setup Instructions

1. Clone this repository:
```bash
git clone <https://github.com/matanbecker/insurance-agent-recommendation>
```
Download the trained machine learning model renewal_model.pkl and place it in the models/ directory of the project.

2. Set up your OpenAI API key. You can set your API key as an environment variable:

```bash
export OPENAI_API_KEY='your-api-key'
```

3. Run the Flask application:
```bash
python app.py
```

This will start the API locally at http://127.0.0.1:5000.


## API Endpoint
```/unified-process```
#### Method: POST

This endpoint takes a customer's information and returns the predicted renewal probability, answers to customer-specific questions, and strategies for improving the renewal likelihood.

#### Request Body:
```json
{
  "Premium_Amount": 1000,
  "Policy_Duration_Years": 5,
  "Num_Contacts": 10,
  "Call_Duration_Minutes": 15,
  "Claims_Filed": 1,
  "Customer_Satisfaction_Score": 80,
  "Discount_Availed": 5,
  "Economic_Trend_Index": 7,
  "Market_Competitiveness": 3,
  "Time_to_Renewal": 30,
  "Industry": "Tech",
  "Location": "New York",
  "Size": "Medium",
  "Policy_Type": "Health"
}
```

#### Response:
The response is a JSON object containing:

* **message**: A brief message explaining the response.
* **renewal_probability**: The predicted probability of renewal as a percentage.
* **customer_profile**: The customer's profile including key details like premium amount, claims filed, etc.
* **answers_to_questions**: GPT-generated answers to the renewal-related questions.
* **recommendations**: GPT-generated actionable strategies to improve the renewal likelihood.

Example response:

```json
{
  "message": [
    "This response contains details about the customer's profile, renewal probability, answers to key questions, and recommendations for improvement."
  ],
  "renewal_probability": ["85.00%"],
  "customer_profile": [
    "Industry: Tech",
    "Location: New York",
    "Size: Medium",
    "Policy Type: Health",
    "Premium Amount: $1,000",
    "Claims Filed: 1",
    "Customer Satisfaction Score: 80",
    "Time to Renewal (days): 30"
  ],
  "answers_to_questions": [
    "1. Factors impacting renewal include customer satisfaction, policy duration, and claims history.",
    "2. Claims history has a direct negative influence on renewal likelihood.",
    "3. Regional trends show higher retention in tech companies in urban areas.",
    "4. Longer policy durations typically correlate with higher renewal chances.",
    "5. Customers of medium size in the tech industry face fewer renewal challenges.",
    "6. Customer satisfaction plays a significant role in renewal likelihood.",
    "7. Upselling opportunities could be based on the premium amount and coverage options.",
    "8. Market competitiveness in New York increases retention efforts.",
    "9. To increase renewal chances, focus on improving customer satisfaction and upselling.",
    "10. Actionable strategies include offering additional coverage and increasing contact frequency."
  ],
  "recommendations": [
    "Offer a 10% discount on premium renewal.",
    "Increase customer satisfaction by improving support services.",
    "Provide additional coverage options to enhance retention."
  ]
}
```

## Explanation of Core Components
1. **Model**: The model is a machine learning model that predicts renewal probability. It is trained using features such as premium amount, customer satisfaction, policy type, etc. The model is loaded from a .pkl file.

2. **GPT Integration**: The API sends the customer's profile and a list of questions to GPT-4, which generates actionable insights for improving renewal likelihood.

3. **Flask**: Flask handles the web server and routes HTTP requests. The main route is /unified-process, where all the logic is processed and returned to the user in a structured format.

## Example Usage
To use the API, send a POST request to http://127.0.0.1:5000/unified-process with the necessary customer data. You can use tools like Postman or curl to send requests.

## Example using curl:
```bash
curl -X POST http://127.0.0.1:5000/unified-process -H "Content-Type: application/json" -d '{
  "Premium_Amount": 1000,
  "Policy_Duration_Years": 5,
  "Num_Contacts": 10,
  "Call_Duration_Minutes": 15,
  "Claims_Filed": 1,
  "Customer_Satisfaction_Score": 80,
  "Discount_Availed": 5,
  "Economic_Trend_Index": 7,
  "Market_Competitiveness": 3,
  "Time_to_Renewal": 30,
  "Industry": "Tech",
  "Location": "New York",
  "Size": "Medium",
  "Policy_Type": "Health"
}'
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.