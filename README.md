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
* **questions**: Customer tailored renewal-related questions asked to the GPT.
* **answers_to_questions**: GPT-generated answers to the renewal-related questions.
* **recommendations**: GPT-generated actionable strategies to improve the renewal likelihood.

Example response:

```json
{
    "message": [
      "This response contains details about the customer's profile, renewal probability, answers to key questions, and recommendations for improvement."
    ],
    "customer_profile": [
        "Industry: Healthcare",
        "Location: CA",
        "Size: Small",
        "Policy Type: Cyber",
        "Premium Amount: $1,200",
        "Claims Filed: 8",
        "Customer Satisfaction Score: 3.9",
        "Time to Renewal (days): 30"
    ],
    "questions": [
    "1. What factors impact renewal in Healthcare?",
    "2. How does recent claims history influence renewal likelihood?",
    "3. Are there regional trends that impact renewal in CA?",
    "4. How does policy duration affect renewal in Healthcare?",
    "5. What renewal challenges are typical for customers of size Small?",
    "6. How does customer satisfaction impact renewal decisions?",
    "7. Are there upselling opportunities based on 1200 premium?",
    "8. How does market competitiveness affect retention in CA?",
    "9. What strategies can increase renewal chances for this customer?",
    "10. What actions can improve renewal probability in the next 30 days?"
    ],
    "answers_to_questions": [
        "1. Renewal in Healthcare can be influenced by several factors: rates of premium, customer satisfaction, claims history, compliance with policy requirements, competition in the industry and premium adjustments.",
        "2. Recent claims history can certainly influence renewal likelihood. If there are a high number of claims, it could indicate a higher risk, which may lead the insurer to increase premiums leading to reduced likelihood of renewal.",
        "3. In CA, factors like stricter regulatory compliance requirements, higher competition and regional economic conditions may impact renewals.",
        "4. Policy duration can impact renewal in Healthcare. The longer a client is with a company, the higher the chances they remain loyal if they're satisfied with the service.",
        "5. Small-sized customers may struggle with higher premiums, lack of personalized service and fulfillment durations, which can pose challenges to renewal.",
        "6. High customer satisfaction often leads to higher renewal rates. Customers should feel they are receiving value for their premium and are treated respectfully and promptly attended to when they have claims or questions.",
        "7. There could be upselling opportunities even with a premium of 1200. For example, offering additional coverage options that provide value for the client, while also increasing the premium.",
        "8. Market competitiveness can greatly impact retention in CA. High competition can create pressure to decrease prices, provide better service, and offer more comprehensive coverage, impacting a client's decision to renew.",
        "9. To increase renewal chances for this customer, we could look into a personalized communication strategy, reassess the client's coverage needs, offer them a competitive renewal premium, or introduce a no-claim bonus.",
        "10. Actions to improve renewal probability in the next 30 days could include timely and effective communication regarding renewal, providing excellent customer service to deal with any concerns or dissatisfaction, or offering a small discount or additional coverage features upon renewal."
    ],
    "recommendations": [
        "- Offer competitive premiums: The premium amount can be a determining factor in renewal decisions. Evaluate the competition to ensure your prices are competitive.",
        "- Improve customer service: A satisfaction score of 3.9 indicates there might be some room for improvement. Enhance your customer services to ensure you are promptly addressing all their needs and concerns. Frequent and meaningful engagement with the client can significantly increase customer satisfaction.",
        "- Highlight the value of the coverage: Encourage renewal by demonstrating how the specific coverage benefits the customer, especially considering their size and industry. For example, show them how cyber coverage has helped similar clients mitigate cyber risk.",
        "- Address any claims issues: With the customer having filed 8 claims, they might have concerns about the claims process. Address these issues and highlight improvements that have been made to the process.",
        "- Implement a No-Claims Bonus: Implement and highlight a no-claims bonus, rewarding the client for not filing claims. This can be a great incentive for renewal.",
        "- Create a personalized experience: Given that this is a small-sized client, personalized communication and experiences can make them feel valued, thus increasing the chances of renewal.",
        "- Upselling: Present additional coverage options or packages that can provide the client with added value and protection, thereby increasing the chance of renewal.",
        "- Timely reminders: As the time to renewal is 30 days, ensure the client receives timely and adequate reminders about their policy renewal to avoid lapses caused by oversight.",
        "- Regulatory Compliance: As regulatory compliance can be quite challenging in CA, provide the client with as much assistance as possible to complete all the necessary documentation. This will make the renewal process easier and more efficient for the client."
    ],
    "renewal_probability": [
        "52.00%"
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