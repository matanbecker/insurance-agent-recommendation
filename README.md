# Insurance Agent Recommendation System

This project provides an AI-driven recommendation system designed to help insurance agents optimize policy renewals. The system utilizes machine learning models to predict renewal likelihood and provide personalized recommendations to agents based on customer profile data. Additionally, the system identifies similar businesses to assist in strategizing for successful renewals.

## Features

- **Renewal Prediction**: Predict the likelihood of a customer renewing their policy using a machine learning model.
- **Top Similar Businesses**: Identify the top N similar businesses to the current customer, leveraging machine learning to determine the most relevant similarities.
- **Personalized Recommendations**: Generate actionable, personalized strategies for agents to close deals with customers, based on a detailed customer profile and similar business data.
- **API Endpoint**: A RESTful API is provided for interacting with the model, receiving predictions, and getting recommendations.

## Getting Started

Follow these steps to set up and use the recommendation system locally:

### Prerequisites

- Python 3.7 or higher
- Flask for the API
- `openai` Python package for GPT-based recommendations
- `sklearn` and `pandas` for data processing and machine learning
- A valid OpenAI API key

### Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/insurance-agent-recommendation.git
cd insurance-agent-recommendation
