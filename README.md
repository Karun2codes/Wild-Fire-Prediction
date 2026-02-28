# Wildfire Prediction System

An interactive machine learning web application built with Streamlit that predicts the likelihood of a wildfire based on real-time environmental parameters.



This system utilizes a Random Forest Classifier to analyze meteorological and environmental data—such as temperature, relative humidity, wind speed, and various fire weather indices—to output a risk level and a fire probability percentage.

## Features

* **Interactive Web Interface:** User-friendly input fields to test various environmental conditions.
* **Real-time Predictions:** Instantly calculates the probability of a wildfire and categorizes the risk level (Low, Moderate, High, Very High).
* **Data Visualization:** Optional toggle to view a correlation heatmap of the dataset features.
* **Model Diagnostics:** Optional toggle to view the internal model performance, including the classification report and confusion matrix.
* **Confidence Scoring:** Displays the model's confidence percentage for its prediction.

## Technology Stack

* **Frontend:** Streamlit
* **Machine Learning:** scikit-learn (RandomForestClassifier, StandardScaler)
* **Data Manipulation:** pandas, numpy
* **Data Visualization:** seaborn, matplotlib

## Installation and Setup

**1. Clone the repository**
```bash
git clone [https://github.com/yourusername/wildfire-prediction-system.git](https://github.com/yourusername/wildfire-prediction-system.git)
cd wildfire-prediction-system
