import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Wildfire Prediction System",
    page_icon="",
    layout="centered"
)

# Title
st.title("")
st.markdown("Enter environmental parameters to predict wildfire risk")

# Load and prepare data
wd = pd.read_csv(r'C:\Users\K\Desktop\wild fire\Fire_dataset_cleaned.csv')
lb = LabelEncoder()
wd['Classes']=lb.fit_transform(wd['Classes'])

# Show correlation heatmap (optional)
if st.checkbox("Show Correlation Heatmap"):
    plt.figure(figsize=(10,10))
    sns.heatmap(wd.corr(numeric_only=True),annot=True,cmap='coolwarm')
    st.pyplot(plt)
    plt.clf()

wd.drop(['Unnamed: 0','year'],axis=1,inplace= True)
x = wd.drop(['Classes'],axis = 1).values
y = wd['Classes'].values
feature_names = wd.drop(['Classes'], axis=1).columns.tolist()

# Split and scale data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)

# Train model
md = RandomForestClassifier(class_weight='balanced',max_depth=None,n_estimators= 200,min_samples_leaf=10)
md.fit(x_train_scaled,y_train)
pred = md.predict(x_test_scaled)

# Show model performance (optional)
if st.checkbox("Show Model Performance"):
    st.subheader("Model Performance")
    st.text("Classification Report:")
    st.text(classification_report(y_test,pred))
    st.text("Confusion Matrix:")
    st.text(confusion_matrix(y_test,pred))

st.divider()

# Input features section
st.subheader("Input Environmental Parameters")

# Create input fields for each feature
user_inputs = {}

# Get reasonable default values
default_values = {
    'day': 15,
    'month': 6,
    'Temperature': 25.0,
    'RH': 60.0,
    'Ws': 15.0,
    'Rain': 0.0,
    'FFMC': 65.0,
    'DMC': 10.0,
    'DC': 50.0,
    'ISI': 5.0,
    'BUI': 20.0,
    'FWI': 10.0,
    'Region': 1
}

# Create input fields in two columns
col1, col2 = st.columns(2)

for i, feature in enumerate(feature_names):
    default_val = default_values.get(feature, 0.0)
    if i % 2 == 0:
        with col1:
            if feature == 'Region':
                user_inputs[feature] = st.selectbox(f"{feature}", [1, 2], index=0)
            else:
                user_inputs[feature] = st.number_input(f"{feature}", value=float(default_val), step=0.1)
    else:
        with col2:
            if feature == 'Region':
                user_inputs[feature] = st.selectbox(f"{feature}", [1, 2], index=0)
            else:
                user_inputs[feature] = st.number_input(f"{feature}", value=float(default_val), step=0.1)

# Prediction button
if st.button("Predict Wildfire Risk", type="primary"):
    # Convert input to DataFrame
    input_df = pd.DataFrame([user_inputs])
    
    # Ensure correct column order
    input_df = input_df[feature_names]
    
    # Scale the input
    input_scaled = sc.transform(input_df.values)
    
    # Make prediction
    prediction = md.predict(input_scaled)
    prediction_proba = md.predict_proba(input_scaled)
    
    # Convert prediction back to original label
    prediction_label = lb.inverse_transform(prediction)[0]
    fire_prob = prediction_proba[0][1] * 100
    
    # Display results
    st.divider()
    st.subheader("Prediction Results")
    
    if prediction_label == 'fire':
        st.error("")
        st.metric("Fire Probability", f"{fire_prob:.1f}%")
    else:
        st.success("")
        st.metric("Fire Probability", f"{fire_prob:.1f}%")
    
    # Show additional details
    st.subheader("Additional Information")
    col1, col2 = st.columns(2)
    
    with col1:
        risk_level = "Very High" if fire_prob > 70 else "High" if fire_prob > 50 else "Moderate" if fire_prob > 30 else "Low"
        st.metric("Risk Level", risk_level)
    
    with col2:
        confidence = max(prediction_proba[0]) * 100
        st.metric("Model Confidence", f"{confidence:.1f}%")