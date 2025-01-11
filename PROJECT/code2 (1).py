import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load("sample.pkl")



# Streamlit App
st.title("Customer Campaign Response Prediction")

st.write("This app predicts whether a customer will accept a campaign offer based on their details and purchase history.")

# Collect user input for features
st.sidebar.header("Input Features")

def user_input_features():
    feature_values = {}
    feature_columns = [
        "Income", "Recency", "MntFruits", "MntMeatProducts", "MntFishProducts", 
        "MntSweetProducts", "MntGoldProds", "NumDealsPurchases", "AcceptedCmp1", 
        "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", 
        "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth", 
        "Kidhome", "Teenhome", "Complain"
    ]

    for feature in feature_columns:
        if feature in ["Kidhome", "Teenhome", "Complain", "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5"]:
            feature_values[feature] = st.sidebar.selectbox(f"{feature}", [0, 1], index=0)
        else:
            feature_values[feature] = st.sidebar.number_input(f"{feature}", value=0.0)

    return pd.DataFrame([feature_values])

# Get user inputs
input_df = user_input_features()

# Display user inputs
st.write("### User Input Values:")
st.dataframe(input_df)

# Make predictions
if st.button("Predict"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.write("### Prediction Result:")
    result = "Accepted" if prediction[0] == 1 else "Not Accepted"
    st.write(f"Predicted Response: {result}")
    st.write(f"Probability of Acceptance: {prediction_proba[0][1]:.2f}")

# Additional Information
st.write("---")
st.write("This application uses a trained model to predict the likelihood of a customer accepting a campaign offer based on their purchase history and personal details. Please ensure the feature values are entered correctly.")