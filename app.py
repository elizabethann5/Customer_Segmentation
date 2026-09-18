%%writefile app.py

import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
data = joblib.load("customer_segmentation_model.pkl")

model = data["model"]
scaler = data["scaler"]

st.title("Customer Segmentation")

st.write("Enter customer details")

income = st.number_input(
    "Annual Income (INR)",
    min_value=0.0,
    value=600000.0
)

spending = st.number_input(
    "Monthly Spending (INR)",
    min_value=0.0,
    value=12000.0
)

visits = st.number_input(
    "Visits Per Month",
    min_value=0,
    value=5
)

if st.button("Predict Customer Segment"):

    new_customer = pd.DataFrame({
        "Annual_Income_INR": [income],
        "Monthly_Spending_INR": [spending],
        "Visits_Per_Month": [visits]
    })

    scaled = scaler.transform(new_customer)

    cluster = model.predict(scaled)[0]

    st.success(f"Customer belongs to Cluster {cluster}")
