
import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
data = joblib.load("customer_segmentation_model.pkl")

model = data["model"]
scaler = data["scaler"]

# App title
st.title("Customer Segmentation")

st.write("Enter customer details to identify the customer segment.")

# User inputs
income = st.number_input(
    "Annual Income (INR)",
    min_value=0.0,
    value=600000.0,
    step=10000.0
)

spending = st.number_input(
    "Monthly Spending (INR)",
    min_value=0.0,
    value=12000.0,
    step=500.0
)

visits = st.number_input(
    "Visits Per Month",
    min_value=0,
    value=5,
    step=1
)

# Prediction button
if st.button("Predict Customer Segment"):

    # Validate income
    if income <= 300000 or income >= 2500000:
        st.warning(
            "Annual Income must be greater than ₹3,00,000 "
            "and less than ₹25,00,000."
        )

    # Validate spending
    elif spending <= 2000 or spending >= 75000:
        st.warning(
            "Monthly Spending must be greater than ₹2,000 "
            "and less than ₹75,000."
        )

    else:
        new_customer = pd.DataFrame({
            "Annual_Income_INR": [income],
            "Monthly_Spending_INR": [spending],
            "Visits_Per_Month": [visits]
        })

        # Scale input
        scaled = scaler.transform(new_customer)

        # Predict cluster
        cluster = model.predict(scaled)[0]

        st.success(
            f"Customer belongs to Cluster {cluster}"
        )
