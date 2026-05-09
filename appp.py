import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("Churn_model.pkl")

# Load columns
model_columns = joblib.load("model_columns.pkl")

# Page settings
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊"
)

# Title
st.title("📊 AI Customer Churn Predictor")

st.write(
    "Predict whether a telecom customer is likely to churn."
)

# User inputs
tenure = st.slider(
    "Tenure (Months)",
    0,
    72,
    12
)

monthly_charges = st.number_input(
    "Monthly Charges",
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    value=1000.0
)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["Yes", "No"]
)

online_security = st.selectbox(
    "Online Security",
    ["Yes", "No"]
)

paperless = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

# Create input data
input_data = {
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'Contract_One year': 1 if contract == "One year" else 0,
    'Contract_Two year': 1 if contract == "Two year" else 0,
    'InternetService_Fiber optic': 1 if internet == "Fiber optic" else 0,
    'OnlineSecurity_Yes': 1 if online_security == "Yes" else 0,
    'TechSupport_Yes': 1 if tech_support == "Yes" else 0,
    'PaperlessBilling_Yes': 1 if paperless == "Yes" else 0
}

# Convert to dataframe
input_df = pd.DataFrame([input_data])

# Add missing columns
for col in model_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Correct column order
input_df = input_df[model_columns]

# Prediction
if st.button("🔍 Predict Churn"):

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer likely to churn")
    else:
        st.success("✅ Customer likely to stay")

    st.write(
        f"Churn Probability: {probability:.2f}"
    )