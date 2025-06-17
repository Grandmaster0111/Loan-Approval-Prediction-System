import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("loan_model.pkl")

st.title("🏦 Loan Approval Predictor")
st.write("Fill in the details below to predict if your loan will be approved.")

# User input
def user_input():
    Gender = st.selectbox("Gender", ['Male', 'Female'])
    Married = st.selectbox("Married", ['Yes', 'No'])
    Dependents = st.selectbox("Dependents", ['0', '1', '2', '3+'])
    Education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
    Self_Employed = st.selectbox("Self Employed", ['Yes', 'No'])
    ApplicantIncome = st.number_input("Applicant Income", min_value=0)
    CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
    LoanAmount = st.number_input("Loan Amount (in thousands)", min_value=0)
    Loan_Amount_Term = st.number_input("Loan Term (in days)", min_value=0)
    Credit_History = st.selectbox("Credit History", [1.0, 0.0])
    Property_Area = st.selectbox("Property Area", ['Urban', 'Rural', 'Semiurban'])

    # Convert to encoded format
    input_data = {
        'Gender': 1 if Gender == 'Male' else 0,
        'Married': 1 if Married == 'Yes' else 0,
        'Dependents': 3 if Dependents == '3+' else int(Dependents),
        'Education': 0 if Education == 'Graduate' else 1,
        'Self_Employed': 1 if Self_Employed == 'Yes' else 0,
        'ApplicantIncome': ApplicantIncome,
        'CoapplicantIncome': CoapplicantIncome,
        'LoanAmount': LoanAmount,
        'Loan_Amount_Term': Loan_Amount_Term,
        'Credit_History': Credit_History,
        'Property_Area': ['Urban', 'Rural', 'Semiurban'].index(Property_Area)
    }

    return pd.DataFrame([input_data])

input_df = user_input()

if st.button("Predict"):
    prediction = model.predict(input_df)
    result = "✅ Approved" if prediction[0] == 1 else "❌ Rejected"
    st.subheader(f"Loan Prediction Result: {result}")
