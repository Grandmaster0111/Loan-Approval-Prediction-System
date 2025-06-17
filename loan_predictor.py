import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/Grandmaster0111/ML-Datasets/main/Loan-Approval/loan.csv")
    return df

df = load_data()

# Preprocessing
df.dropna(inplace=True)
le = LabelEncoder()
for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status', 'Dependents']:
    df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop(columns=['Loan_ID', 'Loan_Status'])
y = df['Loan_Status']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)

# Streamlit UI
st.title("🏦 Loan Approval Prediction System")
st.markdown("Predict loan approval using ML with accuracy: **{:.2f}%**".format(acc * 100))

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

    user_data = {
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

    return pd.DataFrame([user_data])

input_df = user_input()

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_df)
    result = 'Approved' if prediction[0] == 1 else 'Rejected'
    st.success(f"🏁 Loan Status: **{result}**")

