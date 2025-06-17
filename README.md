# 🏦 Loan Approval Prediction System

A machine learning-based system to predict whether a loan should be approved or not based on applicant details. This project showcases the end-to-end data science workflow — from data preprocessing and model training to deployment using a **Streamlit** web app.


---

## 📌 Problem Statement

Financial institutions face significant risks when approving loan applications. This project helps predict the likelihood of loan approval using historical loan data and machine learning techniques.

---

## 🧠 Technologies Used

- **Python**
- **Pandas**, **NumPy**
- **scikit-learn**
- **RandomForestClassifier**
- **GridSearchCV**, **StratifiedKFold**
- **Streamlit**
- **Joblib**

---

## 📂 Dataset

Dataset Source: [Kaggle - Loan Prediction Problem Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)

- `train.csv`: Used for training and model evaluation
- `test.csv`: (Optional) For custom evaluation

---

## 🔍 Features

The model uses the following input features:

- Gender
- Marital Status
- Dependents
- Education
- Self Employed
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Term
- Credit History
- Property Area

Additional engineered features:
- Total Income (Applicant + Coapplicant)
- Log-transformed Loan Amount and Total Income for skewness correction

---

## ⚙️ Model Training

- Used `RandomForestClassifier` with hyperparameter tuning via `GridSearchCV`
- Achieved **85% accuracy**
- Model trained using a **stratified 80/20 train-test split**
- Final model saved with `joblib`

```bash
python train_model.py
