# Telecom-Customer-Churn-Predictor

A machine learning project focused on analyzing customer behavior and predicting churn using classification models. The project involves a complete ML pipeline—from data cleaning and exploratory analysis to model building and deployment using a Streamlit web app hosted on Hugging Face Spaces.

---

## 🎯 Objective

To build a predictive model that identifies whether a customer will **churn** (i.e., leave the telecom service) based on historical usage and demographic data.

---

## Features Used

### Categorical & Numeric Features:
- `gender`
- `SeniorCitizen`
- `Partner`
- `Dependents`
- `tenure`
- `PhoneService`
- `MultipleLines`
- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`
- `Contract`
- `PaperlessBilling`
- `PaymentMethod`
- `MonthlyCharges`
- `TotalCharges`

### 🎯 Target Variable:
- `Churn` (Yes / No)

---

## Process Followed

1. **Dataset Exploration**
2. **Initial Data Cleaning** (handling nulls, data types, etc.)
3. **Exploratory Data Analysis**
   - Univariate & Bivariate analysis
4. **Data Preprocessing**
   - Encoding categorical variables
   - Scaling numeric features
5. **Model Training**
   - Logistic Regression with Ridge Regularization
   - Random Forest Classifier
6. **Model Evaluation**
   - Confusion Matrix
   - Accuracy
   - Precision, Recall, F1-Score
7. **Deployment**
   - Final model: **Logistic Regression**
   - Built using Streamlit
   - Deployed on Hugging Face Spaces

---

## 🧰 Tech Stack

- Python
- Jupyter Notebook
- Pandas, NumPy, Seaborn, Matplotlib
- Scikit-learn
- Streamlit
- Hugging Face Spaces

---

## 🌐 Deployment

👉 **Live Demo:** Telecom Churn Predictor on Hugging Face Spaces

https://huggingface.co/spaces/rohithmasineni/Customer_Churn_Predictor
