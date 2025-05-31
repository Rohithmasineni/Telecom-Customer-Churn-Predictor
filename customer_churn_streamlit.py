import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

st.title(":blue[Customer Churn Predictor]")

# Read dataset
df = pd.read_csv('Telco-Customer-Churn.csv')
df.dropna(inplace=True)

# Save unprocessed copy for UI dropdowns
df_unencoded = df.copy()

df.drop(['customerID', 'TotalCharges'], axis=1, inplace=True)

# Encode binary and categorical features
df['gender'] = df['gender'].replace({'Male': 1, 'Female': 0})

binary_cols = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].replace({'Yes': 1, 'No': 0})

# Replace "No internet/phone service" with "No"
cols_to_fix = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
               'TechSupport', 'StreamingTV', 'StreamingMovies']
for col in cols_to_fix:
    df[col] = df[col].replace({'No internet service': 'No', 'No phone service': 'No'})
    df[col] = df[col].replace({'Yes': 1, 'No': 0})

# Contract mapping
df['Contract'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})

# Separate features and target
x = df.drop('Churn', axis=1)
y = df['Churn']

# Define columns
num_cols = ['tenure', 'MonthlyCharges']
cat_cols = ['PaymentMethod', 'InternetService']

# One-Hot Encode categorical columns using sklearn
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
x_encoded = encoder.fit_transform(x[cat_cols])
x_encoded_df = pd.DataFrame(x_encoded, columns=encoder.get_feature_names_out(cat_cols), index=x.index)

# Final feature set
x_final = pd.concat([x.drop(cat_cols, axis=1), x_encoded_df], axis=1)

# Scale numeric features
scaler = StandardScaler()
x_final[num_cols] = scaler.fit_transform(x_final[num_cols])

# Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(x_final, y, test_size=0.3, random_state=42)

# Train Logistic Regression model
ridge_model = LogisticRegression(penalty='l2', class_weight='balanced', max_iter=1000, random_state=42)
ridge_model.fit(x_train, y_train)

# --- Streamlit Inputs ---
st.subheader("Enter Customer Details for Prediction")

Tenure = st.number_input("Tenure (in months)", min_value=0)
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)

# Gender input (show original values, not encoded)
selected_gender = st.selectbox("Select Gender", ['Male', 'Female'])

selected_SeniorCitizen = st.selectbox("Senior Citizen", ['Yes', 'No'])
selected_Partner = st.selectbox("Partner", ['Yes', 'No'])
selected_Dependents = st.selectbox("Dependents", ['Yes', 'No'])
selected_PhoneService = st.selectbox("Phone Service", ['Yes', 'No'])
selected_MultipleLines = st.selectbox("Multiple Lines", ['Yes', 'No'])
selected_OnlineSecurity = st.selectbox("Online Security", ['Yes', 'No'])
selected_OnlineBackup = st.selectbox("Online Backup", ['Yes', 'No'])
selected_DeviceProtection = st.selectbox("Device Protection", ['Yes', 'No'])
selected_TechSupport = st.selectbox("Tech Support", ['Yes', 'No'])
selected_StreamingTV = st.selectbox("Streaming TV", ['Yes', 'No'])
selected_StreamingMovies = st.selectbox("Streaming Movies", ['Yes', 'No'])
selected_Contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
selected_PaperlessBilling = st.selectbox("Paperless Billing", ['Yes', 'No'])
selected_PaymentMethod = st.selectbox("Payment Method", df_unencoded['PaymentMethod'].unique())
selected_InternetService = st.selectbox("Internet Service", df_unencoded['InternetService'].unique())

# Convert inputs to match encoded values
query_dict = {
    'tenure': Tenure,
    'MonthlyCharges': MonthlyCharges,
    'gender': 1 if selected_gender == 'Male' else 0,  # Gender encoding
    'SeniorCitizen': 1 if selected_SeniorCitizen == 'Yes' else 0,  # SeniorCitizen encoding
    'Partner': 1 if selected_Partner == 'Yes' else 0,  # Partner encoding
    'Dependents': 1 if selected_Dependents == 'Yes' else 0,  # Dependents encoding
    'PhoneService': 1 if selected_PhoneService == 'Yes' else 0,  # PhoneService encoding
    'MultipleLines': 1 if selected_MultipleLines == 'Yes' else 0,  # MultipleLines encoding
    'OnlineSecurity': 1 if selected_OnlineSecurity == 'Yes' else 0,  # OnlineSecurity encoding
    'OnlineBackup': 1 if selected_OnlineBackup == 'Yes' else 0,  # OnlineBackup encoding
    'DeviceProtection': 1 if selected_DeviceProtection == 'Yes' else 0,  # DeviceProtection encoding
    'TechSupport': 1 if selected_TechSupport == 'Yes' else 0,  # TechSupport encoding
    'StreamingTV': 1 if selected_StreamingTV == 'Yes' else 0,  # StreamingTV encoding
    'StreamingMovies': 1 if selected_StreamingMovies == 'Yes' else 0,  # StreamingMovies encoding
    'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2}[selected_Contract],  # Contract encoding
    'PaperlessBilling': 1 if selected_PaperlessBilling == 'Yes' else 0,  # PaperlessBilling encoding
    'PaymentMethod': selected_PaymentMethod,  # No encoding needed for PaymentMethod as it's already categorical
    'InternetService': selected_InternetService  # No encoding needed for InternetService as it's already categorical
}

# Create DataFrame from input data
query_df = pd.DataFrame([query_dict])

# Encode categorical columns using the encoder (same as training)
encoded_query = encoder.transform(query_df[cat_cols])
encoded_query_df = pd.DataFrame(encoded_query, columns=encoder.get_feature_names_out(cat_cols), index=query_df.index)

# Drop original categorical columns as they have been encoded
query_df.drop(cat_cols, axis=1, inplace=True)

# Scale numerical columns (same as training)
query_df[num_cols] = scaler.transform(query_df[num_cols])

# Ensure the final query has the same columns as the training data
query_final = pd.concat([query_df.reset_index(drop=True), encoded_query_df], axis=1)

# Make sure the columns are in the same order as during training
query_final = query_final[x_train.columns]  # Ensure column order matches training data

# --- Prediction ---
if st.button("Predict Churn"):
    result = ridge_model.predict(query_final)[0]
    st.success(f"Predicted Churn: {'Yes' if result == 1 else 'No'}")
