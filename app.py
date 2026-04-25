import streamlit as st
import pandas as pd
import joblib

# Load model - ensure the .pkl file is in your GitHub root!
@st.cache_resource # This speeds up your app by loading the model only once
def load_model():
    return joblib.load('churn_model.pkl')

model = load_model()

st.title("📡 Customer Churn Predictor")

# Input features (Match your training features exactly)
tenure = st.slider("Tenure (Months)", 0, 72, 12)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
monthly_charges = st.number_input("Monthly Charges", 18, 120, 50)
internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])

if st.button("Predict Churn"):
    # Create input dataframe
    data = pd.DataFrame([[tenure, contract, monthly_charges, internet]], 
                        columns=['tenure', 'Contract', 'MonthlyCharges', 'InternetService'])
    
    # Preprocessing is handled by the pipeline in our .pkl!
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    
    if prediction == 1:
        st.error(f"High Risk of Churn: {prob:.2%}")
    else:
        st.success(f"Low Risk of Churn: {prob:.2%}")