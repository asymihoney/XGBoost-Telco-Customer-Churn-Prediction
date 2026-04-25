import streamlit as st
import pandas as pd
import joblib

# 1. Load the saved Pipeline
model = joblib.load('models/churn_model.pkl')

st.set_page_config(page_title="Churn Sentinel", layout="centered")

st.title("Churn Sentinel: Customer Retention Tool")
st.write("Enter customer details to calculate the risk of cancellation.")

# 2. Create Input Fields (Match your column names exactly!)
col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
    contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
    monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 70)

with col2:
    internet = st.selectbox("Internet Service", ['Fiber optic', 'DSL', 'No'])
    security = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
    tech_support = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])

# 3. Prediction Logic
if st.button("Analyze Risk"):
    # Create a dataframe for the model
    input_df = pd.DataFrame([{
        'tenure': tenure,
        'Contract': contract,
        'MonthlyCharges': monthly_charges,
        'InternetService': internet,
        'OnlineSecurity': security,
        'TechSupport': tech_support,
        # ... Add other columns with default values if they aren't in the UI
    }])
    
    # Get Probability
    prob = model.predict_proba(input_df)[0][1]
    
    st.divider()
    if prob > 0.7:
        st.error(f"High Risk! Probability: {prob:.2%}")
        st.write("Suggestion: Offer a 1-year contract discount.")
    elif prob > 0.4:
        st.warning(f"Medium Risk. Probability: {prob:.2%}")
    else:
        st.success(f"Low Risk. Probability: {prob:.2%}")