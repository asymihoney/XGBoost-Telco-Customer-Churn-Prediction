import streamlit as st
import pandas as pd
import xgboost as xgb
import json

# 1. Load Assets
@st.cache_resource
def load_assets():
    model = xgb.XGBClassifier()
    model.load_model("./models/model.json")
    with open('./models/schema.json', 'r') as f:
        columns = json.load(f)
    return model, columns

model, schema_columns = load_assets()

st.title("Telco Churn Predictor")

# 2. UI Inputs (Using the most impactful features)
col1, col2 = st.columns(2)
with col1:
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])

with col2:
    monthly = st.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0)
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    payment = st.selectbox("Payment Method", 
                          ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

if st.button("Analyze Risk"):
    # 3. Create the 30-column input (All zeros)
    input_df = pd.DataFrame(0, index=[0], columns=schema_columns)
    
    # 4. Map UI to Columns
    input_df['tenure'] = tenure
    input_df['MonthlyCharges'] = monthly
    input_df['TotalCharges'] = tenure * monthly # Logic fallback
    
    # Handle One-Hot Columns (Matches the names in your JSON)
    if f"Contract_{contract}" in input_df.columns:
        input_df[f"Contract_{contract}"] = 1
        
    if f"InternetService_{internet}" in input_df.columns:
        input_df[f"InternetService_{internet}"] = 1

    # Handle Online Security
    if f"OnlineSecurity_{online_security}" in input_df.columns:
        input_df[f"OnlineSecurity_{online_security}"] = 1
        
    # Handle Tech Support
    if f"TechSupport_{tech_support}" in input_df.columns:
        input_df[f"TechSupport_{tech_support}"] = 1

    # Logic Check: If Internet is "No", set the "No internet service" flags automatically
    if internet == "No":
        if "OnlineSecurity_No internet service" in input_df.columns:
            input_df["OnlineSecurity_No internet service"] = 1
        if "TechSupport_No internet service" in input_df.columns:
            input_df["TechSupport_No internet service"] = 1

    # Map Payment Method
    if f"PaymentMethod_{payment}" in input_df.columns:
        input_df[f"PaymentMethod_{payment}"] = 1

    # Optional: Force some 'hidden' defaults that typically drive churn
    # Since these aren't in your UI, the model assumes 0 (No)
    # Setting Paperless Billing to 'Yes' usually spikes the risk
    if "PaperlessBilling_Yes" in input_df.columns:
        input_df["PaperlessBilling_Yes"] = 1

    # 5. Predict
    prob = model.predict_proba(input_df)[0][1]
    
    # 6. Display Result
    st.divider()
    st.subheader(f"Churn Risk")

    prob_float = float(prob)

    if prob_float > 0.5:
        st.error("HIGH RISK: This customer is likely to leave.")
        st.metric(label="Churn Probability:", value=f"{prob_float:.1%}")
        st.progress(prob_float)
    elif prob_float > 0.3:
        st.warning("MEDIUM RISK: Customer showing early warning signs.")
        st.metric(label="Churn Probability:", value=f"{prob_float:.1%}")
        st.progress(prob_float)
        st.markdown("### Suggested Retention Strategy:")
        if contract == "Month-to-month":
            st.write("- **Upsell Opportunity:** Offer a 15% discount to switch to a 1-year contract.")
        if internet == "Fiber optic" and online_security == "No":
            st.write("- **Value Add:** Provide a 3-month free trial of Online Security to increase 'stickiness'.")
    else:
        st.success("LOW RISK: Customer appears stable.")
        st.metric(label="Churn Probability:", value=f"{prob_float:.1%}")
        st.progress(prob_float)

with st.expander("See Model Details"):
    st.write("""
        - **Model:** XGBoost Classifier
        - **Format:** Native JSON (Version Independent)
        - **Top Predictors:** Contract Type, Internet Service, Payment Method
        - **Baseline Churn Rate:** ~26% (A 50% probability is 2x the average risk!)
    """)