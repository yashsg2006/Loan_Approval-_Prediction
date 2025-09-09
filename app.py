import streamlit as st
import pandas as pd
import joblib, json
from pathlib import Path

# Field name mapping 
FIELD_DISPLAY_NAMES = {
    "cb_person_cred_hist_length": "Credit History Length (years)",
    "credit_score": "Credit Score", 
    "person_gender": "Gender",
    "person_education": "Education Level",
    "person_home_ownership": "Home Ownership Status",
    "loan_intent": "Loan Purpose",
    "person_age": "Age (years)",
    "person_income": "Annual Income",
    "person_emp_exp": "Employment Experience (years)", 
    "loan_amnt": "Loan Amount",
    "loan_int_rate": "Interest Rate (%)",
    "loan_percent_income": "Loan as % of Income",
    "previous_loan_defaults_on_file": "Previous Loan Defaults"
}

# ------------------------------
# Loading model, preprocessor, schema
# ------------------------------
MODEL_DIR = Path("model_artifacts")
clf = joblib.load(MODEL_DIR / "xgb_final_model.joblib")
preproc = joblib.load(MODEL_DIR / "preprocessor.joblib")
schema = json.load(open(MODEL_DIR / "input_schema.json"))

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
st.title("🏦 Loan Approval Predictor")

st.sidebar.header("Enter Applicant Details")

# Collecting user input based on schema
inputs = {}
for feat in schema["feature_order"]:
    if feat in schema["numeric"]:
        default = float(schema["defaults"].get(feat, 0.0))
        val = st.sidebar.number_input(feat, value=default)
        inputs[feat] = val
    else:
        choices = schema["categorical"][feat]["choices"]
        default = schema["categorical"][feat]["default"]
        idx = choices.index(default) if default in choices else 0
        sel = st.sidebar.selectbox(feat, options=choices, index=idx)
        inputs[feat] = sel

# Prediction button
if st.sidebar.button("Predict"):
    df = pd.DataFrame([inputs], columns=schema["feature_order"])
    Xtr = preproc.transform(df)
    pred = clf.predict(Xtr)[0]
    proba = clf.predict_proba(Xtr)[0][1]

    if pred == 1:
        st.success(f"✅ Loan Approved (probability = {proba:.2%})")
    else:
        st.error(f"❌ Loan Rejected (probability = {proba:.2%})")

    st.write("### Input Summary")
    st.write(df)

