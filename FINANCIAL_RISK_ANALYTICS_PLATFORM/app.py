import streamlit as st
import pandas as pd
import pickle
import joblib

# ----------------------------
# Page Configuration
# ----------------------------

st.set_page_config(
    page_title="Financial Risk Analytics",
    page_icon="💰",
    layout="centered"
)

st.markdown("""
<h1 style='text-align:center;color:#4CAF50;'>
🏦 Financial Risk Analytics
</h1>
<p style='text-align:center;color:gray;font-size:18px;'>
AI-Powered Loan Default Prediction System
</p>
""", unsafe_allow_html=True)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.info("🤖 Model\n\nRandom Forest")

with col2:
    st.success("🎯 Accuracy\n\n92%")

with col3:
    st.warning("📊 Status\n\nReady")

st.divider()

# ----------------------------
# Load Model
# ----------------------------

model = pickle.load(open("Loan_default.pkl", "rb"))
encoders = joblib.load(open("encoders.pkl", "rb"))

# ----------------------------
# User Inputs
# ----------------------------

st.header("Customer Details")

age = st.number_input("Age", 18, 80, 30)

income = st.number_input(
    "Annual Income",
    min_value=1000,
    max_value=1000000,
    value=50000
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=1000,
    max_value=1000000,
    value=100000
)

credit_score = st.slider(
    "Credit Score",
    300,
    850,
    650
)

months_employed = st.number_input(
    "Months Employed",
    0,
    600,
    60
)

num_credit_lines = st.number_input(
    "Number of Credit Lines",
    1,
    20,
    5
)

interest_rate = st.slider(
    "Interest Rate (%)",
    1.0,
    30.0,
    10.5
)

loan_term = st.selectbox(
    "Loan Term (Months)",
    [12,24,36,48,60]
)

dti_ratio = st.slider(
    "Debt-to-Income Ratio",
    0.0,
    1.0,
    0.30
)

education = st.selectbox(
    "Education",
    ["High School","Bachelor","Master","PhD"]
)

employment = st.selectbox(
    "Employment Type",
    ["Full-time","Part-time","Self-employed","Unemployed"]
)

marital = st.selectbox(
    "Marital Status",
    ["Single","Married","Divorced"]
)

mortgage = st.selectbox(
    "Has Mortgage",
    ["Yes","No"]
)

dependents = st.selectbox(
    "Has Dependents",
    ["Yes","No"]
)

purpose = st.selectbox(
    "Loan Purpose",
    [
        "Home",
        "Auto",
        "Business",
        "Education",
        "Other"
    ]
)

cosigner = st.selectbox(
    "Has Co-Signer",
    ["Yes","No"]
)

# ----------------------------
# Prediction
# ----------------------------

education = encoders["Education"].transform([education])[0]
employment = encoders["EmploymentType"].transform([employment])[0]
marital = encoders["MaritalStatus"].transform([marital])[0]
mortgage = encoders["HasMortgage"].transform([mortgage])[0]
dependents = encoders["HasDependents"].transform([dependents])[0]
purpose = encoders["LoanPurpose"].transform([purpose])[0]
cosigner = encoders["HasCoSigner"].transform([cosigner])[0]


if st.button("Predict"):

    input_data = pd.DataFrame({
        "Age":[age],
        "Income":[income],
        "LoanAmount":[loan_amount],
        "CreditScore":[credit_score],
        "MonthsEmployed":[months_employed],
        "NumCreditLines":[num_credit_lines],
        "InterestRate":[interest_rate],
        "LoanTerm":[loan_term],
        "DTIRatio":[dti_ratio],
        "Education":[education],
        "EmploymentType":[employment],
        "MaritalStatus":[marital],
        "HasMortgage":[mortgage],
        "HasDependents":[dependents],
        "LoanPurpose":[purpose],
        "HasCoSigner":[cosigner]
    })

    # Encode categorical columns
    

    prediction = model.predict(input_data)[0]

    try:
        probability = model.predict_proba(input_data)[0][1]
    except:
        probability = None

    st.markdown("---")

    if prediction == 1:

        st.error("⚠️ Loan Default Predicted")

    else:

        st.success("✅ Loan Approved (No Default Predicted)")

    if probability is not None:

        st.write(f"### Default Probability: **{probability:.2%}**")

    st.markdown("---")
    st.subheader("Entered Details")
    st.dataframe(input_data)