import streamlit as st
import pandas as pd
import pickle
import joblib

# ----------------------------
# Page Configuration
# ----------------------------

st.set_page_config(
    page_title="Financial Risk Intelligence",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Custom CSS Styling
# ----------------------------

st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f172a 0%, #1a202c 100%);
        color: #e2e8f0;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a202c 0%, #0f172a 100%);
        border-right: 1px solid #2d3748;
    }
    
    [data-testid="stSidebarNav"] {
        padding: 2rem 0 0 0;
    }
    
    .main {
        padding: 2rem;
        max-width: 1400px;
        margin: 0 auto;
    }
    
    /* Header Styles */
    .dashboard-header {
        text-align: center;
        margin-bottom: 3rem;
        padding-bottom: 2rem;
        border-bottom: 2px solid #2d3748;
    }
    
    .dashboard-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #f7fafc;
        letter-spacing: -0.5px;
        margin: 0 0 0.5rem 0;
    }
    
    .dashboard-subtitle {
        font-size: 1rem;
        color: #a0aec0;
        font-weight: 400;
        margin-bottom: 1rem;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid #10b981;
        border-radius: 6px;
        color: #10b981;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    /* KPI Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1.5rem;
        margin-bottom: 3rem;
    }
    
    .kpi-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    .kpi-card:hover {
        border-color: #475569;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.1);
    }
    
    .kpi-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.75rem;
        font-weight: 600;
    }
    
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #10b981;
    }
    
    .kpi-value.neutral {
        color: #60a5fa;
    }
    
    .kpi-value.ready {
        color: #10b981;
    }
    
    /* Form Section */
    .form-section {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    .form-section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f7fafc;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #2d3748;
    }
    
    .input-group-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #cbd5e1;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 1rem;
        margin-top: 0.5rem;
    }
    
    /* Button Styles */
    .predict-button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1rem 3rem;
        font-size: 1.1rem;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
        letter-spacing: 0.5px;
    }
    
    .predict-button:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
    }
    
    /* Result Card */
    .result-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 2px solid #2d3748;
        border-radius: 12px;
        padding: 2.5rem;
        margin-top: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    .result-card.low-risk {
        border-color: #10b981;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.05) 0%, rgba(5, 150, 105, 0.02) 100%);
    }
    
    .result-card.high-risk {
        border-color: #ef4444;
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.05) 0%, rgba(220, 38, 38, 0.02) 100%);
    }
    
    .result-card.medium-risk {
        border-color: #f59e0b;
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.05) 0%, rgba(217, 119, 6, 0.02) 100%);
    }
    
    .result-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #f7fafc;
        margin-bottom: 1.5rem;
    }
    
    .risk-badge {
        display: inline-block;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
        letter-spacing: 0.5px;
    }
    
    .risk-badge.low {
        background: #10b981;
        color: white;
    }
    
    .risk-badge.high {
        background: #ef4444;
        color: white;
    }
    
    .risk-badge.medium {
        background: #f59e0b;
        color: white;
    }
    
    .probability-display {
        font-size: 2.5rem;
        font-weight: 700;
        color: #10b981;
        margin: 1rem 0;
    }
    
    .probability-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    .probability-display.high-risk {
        color: #ef4444;
    }
    
    .probability-display.medium-risk {
        color: #f59e0b;
    }
    
    /* Summary Card */
    .summary-card {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 8px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        border-left: 4px solid #10b981;
    }
    
    .summary-card.high-risk {
        background: rgba(239, 68, 68, 0.08);
        border-color: rgba(239, 68, 68, 0.3);
        border-left-color: #ef4444;
    }
    
    .summary-card.medium-risk {
        background: rgba(245, 158, 11, 0.08);
        border-color: rgba(245, 158, 11, 0.3);
        border-left-color: #f59e0b;
    }
    
    .summary-title {
        font-weight: 600;
        color: #cbd5e1;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.75rem;
    }
    
    .summary-text {
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Expandable Section */
    .expandable-section {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 1.5rem;
        margin-top: 1.5rem;
    }
    
    .expandable-title {
        font-weight: 600;
        color: #cbd5e1;
        font-size: 0.95rem;
        cursor: pointer;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding-top: 3rem;
        margin-top: 3rem;
        border-top: 1px solid #2d3748;
        color: #64748b;
        font-size: 0.85rem;
    }
    
    .footer-brand {
        font-weight: 600;
        color: #94a3b8;
        margin-bottom: 0.5rem;
    }
    
    .footer-text {
        color: #475569;
    }
    
    /* Streamlit input overrides */
    .stNumberInput, .stSlider, .stSelectbox {
        margin-bottom: 1rem;
    }
    
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Sidebar Navigation
# ----------------------------

with st.sidebar:
    st.markdown("""
    <div style='padding: 1.5rem 0; margin-bottom: 2rem;'>
        <h2 style='color: #10b981; font-size: 1.5rem; margin: 0; font-weight: 700;'>FRI</h2>
        <p style='color: #94a3b8; font-size: 0.85rem; margin: 0.25rem 0 0 0;'>Financial Risk Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style='padding: 1rem 0;'>
        <p style='color: #cbd5e1; font-weight: 600; margin: 0 0 1rem 0;'>NAVIGATION</p>
        <p style='color: #94a3b8; font-size: 0.9rem; margin: 0.5rem 0;'>📊 Dashboard</p>
        <p style='color: #94a3b8; font-size: 0.9rem; margin: 0.5rem 0;'>⚙️ Risk Assessment</p>
        <p style='color: #94a3b8; font-size: 0.9rem; margin: 0.5rem 0;'>ℹ️ Model Information</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style='padding: 1rem 0; color: #64748b; font-size: 0.85rem;'>
        <p style='margin: 0.5rem 0;'><strong>Model:</strong> Random Forest</p>
        <p style='margin: 0.5rem 0;'><strong>Accuracy:</strong> 92%</p>
        <p style='margin: 0.5rem 0;'><strong>Version:</strong> 1.0</p>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# Main Content Container
# ----------------------------

st.markdown("""
<div class='dashboard-header'>
    <h1 class='dashboard-title'>Financial Risk Intelligence</h1>
    <p class='dashboard-subtitle'>AI-powered loan default risk assessment</p>
    <div style='margin-top: 1rem;'>
        <div class='status-badge'>● MODEL ONLINE</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# KPI Cards
# ----------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class='kpi-card'>
        <div class='kpi-label'>Model</div>
        <div class='kpi-value neutral'>RF</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='kpi-card'>
        <div class='kpi-label'>Accuracy</div>
        <div class='kpi-value'>92%</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='kpi-card'>
        <div class='kpi-label'>Status</div>
        <div class='kpi-value ready'>Ready</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class='kpi-card'>
        <div class='kpi-label'>Assessment</div>
        <div class='kpi-value ready'>Active</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------
# Load Model
# ----------------------------

model = pickle.load(open("Loan_default.pkl", "rb"))
encoders = joblib.load(open("encoders.pkl", "rb"))

# Load categorical classes from encoders
education_classes = list(encoders["Education"].classes_)
employment_classes = list(encoders["EmploymentType"].classes_)
marital_classes = list(encoders["MaritalStatus"].classes_)
mortgage_classes = list(encoders["HasMortgage"].classes_)
dependents_classes = list(encoders["HasDependents"].classes_)
purpose_classes = list(encoders["LoanPurpose"].classes_)
cosigner_classes = list(encoders["HasCoSigner"].classes_)

# ----------------------------
# Form Section
# ----------------------------

st.markdown("""
<div class='form-section'>
    <h3 class='form-section-title'>Customer Risk Assessment</h3>
""", unsafe_allow_html=True)

# Financial Profile
st.markdown("""
<p class='input-group-title'>Financial Profile</p>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    income = st.number_input(
        "Annual Income ($)",
        min_value=1000,
        max_value=1000000,
        value=50000,
        step=5000
    )

with col2:
    loan_amount = st.number_input(
        "Loan Amount ($)",
        min_value=1000,
        max_value=1000000,
        value=100000,
        step=5000
    )

with col3:
    interest_rate = st.slider(
        "Interest Rate (%)",
        1.0,
        30.0,
        10.5,
        0.1
    )

col1, col2, col3 = st.columns(3)

with col1:
    loan_term = st.selectbox(
        "Loan Term (Months)",
        [12, 24, 36, 48, 60],
        index=2
    )

with col2:
    dti_ratio = st.slider(
        "Debt-to-Income Ratio",
        0.0,
        1.0,
        0.30,
        0.05
    )

with col3:
    st.empty()

# Credit Profile
st.markdown("""
<p class='input-group-title'>Credit Profile</p>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    credit_score = st.slider(
        "Credit Score",
        300,
        850,
        650,
        10
    )

with col2:
    num_credit_lines = st.number_input(
        "Number of Credit Lines",
        1,
        20,
        5
    )

with col3:
    months_employed = st.number_input(
        "Months Employed",
        0,
        600,
        60,
        step=6
    )

# Personal Profile
st.markdown("""
<p class='input-group-title'>Personal Profile</p>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        18,
        80,
        30
    )

with col2:
    education = st.selectbox(
        "Education",
        education_classes
    )

with col3:
    employment = st.selectbox(
        "Employment Type",
        employment_classes
    )

col1, col2, col3 = st.columns(3)

with col1:
    marital = st.selectbox(
        "Marital Status",
        marital_classes
    )

with col2:
    mortgage = st.selectbox(
        "Has Mortgage",
        mortgage_classes
    )

with col3:
    dependents = st.selectbox(
        "Has Dependents",
        dependents_classes
    )

# Loan Profile
st.markdown("""
<p class='input-group-title'>Loan Profile</p>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    purpose = st.selectbox(
        "Loan Purpose",
        purpose_classes
    )

with col2:
    cosigner = st.selectbox(
        "Has Co-Signer",
        cosigner_classes
    )

st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Encode Categorical Variables
# ----------------------------

education_encoded = encoders["Education"].transform([education])[0]
employment_encoded = encoders["EmploymentType"].transform([employment])[0]
marital_encoded = encoders["MaritalStatus"].transform([marital])[0]
mortgage_encoded = encoders["HasMortgage"].transform([mortgage])[0]
dependents_encoded = encoders["HasDependents"].transform([dependents])[0]
purpose_encoded = encoders["LoanPurpose"].transform([purpose])[0]
cosigner_encoded = encoders["HasCoSigner"].transform([cosigner])[0]

# ----------------------------
# Predict Button
# ----------------------------

col1, col2, col3 = st.columns([1, 1.5, 1])

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button(
        "Analyze Credit Risk",
        use_container_width=True,
        key="predict_btn"
    )

# ----------------------------
# Prediction Logic
# ----------------------------

if predict_button:
    
    input_data = pd.DataFrame({
        "Age": [age],
        "Income": [income],
        "LoanAmount": [loan_amount],
        "CreditScore": [credit_score],
        "MonthsEmployed": [months_employed],
        "NumCreditLines": [num_credit_lines],
        "InterestRate": [interest_rate],
        "LoanTerm": [loan_term],
        "DTIRatio": [dti_ratio],
        "Education": [education_encoded],
        "EmploymentType": [employment_encoded],
        "MaritalStatus": [marital_encoded],
        "HasMortgage": [mortgage_encoded],
        "HasDependents": [dependents_encoded],
        "LoanPurpose": [purpose_encoded],
        "HasCoSigner": [cosigner_encoded]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Determine risk level
    if probability < 0.35:
        risk_level = "LOW RISK"
        risk_class = "low"
        result_class = "low-risk"
    elif probability < 0.65:
        risk_level = "MEDIUM RISK"
        risk_class = "medium"
        result_class = "medium-risk"
    else:
        risk_level = "HIGH RISK"
        risk_class = "high"
        result_class = "high-risk"

    # Result section
    if prediction == 1:
        result_title = "⚠️ Loan Default Risk Detected"
        summary_text = f"Based on the financial profile provided, there is a {probability:.1%} probability of loan default. Consider requesting additional collateral or co-signer verification."
    else:
        result_title = "✅ No Default Risk Detected"
        summary_text = f"The applicant shows strong creditworthiness. The estimated default probability is {probability:.1%}, which falls within acceptable risk parameters."

    st.markdown(f"""
    <div class='result-card {result_class}'>
        <h2 class='result-title'>{result_title}</h2>
        
        <div class='risk-badge {risk_class}'>{risk_level}</div>
        
        <div class='probability-label'>Default Probability</div>
        <div class='probability-display {result_class if prediction == 1 or probability >= 0.65 else ''} {risk_class}'>
            {probability:.2%}
        </div>
        
        <div class='summary-card {result_class}'>
            <div class='summary-title'>Risk Assessment Summary</div>
            <div class='summary-text'>{summary_text}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Entered Details
    with st.expander("📋 View Entered Customer Details", expanded=False):
        st.markdown("""
        <div style='background: rgba(16, 185, 129, 0.05); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 8px; padding: 1rem;'>
        """, unsafe_allow_html=True)
        
        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Footer
# ----------------------------

st.markdown("""
<div class='footer'>
    <div class='footer-brand'>Financial Risk Intelligence</div>
    <div class='footer-text'>Powered by Machine Learning • Random Forest Classifier</div>
</div>
""", unsafe_allow_html=True)
