import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------
# Custom CSS for Premium Aviation Dashboard
# ----------------------------

custom_css = """
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    /* Hide default Streamlit elements */
    #MainMenu {
        visibility: hidden;
    }
    footer {
        visibility: hidden;
    }
    header {
        visibility: hidden;
    }

    /* Main background and text */
    body, .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
        color: #e8eaed;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Remove default padding */
    .main .block-container {
        padding-top: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1200px;
    }

    /* Hero section */
    .hero-section {
        text-align: center;
        margin-bottom: 2rem;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #1a1f3a 0%, #252e4a 100%);
        border-radius: 12px;
        border: 1px solid rgba(100, 200, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #64c8ff 0%, #7dd3ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #a0a9be;
        margin-bottom: 1rem;
    }

    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        background: rgba(100, 200, 255, 0.15);
        border: 1px solid rgba(100, 200, 255, 0.4);
        border-radius: 20px;
        font-size: 0.9rem;
        color: #64c8ff;
        margin-bottom: 1rem;
    }

    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #00ff00;
        border-radius: 50%;
        margin-right: 0.5rem;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* Card styling */
    .input-card {
        background: linear-gradient(135deg, #141824 0%, #1a1f3a 100%);
        border: 1px solid rgba(100, 200, 255, 0.15);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }

    .input-card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #64c8ff;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.95rem;
    }

    /* Form elements */
    .stSelectbox, .stNumberInput, .stSlider {
        margin-bottom: 1rem;
    }

    .stSelectbox > div > div > select,
    .stNumberInput > div > input,
    .stSlider > div > input {
        background: #0f1419 !important;
        color: #e8eaed !important;
        border: 1px solid rgba(100, 200, 255, 0.2) !important;
        border-radius: 6px !important;
    }

    .stSelectbox > div > div > select:hover,
    .stNumberInput > div > input:hover,
    .stSlider > div > input:hover {
        border-color: rgba(100, 200, 255, 0.5) !important;
    }

    .stSelectbox > div > div > select:focus,
    .stNumberInput > div > input:focus,
    .stSlider > div > input:focus {
        border-color: #64c8ff !important;
        box-shadow: 0 0 0 2px rgba(100, 200, 255, 0.1) !important;
    }

    /* Labels */
    label {
        color: #b0b9ca !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }

    /* Prediction button */
    .stFormSubmitButton > button {
        width: 100% !important;
        padding: 1rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        background: linear-gradient(90deg, #00aeff 0%, #0099ff 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        cursor: pointer;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 174, 255, 0.3) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stFormSubmitButton > button:hover {
        background: linear-gradient(90deg, #00c4ff 0%, #00b3ff 100%) !important;
        box-shadow: 0 6px 25px rgba(0, 174, 255, 0.5) !important;
        transform: translateY(-2px);
    }

    /* Result section */
    .result-card {
        background: linear-gradient(135deg, #141824 0%, #1a1f3a 100%);
        border: 2px solid rgba(100, 200, 255, 0.25);
        border-radius: 10px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .result-card.on-time {
        border-color: rgba(76, 175, 80, 0.4);
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.05) 0%, rgba(76, 175, 80, 0.02) 100%);
    }

    .result-card.delayed {
        border-color: rgba(255, 152, 0, 0.4);
        background: linear-gradient(135deg, rgba(255, 152, 0, 0.05) 0%, rgba(255, 152, 0, 0.02) 100%);
    }

    .prediction-status {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    .prediction-status.on-time {
        color: #4caf50;
    }

    .prediction-status.delayed {
        color: #ff9800;
    }

    .probability-section {
        margin-top: 1.5rem;
    }

    .prob-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
        color: #b0b9ca;
    }

    .prob-bar {
        height: 10px;
        background: #0f1419;
        border-radius: 5px;
        overflow: hidden;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(100, 200, 255, 0.1);
    }

    .prob-fill {
        height: 100%;
        border-radius: 5px;
        transition: width 0.5s ease;
    }

    .prob-fill.on-time {
        background: linear-gradient(90deg, #4caf50 0%, #66bb6a 100%);
    }

    .prob-fill.delayed {
        background: linear-gradient(90deg, #ff9800 0%, #ffb74d 100%);
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #1a1f3a 0%, #252e4a 100%) !important;
        border: 1px solid rgba(100, 200, 255, 0.15) !important;
        color: #64c8ff !important;
    }

    .streamlit-expanderHeader:hover {
        border-color: rgba(100, 200, 255, 0.4) !important;
    }

    /* Footer */
    .footer-section {
        margin-top: 3rem;
        padding: 2rem;
        background: linear-gradient(135deg, #0a0e27 0%, #141824 100%);
        border: 1px solid rgba(100, 200, 255, 0.1);
        border-radius: 10px;
        text-align: center;
        color: #6b7280;
        font-size: 0.85rem;
    }

    .footer-text {
        margin: 0.5rem 0;
    }

    .divider {
        margin: 2rem 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(100, 200, 255, 0.2), transparent);
    }

    /* Responsive */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2rem;
        }

        .hero-subtitle {
            font-size: 0.95rem;
        }

        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .input-card {
            padding: 1rem;
        }
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# ----------------------------
# Model & Scaler Paths
# ----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "flight_delay_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scalar.pkl")


# ----------------------------
# Feature Order
# ----------------------------

FEATURE_ORDER = [
    "Cancelled",
    "Diverted",
    "DepTime",
    "AirTime",
    "CRSElapsedTime",
    "Operated_or_Branded_Code_Share_Partners",
    "OriginState",
    "OriginStateFips",
    "OriginWac",
    "DestState",
    "DestStateFips",
    "DestWac",
    "DistanceGroup",
    "DivAirportLandings",
]


# ----------------------------
# Load Model & Scaler
# ----------------------------

@st.cache_resource
def load_model_and_scaler():

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


model, scaler = load_model_and_scaler()


# ----------------------------
# Hero Section
# ----------------------------

st.markdown("""
<div class="hero-section">
    <div class="status-badge">
        <span class="status-dot"></span>
        ML MODEL ONLINE
    </div>
    <div class="hero-title">✈️ Flight Delay Predictor</div>
    <div class="hero-subtitle">
        Powered by Random Forest Machine Learning Model
    </div>
    <div class="hero-subtitle" style="font-size: 0.95rem; color: #7b8496;">
        Predicts arrival delays (15+ minutes) based on flight characteristics, routing, and operational parameters
    </div>
</div>
""", unsafe_allow_html=True)


# ----------------------------
# Flight Input Form
# ----------------------------

with st.form("flight_form"):

    # Flight Status Section
    st.markdown('<div class="input-card"><div class="input-card-title">✓ Flight Status</div>', unsafe_allow_html=True)
    
    col_status1, col_status2 = st.columns(2)
    
    with col_status1:
        cancelled = st.selectbox(
            "Cancelled?",
            ["No", "Yes"],
            key="cancelled"
        )
    
    with col_status2:
        diverted = st.selectbox(
            "Diverted?",
            ["No", "Yes"],
            key="diverted"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Flight Timing Section
    st.markdown('<div class="input-card"><div class="input-card-title">🕐 Flight Timing</div>', unsafe_allow_html=True)
    
    col_time1, col_time2 = st.columns(2)
    
    with col_time1:
        dep_time = st.number_input(
            "Departure time (HHMM format, e.g., 1430)",
            min_value=0,
            max_value=2359,
            value=1430,
            key="dep_time"
        )
    
    with col_time2:
        air_time = st.number_input(
            "Air time (minutes)",
            min_value=0,
            value=150,
            key="air_time"
        )
    
    crs_elapsed = st.number_input(
        "Scheduled elapsed time (minutes)",
        min_value=0,
        value=180,
        key="crs_elapsed"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Origin Section
    st.markdown('<div class="input-card"><div class="input-card-title">📍 Origin Airport</div>', unsafe_allow_html=True)
    
    col_orig1, col_orig2, col_orig3 = st.columns(3)
    
    with col_orig1:
        origin_state = st.number_input(
            "Origin state (encoded)",
            min_value=0,
            value=4,
            key="origin_state"
        )
    
    with col_orig2:
        origin_fips = st.number_input(
            "Origin state FIPS code",
            min_value=0,
            value=8,
            key="origin_fips"
        )
    
    with col_orig3:
        origin_wac = st.number_input(
            "Origin World Area Code",
            min_value=0,
            value=82,
            key="origin_wac"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Destination Section
    st.markdown('<div class="input-card"><div class="input-card-title">📍 Destination Airport</div>', unsafe_allow_html=True)
    
    col_dest1, col_dest2, col_dest3 = st.columns(3)
    
    with col_dest1:
        dest_state = st.number_input(
            "Destination state (encoded)",
            min_value=0,
            value=6,
            key="dest_state"
        )
    
    with col_dest2:
        dest_fips = st.number_input(
            "Destination state FIPS code",
            min_value=0,
            value=6,
            key="dest_fips"
        )
    
    with col_dest3:
        dest_wac = st.number_input(
            "Destination World Area Code",
            min_value=0,
            value=91,
            key="dest_wac"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Flight & Route Information Section
    st.markdown('<div class="input-card"><div class="input-card-title">✈ Flight & Route Information</div>', unsafe_allow_html=True)
    
    col_route1, col_route2, col_route3 = st.columns(3)
    
    with col_route1:
        carrier_code = st.number_input(
            "Carrier code (encoded)",
            min_value=0,
            value=0,
            key="carrier_code",
            help="LabelEncoder value from training data"
        )
    
    with col_route2:
        distance_group = st.slider(
            "Distance group",
            min_value=1,
            max_value=11,
            value=5,
            key="distance_group",
            help="1 = <250mi | 11 = 2500mi+"
        )
    
    with col_route3:
        div_landings = st.number_input(
            "Diverted airport landings",
            min_value=0,
            value=0,
            key="div_landings"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Submit Button
    st.markdown('<div style="height: 1.5rem;"></div>', unsafe_allow_html=True)
    submitted = st.form_submit_button(
        "🔍 PREDICT DELAY STATUS",
        use_container_width=True
    )


# ----------------------------
# Prediction Logic (UNCHANGED)
# ----------------------------

if submitted:

    row = {
        "Cancelled": 1 if cancelled == "Yes" else 0,
        "Diverted": 1 if diverted == "Yes" else 0,
        "DepTime": dep_time,
        "AirTime": air_time,
        "CRSElapsedTime": crs_elapsed,
        "Operated_or_Branded_Code_Share_Partners": carrier_code,
        "OriginState": origin_state,
        "OriginStateFips": origin_fips,
        "OriginWac": origin_wac,
        "DestState": dest_state,
        "DestStateFips": dest_fips,
        "DestWac": dest_wac,
        "DistanceGroup": distance_group,
        "DivAirportLandings": div_landings,
    }

    # Create DataFrame in exact training order
    X = pd.DataFrame([row], columns=FEATURE_ORDER)

    # Apply the SAME scaler used during training
    X_scaled = scaler.transform(X)

    # Prediction
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]

    # ----------------------------
    # Result Section
    # ----------------------------

    if pred == 1:
        result_type = "delayed"
        result_class = "delayed"
        status_icon = "⏰"
        status_text = "DELAY LIKELY"
        status_color = "delayed"
        primary_prob = proba[1]
        secondary_prob = proba[0]
    else:
        result_type = "on-time"
        result_class = "on-time"
        status_icon = "✅"
        status_text = "ON TIME"
        status_color = "on-time"
        primary_prob = proba[0]
        secondary_prob = proba[1]

    st.markdown(f"""
    <div class="result-card {result_class}">
        <div class="prediction-status {status_color}">
            {status_icon} {status_text}
        </div>
        <div style="font-size: 1.1rem; color: #b0b9ca; margin-bottom: 1.5rem;">
            Model Confidence: <strong style="color: #64c8ff;">{primary_prob * 100:.1f}%</strong>
        </div>
        
        <div class="probability-section">
            <div class="prob-label">
                <span>On-Time Probability</span>
                <span style="color: #4caf50;">{proba[0] * 100:.1f}%</span>
            </div>
            <div class="prob-bar">
                <div class="prob-fill on-time" style="width: {proba[0] * 100}%;"></div>
            </div>
            
            <div class="prob-label">
                <span>Delay Probability (15+ min)</span>
                <span style="color: #ff9800;">{proba[1] * 100:.1f}%</span>
            </div>
            <div class="prob-bar">
                <div class="prob-fill delayed" style="width: {proba[1] * 100}%;"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Raw Input Details
    with st.expander("📋 View raw input data sent to model"):
        st.dataframe(
            X,
            use_container_width=True,
            height=250
        )

    # Divider
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Technical Information
    st.markdown("""
    <div style="background: linear-gradient(135deg, #141824 0%, #1a1f3a 100%); border: 1px solid rgba(100, 200, 255, 0.15); border-radius: 10px; padding: 1.5rem; margin-top: 2rem;">
        <div style="color: #64c8ff; font-weight: 600; margin-bottom: 0.8rem; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 1px;">
            ⚙ Model Information
        </div>
        <div style="color: #a0a9be; font-size: 0.9rem; line-height: 1.6;">
            <div>• <strong>Algorithm:</strong> Random Forest Classifier</div>
            <div>• <strong>Features:</strong> 14 flight & routing parameters</div>
            <div>• <strong>Target:</strong> ArrDel15 (arrival delay ≥ 15 minutes)</div>
            <div>• <strong>Preprocessing:</strong> StandardScaler normalization</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ----------------------------
# Footer
# ----------------------------

st.markdown("""
<div class="footer-section">
    <div class="footer-text">
        ✈️ Flight Delay Prediction Engine
    </div>
    <div class="footer-text" style="color: #4b5563; font-size: 0.8rem;">
        Powered by Random Forest ML Model | Data Science Project
    </div>
    <div class="footer-text" style="color: #4b5563; font-size: 0.8rem;">
        Predictions based on historical flight data patterns and operational parameters
    </div>
</div>
""", unsafe_allow_html=True)