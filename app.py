import warnings

import joblib
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, .stApp, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    .stApp p, .stApp li, .stApp label, .stApp button, .stApp input,
    .stApp textarea, .stApp [data-baseweb="select"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    .stApp {
        background: #ffffff;
    }

    .block-container {
        max-width: 940px;
        padding-top: 3.2rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3, h4 {
        color: #0f172a;
        letter-spacing: -0.02em;
    }

    .hero-badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: #0f766e;
        background: #f0fdfa;
        border: 1px solid #99f6e4;
        border-radius: 999px;
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.02em;
        margin: 0.9rem 0 0.5rem;
    }

    .hero-subtitle {
        font-size: 0.98rem;
        color: #64748b;
        line-height: 1.65;
        margin: 0 0 2.2rem;
        max-width: 46rem;
    }

    .section-heading {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin: 2rem 0 1rem;
        padding-bottom: 0.55rem;
        border-bottom: 1px solid #e2e8f0;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: #0f766e;
    }

    .section-heading::before {
        content: "";
        width: 3px;
        height: 15px;
        background: #14b8a6;
        border-radius: 2px;
    }

    [data-testid="stWidgetLabel"] p {
        font-size: 0.84rem;
        font-weight: 600;
        color: #334155;
    }

    div[data-baseweb="select"] > div,
    [data-testid="stNumberInput"] input {
        border-radius: 8px;
        border-color: #cbd5e1;
    }

    div[data-baseweb="select"] > div:hover,
    [data-testid="stNumberInput"] input:hover {
        border-color: #94a3b8;
    }

    div[data-testid="stForm"] {
        border: none;
        padding: 0;
    }

    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.55rem 1.15rem;
        border: 1px solid #cbd5e1;
        color: #334155;
        background: #ffffff;
    }

    .stButton > button:hover {
        border-color: #94a3b8;
        color: #0f172a;
        background: #f8fafc;
    }

    .stButton > button[kind="primary"] {
        background: #0f766e;
        border: 1px solid #0f766e;
        color: #ffffff;
    }

    .stButton > button[kind="primary"]:hover {
        background: #115e59;
        border-color: #115e59;
        color: #ffffff;
    }

    div[data-testid="stFormSubmitButton"] {
        margin-top: 1.8rem;
    }

    div[data-testid="stFormSubmitButton"] button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.55rem 1.15rem;
        border: 1px solid #0f766e;
        background: #0f766e;
        color: #ffffff;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        background: #115e59;
        border-color: #115e59;
        color: #ffffff;
    }

    .result-panel {
        margin-top: 2rem;
        padding: 1.5rem 1.6rem;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #0f766e;
        border-radius: 10px;
    }

    .result-eyebrow {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 0.45rem;
    }

    .result-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin: 0 0 0.25rem;
    }

    .result-prob {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.02em;
        margin: 0.2rem 0 0.15rem;
    }

    .result-caption {
        font-size: 0.84rem;
        color: #64748b;
        margin-bottom: 1rem;
    }

    .result-track {
        height: 8px;
        background: #e2e8f0;
        border-radius: 999px;
        overflow: hidden;
    }

    .result-fill {
        height: 100%;
        border-radius: 999px;
    }

    .result-meta {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        margin-top: 0.75rem;
        font-size: 0.82rem;
        color: #475569;
    }

    @media (max-width: 640px) {
        .block-container {
            padding-top: 1.9rem;
            padding-left: 1rem;
            padding-right: 1rem;
            padding-bottom: 2.5rem;
        }

        .hero-title {
            font-size: 1.6rem;
        }

        .hero-subtitle {
            margin-bottom: 1.5rem;
        }

        .result-prob {
            font-size: 1.8rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

model = joblib.load("KNN_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

CHEST_PAIN_LABELS = {
    "ATA": "Atypical Angina (ATA)",
    "NAP": "Non-Anginal Pain (NAP)",
    "ASY": "Asymptomatic (ASY)",
    "TA": "Typical Angina (TA)",
}

st.markdown(
    """
    <span class="hero-badge">Clinical Risk Assessment</span>
    <div class="hero-title">Heart Disease Prediction</div>
    <div class="hero-subtitle">
        Enter the patient's clinical details below. The assessment uses a
        pre-trained K-Nearest Neighbors model to estimate the likelihood of
        heart disease.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("prediction_form", clear_on_submit=False):
    st.markdown('<div class="section-heading">Patient Profile</div>', unsafe_allow_html=True)
    profile_col1, profile_col2, profile_col3 = st.columns(3)
    with profile_col1:
        age = st.slider("Age (years)", 1, 100, 45)
    with profile_col2:
        gender = st.selectbox("Sex", ["Male", "Female"])
    with profile_col3:
        chest_pain = st.selectbox(
            "Chest Pain Type",
            list(CHEST_PAIN_LABELS),
            format_func=CHEST_PAIN_LABELS.get,
        )

    st.markdown(
        '<div class="section-heading">Clinical Measurements</div>',
        unsafe_allow_html=True,
    )
    measure_col1, measure_col2, measure_col3 = st.columns(3)
    with measure_col1:
        resting_blood_pressure = st.number_input(
            "Resting Blood Pressure (mm Hg)", 80, 200, 120, step=1
        )
    with measure_col2:
        cholestrol = st.number_input(
            "Cholesterol (mg/dL)", 100, 600, 200, step=1
        )
    with measure_col3:
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])

    st.markdown(
        '<div class="section-heading">Test Results</div>',
        unsafe_allow_html=True,
    )
    test_col1, test_col2, test_col3 = st.columns(3)
    with test_col1:
        resting_ecg = st.selectbox(
            "Resting ECG",
            ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"],
        )
    with test_col2:
        max_heart_rate = st.slider("Maximum Heart Rate", 60, 220, 150)
    with test_col3:
        oldpeak = st.slider("Old Peak (ST Depression)", 0.0, 6.2, 0.0, step=0.1)

    test_col4, test_col5, _ = st.columns(3)
    with test_col4:
        exercise_induced_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    with test_col5:
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    submitted = st.form_submit_button("Predict Risk", type="primary")

if submitted:
    input_data = pd.DataFrame(
        {
            "Age": [age],
            "RestingBP": [resting_blood_pressure],
            "Cholesterol": [cholestrol],
            "FastingBS": [1 if fasting_bs == "Yes" else 0],
            "MaxHR": [max_heart_rate],
            "Oldpeak": [oldpeak],
            "Sex_M": [1 if gender == "Male" else 0],
            "ChestPainType_ATA": [1 if chest_pain == "ATA" else 0],
            "ChestPainType_NAP": [1 if chest_pain == "NAP" else 0],
            "ChestPainType_TA": [1 if chest_pain == "TA" else 0],
            "RestingECG_Normal": [1 if resting_ecg == "Normal" else 0],
            "RestingECG_ST": [1 if resting_ecg == "ST-T Wave Abnormality" else 0],
            "ExerciseAngina_Y": [1 if exercise_induced_angina == "Yes" else 0],
            "ST_Slope_Flat": [1 if st_slope == "Flat" else 0],
            "ST_Slope_Up": [1 if st_slope == "Up" else 0],
        }
    )
    input_data = input_data[expected_columns]

    scaled_data = scaler.transform(input_data)
    prediction = int(model.predict(scaled_data)[0])
    disease_index = list(model.classes_).index(1)
    probability = float(model.predict_proba(scaled_data)[0][disease_index])

    st.session_state["prediction"] = (prediction, probability)

if "prediction" in st.session_state:
    prediction, probability = st.session_state["prediction"]
    percentage = min(max(probability * 100, 0.0), 100.0)
    risk_level = (
        "Low" if probability < 0.3 else "Moderate" if probability < 0.7 else "High"
    )

    if prediction == 1:
        title = "Heart Disease Detected"
        accent = "#b91c1c"
    else:
        title = "No Heart Disease Detected"
        accent = "#047857"

    st.markdown(
        f"""
        <div class="result-panel" style="border-left-color: {accent};">
            <div class="result-eyebrow">Assessment Result</div>
            <div class="result-title" style="color: {accent};">{title}</div>
            <div class="result-prob">{percentage:.1f}%</div>
            <div class="result-caption">Estimated probability of heart disease</div>
            <div class="result-track">
                <div class="result-fill" style="width: {percentage:.1f}%; background: {accent};"></div>
            </div>
            <div class="result-meta">
                <span>Risk level: <strong>{risk_level}</strong></span>
                <span>Model: K-Nearest Neighbors</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "This assessment is for informational purposes only and is not a medical diagnosis. "
        "Please consult a qualified healthcare professional for medical advice."
    )
