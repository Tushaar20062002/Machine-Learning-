import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="CardioPredict | Heart Disease Risk Assessment",
    page_icon="heart",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Load artifacts
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts(artifact_version):
    model = joblib.load(BASE_DIR / "SVM.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    columns = joblib.load(BASE_DIR / "columns.pkl")
    try:
        metrics = joblib.load(BASE_DIR / "metrics.pkl")
    except FileNotFoundError:
        metrics = {"accuracy": None, "f1": None}

    if not hasattr(model, "predict"):
        raise TypeError("SVM.pkl must contain a trained model with a predict method.")
    if not hasattr(scaler, "transform"):
        raise TypeError("scaler.pkl must contain a fitted scaler with a transform method.")
    if not isinstance(columns, list) or not columns:
        raise TypeError("columns.pkl must contain the training feature names as a non-empty list.")

    return model, scaler, columns, metrics


artifact_version = tuple(
    (BASE_DIR / name).stat().st_mtime
    for name in ("SVM.pkl", "scaler.pkl", "columns.pkl")
)
model, scaler, columns, metrics = load_artifacts(artifact_version)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        html, body, [class*="css"]  {
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        .main > div {
            padding-top: 1.2rem;
        }

        .hero {
            background: linear-gradient(135deg, #7f1d1d 0%, #b91c1c 45%, #dc2626 100%);
            padding: 2.2rem 2.5rem;
            border-radius: 18px;
            color: white;
            margin-bottom: 1.6rem;
            box-shadow: 0 10px 30px rgba(185, 28, 28, 0.25);
        }
        .hero h1 {
            margin: 0;
            font-size: 2.1rem;
            font-weight: 800;
            letter-spacing: -0.02em;
        }
        .hero p {
            margin-top: 0.5rem;
            font-size: 1.02rem;
            opacity: 0.92;
            max-width: 700px;
        }

        .section-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: inherit;
            margin-bottom: 0.9rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .metric-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: #fef2f2;
            color: #b91c1c;
            padding: 0.35rem 0.85rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
            margin-right: 0.5rem;
        }

        .result-box-high {
            background: linear-gradient(135deg, #fee2e2, #fecaca);
            border: 1px solid #fca5a5;
            border-radius: 16px;
            padding: 1.8rem;
            text-align: center;
        }
        .result-box-low {
            background: linear-gradient(135deg, #dcfce7, #bbf7d0);
            border: 1px solid #86efac;
            border-radius: 16px;
            padding: 1.8rem;
            text-align: center;
        }
        .result-title-high { color: #991b1b; font-size: 1.6rem; font-weight: 800; margin: 0; }
        .result-title-low { color: #166534; font-size: 1.6rem; font-weight: 800; margin: 0; }
        .result-sub { color: #4b5563; font-size: 0.95rem; margin-top: 0.4rem; }

        .disclaimer {
            background: #fffbeb;
            border: 1px solid #fde68a;
            color: #92400e;
            padding: 0.9rem 1.1rem;
            border-radius: 12px;
            font-size: 0.85rem;
            margin-top: 1rem;
        }

        .stButton > button {
            background: linear-gradient(135deg, #b91c1c, #dc2626);
            color: white;
            font-weight: 700;
            border-radius: 10px;
            padding: 0.7rem 1.2rem;
            border: none;
            width: 100%;
            font-size: 1rem;
            box-shadow: 0 4px 14px rgba(220, 38, 38, 0.35);
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #991b1b, #b91c1c);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>CardioPredict - Heart Disease Risk Assessment</h1>
        <p>An SVM-based clinical decision-support tool that estimates the likelihood
        of heart disease from routine patient measurements. Enter the details below
        to generate an instant risk assessment.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Model Information")
    if metrics.get("accuracy") is not None:
        st.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
        st.metric("F1 Score", f"{metrics['f1']*100:.1f}%")
    st.markdown("**Algorithm:** Support Vector Machine (SVC)")
    st.markdown("**Training data:** 918 patient records")
    st.markdown("---")
    st.markdown("### About")
    st.write(
        "This tool uses a machine learning model trained on the UCI Heart "
        "Failure Prediction dataset. It considers 11 clinical features "
        "including age, chest pain type, cholesterol, and ECG results."
    )
    st.markdown("---")
    st.markdown("### Feature Guide")
    with st.expander("Chest Pain Types"):
        st.write(
            "- **ATA**: Atypical Angina\n"
            "- **NAP**: Non-Anginal Pain\n"
            "- **ASY**: Asymptomatic\n"
            "- **TA**: Typical Angina"
        )
    with st.expander("ST Slope"):
        st.write("Slope of the peak exercise ST segment: **Up**, **Flat**, or **Down**")
    with st.expander("Resting ECG"):
        st.write(
            "- **Normal**\n"
            "- **ST**: ST-T wave abnormality\n"
            "- **LVH**: Left ventricular hypertrophy"
        )

# ----------------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------------
st.markdown("### Patient Details")

with st.form("patient_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=45, step=1)
        sex = st.selectbox("Sex", ["Male", "Female"])
        chest_pain = st.selectbox(
            "Chest Pain Type",
            ["ATA (Atypical Angina)", "NAP (Non-Anginal Pain)", "ASY (Asymptomatic)", "TA (Typical Angina)"],
        )
        resting_bp = st.number_input(
            "Resting Blood Pressure (mm Hg)", min_value=60, max_value=250, value=130, step=1
        )

    with col2:
        cholesterol = st.number_input(
            "Cholesterol (mg/dL)", min_value=0, max_value=700, value=200, step=1
        )
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL?", ["No", "Yes"])
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_hr = st.number_input(
            "Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150, step=1
        )

    with col3:
        exercise_angina = st.selectbox("Exercise-Induced Angina?", ["No", "Yes"])
        oldpeak = st.number_input(
            "Oldpeak (ST depression)", min_value=-3.0, max_value=7.0, value=0.0, step=0.1, format="%.1f"
        )
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
        st.write("")
        st.write("")

    submitted = st.form_submit_button("Assess Risk")



# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
if submitted:
    chest_pain_code = chest_pain.split(" ")[0]  # "ATA", "NAP", "ASY", "TA"

    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": 1 if fasting_bs == "Yes" else 0,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_M": 1 if sex == "Male" else 0,
        "ChestPainType_ATA": 1 if chest_pain_code == "ATA" else 0,
        "ChestPainType_NAP": 1 if chest_pain_code == "NAP" else 0,
        "ChestPainType_TA": 1 if chest_pain_code == "TA" else 0,
        "RestingECG_Normal": 1 if resting_ecg == "Normal" else 0,
        "RestingECG_ST": 1 if resting_ecg == "ST" else 0,
        "ExerciseAngina_Y": 1 if exercise_angina == "Yes" else 0,
        "ST_Slope_Flat": 1 if st_slope == "Flat" else 0,
        "ST_Slope_Up": 1 if st_slope == "Up" else 0,
    }

    missing_columns = [column for column in columns if column not in raw_input]
    if missing_columns:
        st.error(f"The app is missing model input columns: {', '.join(missing_columns)}")
        st.stop()

    input_df = pd.DataFrame([raw_input], columns=columns)
    input_scaled = scaler.transform(input_df)

    prediction = int(model.predict(input_scaled)[0])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_scaled)[0]
    else:
        proba = np.array([1 - prediction, prediction], dtype=float)
    risk_score = proba[1] * 100

    st.markdown("### Assessment Result")
    result_col, gauge_col = st.columns([1.3, 1])

    with result_col:
        if prediction == 1:
            st.markdown(
                f"""
                <div class="result-box-high">
                    <p class="result-title-high">Elevated Risk Detected</p>
                    <p class="result-sub">Estimated probability of heart disease: <strong>{risk_score:.1f}%</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-box-low">
                    <p class="result-title-low">Low Risk</p>
                    <p class="result-sub">Estimated probability of heart disease: <strong>{risk_score:.1f}%</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="disclaimer">
                <strong>Disclaimer:</strong> This tool provides a statistical estimate only and is
                not a medical diagnosis. Always consult a qualified healthcare professional for
                clinical evaluation and decisions.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with gauge_col:
        st.markdown("**Risk Probability**")
        st.progress(min(int(risk_score), 100))
        st.caption(f"{risk_score:.1f}% likelihood of heart disease")

        st.markdown("**Probability Breakdown**")
        prob_df = pd.DataFrame(
            {"Outcome": ["No Disease", "Heart Disease"], "Probability": [proba[0] * 100, proba[1] * 100]}
        ).set_index("Outcome")
        st.bar_chart(prob_df)

    with st.expander("View submitted clinical values"):
        display_df = pd.DataFrame(
            {
                "Feature": [
                    "Age", "Sex", "Chest Pain Type", "Resting BP", "Cholesterol",
                    "Fasting Blood Sugar > 120", "Resting ECG", "Max Heart Rate",
                    "Exercise Angina", "Oldpeak", "ST Slope",
                ],
                "Value": [
                    age, sex, chest_pain, f"{resting_bp} mm Hg", f"{cholesterol} mg/dL",
                    fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope,
                ],
            }
        )
        st.table(display_df.set_index("Feature"))
else:
    st.info("Fill in the patient details above and click **Assess Risk** to generate a prediction.")





