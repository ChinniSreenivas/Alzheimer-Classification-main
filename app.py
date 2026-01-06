import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(
    page_title="Alzheimer’s Disease Prediction",
    layout="centered"
)

# Load Model
try:
    model = joblib.load("best_model.pkl")
except Exception as e:
    st.error("❌ Model loading failed")
    st.exception(e)
    st.stop()

# Get Feature Names from Model
FEATURES = model.feature_names_in_

# App Title
st.title("🧠 Alzheimer’s Disease Prediction App")

st.markdown("""
Predict the **risk of Alzheimer’s Disease** using a trained  
Machine Learning classification model.

⚠️ *Academic & research use only.*
""")

# Sidebar Inputs (MATCH DATASET)
st.sidebar.header("🧾 Patient Details")

def user_input_features():
    input_data = {}

    for feature in FEATURES:
        # Binary features
        if feature in [
            "Gender", "Smoking", "Diabetes", "Hypertension",
            "CardiovascularDisease", "MemoryComplaints", "BehavioralProblems"
        ]:
            input_data[feature] = st.sidebar.selectbox(
                f"{feature}", [0, 1]
            )

        # Age
        elif feature == "Age":
            input_data[feature] = st.sidebar.slider("Age", 50, 100, 70)

        # MMSE
        elif feature == "MMSE":
            input_data[feature] = st.sidebar.slider("MMSE Score", 0, 30, 26)

        # ADL / Functional
        elif feature in ["ADL", "FunctionalAssessment"]:
            input_data[feature] = st.sidebar.slider(feature, 0.0, 10.0, 8.0)

        # Cholesterol values
        elif "Cholesterol" in feature:
            input_data[feature] = st.sidebar.slider(feature, 50, 400, 200)

        # Default numeric
        else:
            input_data[feature] = st.sidebar.number_input(feature, value=0.0)

    return pd.DataFrame([input_data])

# Collect Input
input_df = user_input_features()

# Ensure correct column order
input_df = input_df[FEATURES]

st.subheader("📌 Patient Input Summary")
st.dataframe(input_df, use_container_width=True)

# Prediction
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)[0][1]

# Result
st.subheader("🧪 Prediction Result")

if prediction == 1:
    st.error("⚠️ **High Risk of Alzheimer’s Disease Detected**")
else:
    st.success("✅ **Low Risk of Alzheimer’s Disease**")

# Risk Confidence
st.markdown("### 📊 Risk Confidence")
st.progress(int(prediction_proba * 100))
st.info(f"Model Confidence: **{prediction_proba * 100:.2f}%**")

# Clinical Interpretation
st.markdown("### 🩺 Clinical Interpretation")

if prediction_proba >= 0.75:
    st.warning("""
    - Strong cognitive risk indicators  
    - Immediate clinical evaluation recommended  
    - Early diagnosis helps slow progression
    """)
elif prediction_proba >= 0.40:
    st.info("""
    - Moderate cognitive risk  
    - Lifestyle improvement advised  
    - Regular monitoring recommended
    """)
else:
    st.success("""
    - Cognitive health appears stable  
    - Maintain healthy habits  
    - Routine checkups suggested
    """)

# Footer
st.markdown("---")
st.caption("Developed by **Kusuma K** | Alzheimer’s Disease Prediction | Streamlit ML App")
