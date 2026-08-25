import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

# -------------------------------
# Load model safely
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("model.pkl was not found. Put model.pkl in the same folder as app.py.")
    st.stop()
except Exception as e:
    st.error(f"Could not load model.pkl: {e}")
    st.stop()

# -------------------------------
# Title
# -------------------------------
st.title("🩺 Diabetes Prediction App")
st.write(
    "Enter the patient's information below to predict the diabetes outcome."
)

st.info(
    "This application is for educational purposes only "
    "and should not be used as a medical diagnosis."
)

# -------------------------------
# Input section
# -------------------------------
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=1,
        step=1
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0,
        max_value=300,
        value=120,
        step=1
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0,
        max_value=200,
        value=70,
        step=1
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0,
        max_value=100,
        value=20,
        step=1
    )

with col2:
    insulin = st.number_input(
        "Insulin",
        min_value=0,
        max_value=1000,
        value=80,
        step=1
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        step=0.1
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.47,
        step=0.01
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30,
        step=1
    )

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict Diabetes", type="primary"):

    input_data = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]],
        columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]
    )

    try:
        prediction = int(model.predict(input_data)[0])
        probability = float(model.predict_proba(input_data)[0][1])

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ Prediction: Diabetic")
        else:
            st.success("✅ Prediction: Not Diabetic")

        st.write(f"Probability of diabetes: {probability * 100:.2f}%")

        st.subheader("Patient Data")
        st.dataframe(input_data, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
