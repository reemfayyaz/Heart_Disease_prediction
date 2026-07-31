import streamlit as st
import pandas as pd
import pickle

# ==========================
# Load Model
# ==========================
model = pickle.load(open("Heart_disease_model.pkl", "rb"))

# ==========================
# Page Settings
# ==========================
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Prediction App")
st.markdown("Predict whether a patient is likely to have heart disease using Machine Learning.")

st.sidebar.header("Enter Patient Details")

# ==========================
# Inputs
# ==========================

patient_id = st.sidebar.number_input(
    "Patient ID",
    value=300001,
    step=1
)

age = st.sidebar.slider(
    "Age",
    18,
    100,
    45
)

gender = st.sidebar.selectbox(
    "Gender",
    ("Male", "Female")
)

gender = 1 if gender == "Male" else 0

chest_pain = st.sidebar.selectbox(
    "Chest Pain Type",
    [0, 1, 2, 3]
)

resting_bp = st.sidebar.number_input(
    "Resting Blood Pressure",
    min_value=80,
    max_value=250,
    value=120
)

cholesterol = st.sidebar.number_input(
    "Cholesterol",
    min_value=100,
    max_value=700,
    value=200
)

fasting_bs = st.sidebar.selectbox(
    "Fasting Blood Sugar",
    ("No", "Yes")
)

fasting_bs = 1 if fasting_bs == "Yes" else 0

max_hr = st.sidebar.number_input(
    "Maximum Heart Rate",
    min_value=60,
    max_value=220,
    value=150
)

exercise_angina = st.sidebar.selectbox(
    "Exercise Angina",
    ("No", "Yes")
)

exercise_angina = 1 if exercise_angina == "Yes" else 0

oldpeak = st.sidebar.number_input(
    "Oldpeak",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)

smoking = st.sidebar.selectbox(
    "Smoking",
    ("No", "Yes")
)

smoking = 1 if smoking == "Yes" else 0

# ==========================
# Prediction
# ==========================

if st.button("🔍 Predict"):

    input_data = pd.DataFrame({
        "Patient_ID": [patient_id],
        "Age": [age],
        "Gender": [gender],
        "Chest_Pain_Type": [chest_pain],
        "Resting_BP": [resting_bp],
        "Cholesterol": [cholesterol],
        "Fasting_BS": [fasting_bs],
        "Max_Heart_Rate": [max_hr],
        "Exercise_Angina": [exercise_angina],
        "Oldpeak": [oldpeak],
        "Smoking": [smoking]
    })

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.subheader("Input Data")

    st.dataframe(input_data)

st.markdown("---")
st.caption("Developed using Streamlit and Scikit-Learn")