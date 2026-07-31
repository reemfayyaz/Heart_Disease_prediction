import streamlit as st
import pandas as pd
import pickle

# ==========================
# Load Trained Model
# ==========================
with open("Heart_disease_model.pkl", "rb") as file:
    model = pickle.load(file)

# ==========================
# Streamlit Page
# ==========================
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction System")
st.write("Enter the patient details below and click Predict.")

# ==========================
# Input Fields
# ==========================

patient_id = st.number_input("Patient ID", value=300001)

age = st.number_input("Age", min_value=1, max_value=120, value=45)

gender = st.selectbox(
    "Gender",
    [0, 1],
    format_func=lambda x: "Female" if x == 0 else "Male"
)

chest_pain = st.selectbox(
    "Chest Pain Type",
    [0, 1, 2, 3]
)

resting_bp = st.number_input(
    "Resting Blood Pressure",
    min_value=50,
    max_value=250,
    value=120
)

cholesterol = st.number_input(
    "Cholesterol",
    min_value=50,
    max_value=700,
    value=200
)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

max_hr = st.number_input(
    "Maximum Heart Rate",
    min_value=50,
    max_value=250,
    value=150
)

exercise_angina = st.selectbox(
    "Exercise Angina",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

oldpeak = st.number_input(
    "Oldpeak",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)

smoking = st.selectbox(
    "Smoking",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

# ==========================
# Prediction
# ==========================

if st.button("Predict"):

    input_df = pd.DataFrame({
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

    prediction = model.predict(input_df)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")