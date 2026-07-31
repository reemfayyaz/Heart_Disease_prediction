import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

@st.cache_resource
def load_model():
    return joblib.load("Heart_disease_model.pkl")

model = load_model()

st.title("❤️ Heart Disease Prediction System")
st.write("Enter patient details below and click Predict.")

patient_id = st.number_input("Patient ID", value=300001)
age = st.number_input("Age", 1, 120, 45)
gender = st.selectbox("Gender", [0,1], format_func=lambda x: "Female" if x==0 else "Male")
chest_pain = st.selectbox("Chest Pain Type", [0,1,2,3])
resting_bp = st.number_input("Resting Blood Pressure", 50, 250, 120)
cholesterol = st.number_input("Cholesterol", 50, 700, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar", [0,1])
max_hr = st.number_input("Maximum Heart Rate", 60, 250, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", [0,1])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0, 0.1)
smoking = st.selectbox("Smoking", [0,1])

if st.button("Predict"):
    features = np.array([[patient_id, age, gender, chest_pain, resting_bp,
                          cholesterol, fasting_bs, max_hr,
                          exercise_angina, oldpeak, smoking]])
    pred = model.predict(features)[0]
    if pred == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")
    try:
        prob = model.predict_proba(features)[0]
        st.write(f"No Heart Disease: {prob[0]*100:.2f}%")
        st.write(f"Heart Disease: {prob[1]*100:.2f}%")
        st.progress(float(prob[1]))
    except Exception:
        pass
