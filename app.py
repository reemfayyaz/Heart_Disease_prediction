import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")
st.title("❤️ Heart Disease Prediction")
st.write("Enter patient details below.")

@st.cache_resource
def load_model():
    """Try to find and load the trained model file.
    Looks for any .pkl file with "heart" in the filename first, then falls back to common names.
    Raises FileNotFoundError if no candidate is found.
    """
    cwd_files = os.listdir()
    # Prefer files with 'heart' in the name
    for fname in cwd_files:
        if fname.lower().endswith('.pkl') and 'heart' in fname.lower():
            return joblib.load(fname)
    # explicit fallbacks
    for fname in ("Heart_disease_model.pkl", "heart_disease_model.pkl"):
        if os.path.exists(fname):
            return joblib.load(fname)
    raise FileNotFoundError("No heart disease model .pkl found in the current directory.")

try:
    model = load_model()

    age = st.number_input("Age", 1, 120, 45)
    sex = 1 if st.selectbox("Sex", ["Male", "Female"]) == "Male" else 0
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 50, 300, 120)
    chol = st.number_input("Cholesterol", 50, 1000, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG", [0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved", 20, 300, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST depression induced by exercise relative to rest (Oldpeak)", 0.0, 10.0, 1.0, step=0.1)
    slope = st.selectbox("Slope of the peak exercise ST segment", [0, 1, 2])
    ca = st.selectbox("Number of major vessels (0-4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

    if st.button("Predict"):
        X = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                         columns=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"])
        pred = model.predict(X)[0]
        if int(pred) == 1:
            st.error("Heart Disease Detected")
        else:
            st.success("No Heart Disease Detected")

except FileNotFoundError as e:
    pkl_files = [f for f in os.listdir() if f.lower().endswith('.pkl')]
    if pkl_files:
        st.warning(f"Model file not found under expected names. Found .pkl files: {pkl_files}. Rename your model to include 'heart' in the filename or to 'Heart_disease_model.pkl'.")
    else:
        st.warning("heart disease model .pkl not found. Please add the trained model (e.g. 'Heart_disease_model.pkl') to the repository directory.")
except Exception as e:
    st.error(f"An unexpected error occurred while loading the model: {e}")
