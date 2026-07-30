import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")
st.title("❤️ Heart Disease Prediction")
st.write("Enter patient details below.")

@st.cache_resource
def load_model():
    return joblib.load("heart_disease_model.pkl")

try:
    model = load_model()

    age = st.number_input("Age",20,100,45)
    sex = 1 if st.selectbox("Sex",["Male","Female"])=="Male" else 0
    cp = st.selectbox("Chest Pain Type",[0,1,2,3])
    trestbps = st.number_input("Resting Blood Pressure",80,220,120)
    chol = st.number_input("Cholesterol",100,600,200)
    fbs = st.selectbox("Fasting Blood Sugar >120",[0,1])
    restecg = st.selectbox("Resting ECG",[0,1,2])
    thalach = st.number_input("Maximum Heart Rate",60,220,150)
    exang = st.selectbox("Exercise Induced Angina",[0,1])
    oldpeak = st.number_input("Oldpeak",0.0,10.0,1.0)
    slope = st.selectbox("Slope",[0,1,2])
    ca = st.selectbox("Major Vessels",[0,1,2,3,4])
    thal = st.selectbox("Thal",[0,1,2,3])

    if st.button("Predict"):
        X = pd.DataFrame([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]],
                         columns=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"])
        pred = model.predict(X)[0]
        if pred==1:
            st.error("Heart Disease Detected")
        else:
            st.success("No Heart Disease Detected")
except FileNotFoundError:
    st.warning("heart_disease_model.pkl not found.")
