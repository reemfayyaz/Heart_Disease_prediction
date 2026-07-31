
# ❤️ Heart Disease Prediction System

A Streamlit web application that predicts whether a patient is likely to have heart disease using a trained Logistic Regression machine learning model.

## Features

- Predicts heart disease risk
- Simple and interactive Streamlit interface
- Displays prediction result
- Shows prediction probabilities (when supported)
- Easy to deploy on Streamlit Cloud

## Project Structure

```
Heart_Disease_Project/
│── app.py
│── Heart_disease_model.pkl
│── requirements.txt
│── README.md
```

## Installation

1. Clone or download this project.
2. Install the required packages:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install streamlit scikit-learn numpy joblib
```

## Run the Application

```bash
streamlit run app.py
```

The application will automatically open in your browser.

## Input Features

- Patient ID
- Age
- Gender
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Maximum Heart Rate
- Exercise Induced Angina
- Oldpeak
- Smoking

## Prediction Output

- ✅ No Heart Disease
- ⚠️ Heart Disease Detected

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- NumPy
- Joblib

## Author

Developed as a Machine Learning & Streamlit project.
