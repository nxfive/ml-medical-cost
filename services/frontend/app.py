import os

import requests
import streamlit as st


st.title("Medical Cost Prediction")
st.markdown("Enter data:")

age = st.slider("Age", 18, 100, 35)
bmi = st.number_input("BMI", value=25.0)
children = st.slider("Children", 0, 15, 0)

smoker = st.selectbox("Smoker", ["Yes", "No"])
sex = st.selectbox("Sex", ["Male", "Female"])
region = st.selectbox("Region", ["Southwest", "Southeast", "Northwest", "Northeast"])

input_data = {
    "age": age,
    "bmi": bmi,
    "children": children,
    "smoker": smoker.lower(),
    "sex": sex.lower(),
    "region": region.lower(),
}

backend_host = os.getenv("BACKEND_HOST", "127.0.0.1")
backend_port = os.getenv("BACKEND_PORT", "3000")

if st.button("Predict Medical Cost"):
    response = requests.post(
        f"http://{backend_host}:{backend_port}/predict", json=input_data
    )
    if response.status_code == 200:
        result = response.json()
        predicted = result.get("charges")
        st.success(f"Your medical costs: {round(predicted, 2)}$")
    else:
        st.error("Server Error")
