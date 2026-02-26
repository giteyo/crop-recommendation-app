import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="Crop Predictor", layout="centered")

st.title("🌱 Crop Recommendation System")

# 1. Check if the model file actually exists in the cloud folder
if not os.path.exists('crop_pipeline.joblib'):
    st.error("Model file 'crop_pipeline.joblib' not found in the repository!")
else:
    # 2. Input Fields (Always visible)
    st.subheader("Enter Soil & Environmental Details")
    
    col1, col2 = st.columns(2)
    with col1:
        n = st.number_input("Nitrogen (N)", 0, 150, 50)
        p = st.number_input("Phosphorus (P)", 0, 150, 50)
        k = st.number_input("Potassium (K)", 0, 210, 50)
    with col2:
        temp = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
        hum = st.number_input("Humidity (%)", 0.0, 100.0, 80.0)
        rain = st.number_input("Rainfall (mm)", 0.0, 300.0, 200.0)
    
    ph = st.slider("Soil pH", 0.0, 14.0, 6.5)

    # 3. Prediction Logic
    if st.button("Get Recommendation"):
        try:
            model = joblib.load('crop_pipeline.joblib')
            # Ensure the order matches your training data (N, P, K, Temp, Hum, pH, Rain)
            features = np.array([[n, p, k, temp, hum, ph, rain]])
            prediction = model.predict(features)
            
            st.success(f"### Result: The best crop is **{prediction[0]}**")
            st.balloons()
        except Exception as e:
            st.error(f"Prediction Error: {e}")
