import streamlit as st
import numpy as np
import pickle

st.title("Seattle Weather Prediction")

# Load model files
model = pickle.load(open("seattle_weather_best_model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

# User inputs
precipitation = st.number_input("Precipitation")
temp_max = st.number_input("Max Temperature")
temp_min = st.number_input("Min Temperature")
wind = st.number_input("Wind Speed")

# Predict button
if st.button("Predict"):
    input_data = np.array([[precipitation, temp_max, temp_min, wind]])
    prediction = model.predict(input_data)
    weather = encoder.inverse_transform(prediction)

    st.success(f"Predicted Weather: {weather[0]}")
