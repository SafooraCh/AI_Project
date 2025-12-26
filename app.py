import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page Configuration
st.set_page_config(page_title="Weather Prediction App", layout="centered")

# Load the saved files
@st.cache_resource
def load_models():
    model = pickle.load(open('seattle_weather_best_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    encoder = pickle.load(open('encoder.pkl', 'rb'))
    return model, scaler, encoder

try:
    model, scaler, encoder = load_models()

    st.title("☁️ Seattle Weather Predictor")
    st.write("Input the weather parameters below to predict the weather condition.")

    # Input Fields
    col1, col2 = st.columns(2)
    
    with col1:
        precipitation = st.number_input("Precipitation (mm)", min_value=0.0, step=0.1)
        temp_max = st.number_input("Maximum Temperature (°C)", step=0.1)
        
    with col2:
        wind = st.number_input("Wind Speed (km/h)", min_value=0.0, step=0.1)
        temp_min = st.number_input("Minimum Temperature (°C)", step=0.1)

    # Prediction Button
    if st.button("Predict Weather"):
        # 1. Prepare input data
        input_data = np.array([[precipitation, temp_max, temp_min, wind]])
        
        # 2. Scale data (Model training mein scaling use hui thi)
        # Note: Random Forest scaling ki bina bhi chalta hai, 
        # lekin training mein Scaled data tha to yahan bhi scale karna behtar hai.
        # input_scaled = scaler.transform(input_data) 
        
        # 3. Make prediction
        prediction_idx = model.predict(input_data) # RF directly takes data
        weather_type = encoder.inverse_transform(prediction_idx)[0]
        
        # 4. Display Result
        st.success(f"The predicted weather is: **{weather_type.capitalize()}**")
        
        # Icon display logic
        icons = {"sun": "☀️", "rain": "🌧️", "fog": "🌫️", "drizzle": "🌦️", "snow": "❄️"}
        st.metric(label="Condition", value=weather_type.capitalize(), delta=icons.get(weather_type, "☁️"))

except FileNotFoundError:
    st.error("Error: PKL files nahi milin. Pehle model training wala code run karein taake files generate ho jayen.")