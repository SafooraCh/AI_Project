import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Weather Prediction App",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 1.5rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        with open('logistic_regression_model.pkl', 'rb') as f:
            log_reg = pickle.load(f)
        with open('random_forest_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        with open('linear_regression_model.pkl', 'rb') as f:
            lin_reg = pickle.load(f)
        with open('scaler_classification.pkl', 'rb') as f:
            scaler_class = pickle.load(f)
        with open('scaler_regression.pkl', 'rb') as f:
            scaler_reg = pickle.load(f)
        with open('feature_info.pkl', 'rb') as f:
            feature_info = pickle.load(f)
        return log_reg, rf_model, lin_reg, scaler_class, scaler_reg, feature_info
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please ensure all .pkl files are in the app directory.")
        return None, None, None, None, None, None

# Load models
log_reg, rf_model, lin_reg, scaler_class, scaler_reg, feature_info = load_models()

# Title and description
st.title("🌦️ Weather Prediction & Analysis System")
st.markdown("### AI-Powered Weather Forecasting | By Safoora (2330-0022)")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📋 Project Information")
    st.info("""
    **Project:** Weather Prediction using ML
    
    **Student:** Safoora
    
    **Roll No:** 2330-0022
    
    **Models Used:**
    - Logistic Regression
    - Random Forest
    - Linear Regression
    """)
    
    st.markdown("---")
    st.header("🎯 Prediction Type")
    prediction_type = st.radio(
        "Select prediction task:",
        ["Rain Prediction (Classification)", "Temperature Prediction (Regression)"]
    )

# Main content
if prediction_type == "Rain Prediction (Classification)":
    st.header("🌧️ Rain Prediction")
    st.markdown("Predict whether it will rain based on weather conditions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Weather Parameters")
        
        precipitation = st.slider(
            "Precipitation (mm)",
            min_value=0.0,
            max_value=50.0,
            value=0.0,
            step=0.1,
            help="Amount of precipitation in millimeters"
        )
        
        temp_max = st.slider(
            "Maximum Temperature (°C)",
            min_value=-10.0,
            max_value=40.0,
            value=15.0,
            step=0.5
        )
        
        temp_min = st.slider(
            "Minimum Temperature (°C)",
            min_value=-15.0,
            max_value=30.0,
            value=8.0,
            step=0.5
        )
        
        wind = st.slider(
            "Wind Speed",
            min_value=0.0,
            max_value=10.0,
            value=3.0,
            step=0.1
        )
    
    with col2:
        st.subheader("Date Information")
        
        selected_date = st.date_input(
            "Select Date",
            value=datetime.now(),
            min_value=datetime(2020, 1, 1),
            max_value=datetime(2030, 12, 31)
        )
        
        month = selected_date.month
        day_of_year = selected_date.timetuple().tm_yday
        temp_range = temp_max - temp_min
        
        st.metric("Month", month)
        st.metric("Day of Year", day_of_year)
        st.metric("Temperature Range", f"{temp_range:.1f}°C")
        
        model_choice = st.selectbox(
            "Select Model",
            ["Random Forest (Recommended)", "Logistic Regression"]
        )
    
    st.markdown("---")
    
    if st.button("🔮 Predict Rain", type="primary", use_container_width=True):
        if log_reg and rf_model and scaler_class:
            # Prepare input data
            input_data = np.array([[precipitation, temp_max, temp_min, wind, 
                                   month, day_of_year, temp_range]])
            input_scaled = scaler_class.transform(input_data)
            
            # Make prediction
            if model_choice == "Random Forest (Recommended)":
                prediction = rf_model.predict(input_scaled)[0]
                probability = rf_model.predict_proba(input_scaled)[0]
            else:
                prediction = log_reg.predict(input_scaled)[0]
                probability = log_reg.predict_proba(input_scaled)[0]
            
            # Display results
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if prediction == 1:
                    st.markdown("""
                        <div style='background-color: #1e3a8a; padding: 2rem; 
                        border-radius: 1rem; text-align: center;'>
                            <h1 style='color: white; margin: 0;'>🌧️ RAIN EXPECTED</h1>
                            <p style='color: #93c5fd; font-size: 1.2rem; margin-top: 1rem;'>
                                High probability of rainfall
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div style='background-color: #047857; padding: 2rem; 
                        border-radius: 1rem; text-align: center;'>
                            <h1 style='color: white; margin: 0;'>☀️ NO RAIN</h1>
                            <p style='color: #86efac; font-size: 1.2rem; margin-top: 1rem;'>
                                Clear weather expected
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Probability visualization
            st.subheader("📊 Prediction Confidence")
            
            fig = go.Figure(go.Bar(
                x=['No Rain', 'Rain'],
                y=[probability[0] * 100, probability[1] * 100],
                text=[f'{probability[0]*100:.1f}%', f'{probability[1]*100:.1f}%'],
                textposition='auto',
                marker_color=['#10b981', '#3b82f6']
            ))
            
            fig.update_layout(
                title="Probability Distribution",
                yaxis_title="Probability (%)",
                xaxis_title="Prediction",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature contribution
            st.subheader("🔍 Input Summary")
            summary_data = {
                'Feature': ['Precipitation', 'Max Temp', 'Min Temp', 'Wind Speed', 
                           'Month', 'Day of Year', 'Temp Range'],
                'Value': [f'{precipitation} mm', f'{temp_max}°C', f'{temp_min}°C', 
                         f'{wind}', month, day_of_year, f'{temp_range:.1f}°C']
            }
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

else:  # Temperature Prediction
    st.header("🌡️ Temperature Prediction")
    st.markdown("Predict maximum temperature based on weather conditions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Weather Parameters")
        
        precipitation = st.slider(
            "Precipitation (mm)",
            min_value=0.0,
            max_value=50.0,
            value=0.0,
            step=0.1
        )
        
        temp_min = st.slider(
            "Minimum Temperature (°C)",
            min_value=-15.0,
            max_value=30.0,
            value=8.0,
            step=0.5
        )
        
        wind = st.slider(
            "Wind Speed",
            min_value=0.0,
            max_value=10.0,
            value=3.0,
            step=0.1
        )
    
    with col2:
        st.subheader("Date Information")
        
        selected_date = st.date_input(
            "Select Date",
            value=datetime.now(),
            min_value=datetime(2020, 1, 1),
            max_value=datetime(2030, 12, 31)
        )
        
        month = selected_date.month
        day_of_year = selected_date.timetuple().tm_yday
        
        st.metric("Month", month)
        st.metric("Day of Year", day_of_year)
    
    st.markdown("---")
    
    if st.button("🔮 Predict Temperature", type="primary", use_container_width=True):
        if lin_reg and scaler_reg:
            # Prepare input data
            input_data = np.array([[precipitation, temp_min, wind, month, day_of_year]])
            input_scaled = scaler_reg.transform(input_data)
            
            # Make prediction
            predicted_temp = lin_reg.predict(input_scaled)[0]
            
            # Display result
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                st.markdown(f"""
                    <div style='background-color: #dc2626; padding: 2rem; 
                    border-radius: 1rem; text-align: center;'>
                        <h1 style='color: white; margin: 0;'>🌡️ {predicted_temp:.1f}°C</h1>
                        <p style='color: #fca5a5; font-size: 1.2rem; margin-top: 1rem;'>
                            Predicted Maximum Temperature
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Temperature gauge
            st.subheader("📊 Temperature Gauge")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=predicted_temp,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Predicted Max Temperature (°C)"},
                delta={'reference': temp_min, 'increasing': {'color': "red"}},
                gauge={
                    'axis': {'range': [-10, 40]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [-10, 0], 'color': "lightblue"},
                        {'range': [0, 15], 'color': "lightgreen"},
                        {'range': [15, 25], 'color': "yellow"},
                        {'range': [25, 40], 'color': "orange"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': predicted_temp
                    }
                }
            ))
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Input summary
            st.subheader("🔍 Input Summary")
            summary_data = {
                'Feature': ['Precipitation', 'Min Temp', 'Wind Speed', 'Month', 'Day of Year'],
                'Value': [f'{precipitation} mm', f'{temp_min}°C', f'{wind}', month, day_of_year]
            }
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
            
            # Additional insights
            st.info(f"""
            **Insights:**
            - Temperature range: {predicted_temp - temp_min:.1f}°C
            - Season: {'Winter' if month in [12, 1, 2] else 'Spring' if month in [3, 4, 5] else 'Summer' if month in [6, 7, 8] else 'Autumn'}
            - Wind influence: {'High' if wind > 5 else 'Moderate' if wind > 3 else 'Low'}
            """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #6b7280;'>
        <p>Weather Prediction System | AI Semester Project | Safoora (2330-0022)</p>
        <p>Powered by Machine Learning | Data Source: Seattle Weather Dataset</p>
    </div>
""", unsafe_allow_html=True)
