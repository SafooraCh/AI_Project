import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Weather Prediction App", page_icon="🌤️", layout="wide")

# Title and description
st.title("🌤️ Seattle Weather Prediction App")
st.markdown("### Predict weather conditions using Machine Learning models")

# Sidebar for file upload and model selection
st.sidebar.header("⚙️ Configuration")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload Weather Dataset (CSV)", type=['csv'])

# Initialize session state for models
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
    st.session_state.models = {}
    st.session_state.scaler = None
    st.session_state.le = None
    st.session_state.model_scores = {}

# Load and process data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Display dataset info
    with st.expander("📊 Dataset Overview"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", len(df))
        col2.metric("Features", len(df.columns))
        col3.metric("Weather Types", df['weather'].nunique())
        
        st.dataframe(df.head(10))
        
        # Weather distribution
        fig = px.pie(df, names='weather', title='Weather Type Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    # Preprocessing
    if st.sidebar.button("🔄 Train Models", type="primary"):
        with st.spinner("Training models... Please wait"):
            # Encoding
            le = LabelEncoder()
            df['weather_encoded'] = le.fit_transform(df['weather'])
            
            # Features and target
            X = df[['precipitation', 'temp_max', 'temp_min', 'wind']]
            y = df['weather_encoded']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train models
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Support Vector Machine": SVC(kernel='linear', probability=True),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
            }
            
            model_scores = {}
            trained_models = {}
            
            for name, model in models.items():
                if name == "Random Forest":
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                else:
                    model.fit(X_train_scaled, y_train)
                    predictions = model.predict(X_test_scaled)
                
                acc = accuracy_score(y_test, predictions)
                model_scores[name] = acc
                trained_models[name] = model
            
            # Store in session state
            st.session_state.models_trained = True
            st.session_state.models = trained_models
            st.session_state.scaler = scaler
            st.session_state.le = le
            st.session_state.model_scores = model_scores
            st.session_state.X_test = X_test
            st.session_state.X_test_scaled = X_test_scaled
            st.session_state.y_test = y_test
            
            st.success("✅ Models trained successfully!")
            st.rerun()

# Main app functionality
if st.session_state.models_trained:
    
    # Display model performance
    st.header("📈 Model Performance Comparison")
    
    col1, col2, col3 = st.columns(3)
    scores = st.session_state.model_scores
    
    with col1:
        st.metric("Logistic Regression", f"{scores['Logistic Regression']:.2%}")
    with col2:
        st.metric("Support Vector Machine", f"{scores['Support Vector Machine']:.2%}")
    with col3:
        st.metric("Random Forest", f"{scores['Random Forest']:.2%}")
    
    # Bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=list(scores.keys()),
            y=list(scores.values()),
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
            text=[f"{v:.2%}" for v in scores.values()],
            textposition='auto',
        )
    ])
    fig.update_layout(
        title="Model Accuracy Comparison",
        xaxis_title="Model",
        yaxis_title="Accuracy",
        yaxis_range=[0, 1],
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Model selection and prediction
    st.header("🔮 Make Predictions")
    
    selected_model = st.selectbox(
        "Select Model for Prediction",
        options=list(st.session_state.models.keys()),
        index=2  # Default to Random Forest
    )
    
    # Input features
    col1, col2 = st.columns(2)
    
    with col1:
        precipitation = st.slider("Precipitation (mm)", 0.0, 50.0, 5.0, 0.1)
        temp_max = st.slider("Maximum Temperature (°C)", -10.0, 40.0, 20.0, 0.5)
    
    with col2:
        temp_min = st.slider("Minimum Temperature (°C)", -10.0, 40.0, 10.0, 0.5)
        wind = st.slider("Wind Speed (km/h)", 0.0, 15.0, 5.0, 0.1)
    
    # Predict button
    if st.button("🎯 Predict Weather", type="primary"):
        # Prepare input
        input_data = np.array([[precipitation, temp_max, temp_min, wind]])
        
        # Get model
        model = st.session_state.models[selected_model]
        
        # Make prediction
        if selected_model == "Random Forest":
            prediction = model.predict(input_data)
            probabilities = model.predict_proba(input_data)[0]
        else:
            input_scaled = st.session_state.scaler.transform(input_data)
            prediction = model.predict(input_scaled)
            probabilities = model.predict_proba(input_scaled)[0]
        
        # Decode prediction
        predicted_weather = st.session_state.le.inverse_transform(prediction)[0]
        
        # Display result
        st.success(f"### Predicted Weather: **{predicted_weather.upper()}** 🌈")
        
        # Probability distribution
        st.subheader("Prediction Probabilities")
        prob_df = pd.DataFrame({
            'Weather': st.session_state.le.classes_,
            'Probability': probabilities
        }).sort_values('Probability', ascending=False)
        
        fig = px.bar(
            prob_df,
            x='Weather',
            y='Probability',
            color='Probability',
            color_continuous_scale='Viridis',
            text=prob_df['Probability'].apply(lambda x: f'{x:.2%}')
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed model analysis
    with st.expander("📊 Detailed Model Analysis"):
        selected_analysis_model = st.selectbox(
            "Select Model for Detailed Analysis",
            options=list(st.session_state.models.keys()),
            key="analysis_model"
        )
        
        model = st.session_state.models[selected_analysis_model]
        
        # Make predictions on test set
        if selected_analysis_model == "Random Forest":
            y_pred = model.predict(st.session_state.X_test)
        else:
            y_pred = model.predict(st.session_state.X_test_scaled)
        
        # Classification report
        st.subheader("Classification Report")
        report = classification_report(
            st.session_state.y_test,
            y_pred,
            target_names=st.session_state.le.classes_,
            output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.highlight_max(axis=0, color='lightgreen'))
        
        # Confusion matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(st.session_state.y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=st.session_state.le.classes_,
            yticklabels=st.session_state.le.classes_,
            ax=ax
        )
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix: {selected_analysis_model}')
        st.pyplot(fig)

else:
    st.info("👆 Please upload a dataset and click 'Train Models' to get started!")
    
    # Sample data format
    st.subheader("Expected Data Format")
    sample_data = pd.DataFrame({
        'date': ['2012-01-01', '2012-01-02', '2012-01-03'],
        'precipitation': [0.0, 10.9, 0.8],
        'temp_max': [12.8, 10.6, 11.7],
        'temp_min': [5.0, 2.8, 7.2],
        'wind': [4.7, 4.5, 2.3],
        'weather': ['drizzle', 'rain', 'sun']
    })
    st.dataframe(sample_data)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**How to use:**
1. Upload your weather dataset (CSV)
2. Click 'Train Models'
3. Select a model
4. Adjust weather parameters
5. Click 'Predict Weather'
""")
