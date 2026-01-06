import streamlit as st
import pandas as pd
import numpy as np
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
        st.subheader("Weather Type Distribution")
        weather_counts = df['weather'].value_counts()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = sns.color_palette('Set2', len(weather_counts))
        ax.pie(weather_counts.values, labels=weather_counts.index, autopct='%1.1f%%',
               startangle=90, colors=colors)
        ax.set_title('Weather Type Distribution')
        st.pyplot(fig)
        plt.close()
    
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
            
            progress_bar = st.progress(0)
            for idx, (name, model) in enumerate(models.items()):
                if name == "Random Forest":
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                else:
                    model.fit(X_train_scaled, y_train)
                    predictions = model.predict(X_test_scaled)
                
                acc = accuracy_score(y_test, predictions)
                model_scores[name] = acc
                trained_models[name] = model
                progress_bar.progress((idx + 1) / len(models))
            
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
    fig, ax = plt.subplots(figsize=(10, 5))
    models_list = list(scores.keys())
    values_list = list(scores.values())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars = ax.bar(models_list, values_list, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
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
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        colors_gradient = plt.cm.viridis(prob_df['Probability'].values)
        bars = ax.barh(prob_df['Weather'], prob_df['Probability'], color=colors_gradient)
        
        # Add percentage labels
        for i, (weather, prob) in enumerate(zip(prob_df['Weather'], prob_df['Probability'])):
            ax.text(prob + 0.01, i, f'{prob:.2%}', va='center', fontweight='bold')
        
        ax.set_xlabel('Probability', fontsize=12)
        ax.set_ylabel('Weather Type', fontsize=12)
        ax.set_title(f'Prediction Probabilities - {selected_model}', fontsize=14, fontweight='bold')
        ax.set_xlim(0, max(prob_df['Probability']) + 0.1)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Show input summary
        with st.expander("📝 Input Summary"):
            input_df = pd.DataFrame({
                'Feature': ['Precipitation', 'Max Temperature', 'Min Temperature', 'Wind Speed'],
                'Value': [f"{precipitation} mm", f"{temp_max} °C", f"{temp_min} °C", f"{wind} km/h"]
            })
            st.table(input_df)
    
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
        st.dataframe(report_df.style.background_gradient(cmap='YlGn', subset=['precision', 'recall', 'f1-score']))
        
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
            ax=ax,
            cbar_kws={'label': 'Count'}
        )
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title(f'Confusion Matrix: {selected_analysis_model}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

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

**Models Available:**
- Logistic Regression
- Support Vector Machine
- Random Forest
""")

st.sidebar.markdown("---")
st.sidebar.success("Built with Streamlit 🎈")
