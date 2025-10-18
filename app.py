"""
🌋 Landslide Prediction Dashboard
================================

A Streamlit web application that predicts landslide risk using
Support Vector Machine (SVM) based on rainfall and soil moisture data.

Author: Shoeb Alam
Date: 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Page configuration
st.set_page_config(
    page_title="🌋 Landslide Prediction Dashboard",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .low-risk {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .medium-risk {
        background-color: #fff3cd;
        color: #856404;
        border: 2px solid #ffeaa7;
    }
    .high-risk {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model_and_scaler():
    """Load the trained model and scaler with caching."""
    try:
        model = joblib.load('models/landslide_model.pkl')
        scaler = joblib.load('models/feature_scaler.pkl')
        model_info = joblib.load('models/model_info.pkl')
        return model, scaler, model_info
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.error("Please run 'python train_model.py' first to train the model.")
        return None, None, None

@st.cache_data
def load_dataset():
    """Load the dataset with caching."""
    try:
        df = pd.read_csv('data/soil_rainfall_landslide_dataset.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure the dataset is in the 'data' folder.")
        return None

def predict_landslide_risk(model, scaler, rainfall, soil_moisture):
    """Predict landslide risk based on input parameters."""
    # Prepare input data
    input_data = np.array([[rainfall, soil_moisture]])
    
    # Scale the input data
    input_scaled = scaler.transform(input_data)
    
    # Get prediction probability
    proba = model.predict_proba(input_scaled)[0]
    landslide_probability = proba[1]  # Probability of landslide
    
    return landslide_probability

def classify_risk_level(probability):
    """Classify risk level based on probability."""
    if probability < 0.4:
        return "Low", "🟢 Low Risk - Safe"
    elif 0.4 <= probability < 0.7:
        return "Medium", "🟡 Medium Risk - Moderate, Stay Alert"
    else:
        return "High", "🔴 High Risk - ⚠️ High Landslide Probability! Take Precautions."

def create_risk_visualization(probability, risk_level):
    """Create a visual representation of the risk level."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Landslide Risk Probability (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🌋 Landslide Prediction Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Predict landslide risk based on rainfall and soil moisture data")
    
    # Load model and data
    model, scaler, model_info = load_model_and_scaler()
    df = load_dataset()
    
    if model is None or df is None:
        st.stop()
    
    # Sidebar
    st.sidebar.title("📋 About")
    st.sidebar.markdown("**Developer:** Shoeb Alam")
    st.sidebar.markdown("**Model:** Support Vector Machine (SVM)")
    st.sidebar.markdown("**Algorithm:** RBF Kernel")
    st.sidebar.markdown(f"**Model Accuracy:** {model_info['accuracy']:.2%}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Information")
    st.sidebar.write("This model predicts landslide risk levels using:")
    st.sidebar.write("• **Rainfall (mm)**")
    st.sidebar.write("• **Soil Moisture (%)**")
    st.sidebar.write("• **Risk Classification:**")
    st.sidebar.write("  - 🟢 Low: < 40%")
    st.sidebar.write("  - 🟡 Medium: 40-70%")
    st.sidebar.write("  - 🔴 High: > 70%")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📝 Input Parameters")
        
        # Input fields
        rainfall = st.number_input(
            "Enter Rainfall (mm)",
            min_value=0.0,
            max_value=1000.0,
            value=50.0,
            step=0.1,
            help="Enter the rainfall amount in millimeters"
        )
        
        soil_moisture = st.number_input(
            "Enter Soil Moisture (%)",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=0.1,
            help="Enter the soil moisture percentage"
        )
        
        # Predict button
        if st.button("🔮 Predict Risk", type="primary", use_container_width=True):
            with st.spinner("Analyzing data..."):
                # Make prediction
                probability = predict_landslide_risk(model, scaler, rainfall, soil_moisture)
                risk_level, risk_message = classify_risk_level(probability)
                
                # Display results
                st.markdown("### 🎯 Prediction Results")
                
                # Risk level display
                if risk_level == "Low":
                    st.markdown(f'<div class="prediction-box low-risk">{risk_message}</div>', unsafe_allow_html=True)
                elif risk_level == "Medium":
                    st.markdown(f'<div class="prediction-box medium-risk">{risk_message}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-box high-risk">{risk_message}</div>', unsafe_allow_html=True)
                
                # Probability display
                st.metric(
                    label="Landslide Probability",
                    value=f"{probability:.1%}",
                    delta=f"{probability*100:.1f}%"
                )
                
                # Store results in session state for visualization
                st.session_state.prediction_results = {
                    'probability': probability,
                    'risk_level': risk_level,
                    'rainfall': rainfall,
                    'soil_moisture': soil_moisture
                }
    
    with col2:
        st.markdown("### 📊 Risk Visualization")
        
        if 'prediction_results' in st.session_state:
            results = st.session_state.prediction_results
            fig = create_risk_visualization(results['probability'], results['risk_level'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👆 Enter parameters and click 'Predict Risk' to see the visualization")
    
    # Dataset visualization
    st.markdown("---")
    st.markdown("### 📈 Dataset Overview")
    
    if df is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Records", f"{len(df):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            landslide_count = df['Landslide'].sum()
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Landslide Events", f"{landslide_count:,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            landslide_rate = (df['Landslide'].sum() / len(df)) * 100
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Landslide Rate", f"{landslide_rate:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Scatter plot
        fig_scatter = px.scatter(
            df,
            x='Rainfall_mm',
            y='Soil_Moisture_Percentage',
            color='Landslide',
            title='Rainfall vs Soil Moisture (Colored by Landslide Risk)',
            labels={
                'Rainfall_mm': 'Rainfall (mm)',
                'Soil_Moisture_Percentage': 'Soil Moisture (%)',
                'Landslide': 'Landslide Risk'
            },
            color_discrete_map={0: 'green', 1: 'red'},
            opacity=0.7
        )
        
        fig_scatter.update_layout(
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Distribution plots
        col1, col2 = st.columns(2)
        
        with col1:
            fig_rainfall = px.histogram(
                df,
                x='Rainfall_mm',
                color='Landslide',
                title='Rainfall Distribution',
                labels={'Rainfall_mm': 'Rainfall (mm)', 'count': 'Frequency'},
                color_discrete_map={0: 'green', 1: 'red'}
            )
            st.plotly_chart(fig_rainfall, use_container_width=True)
        
        with col2:
            fig_moisture = px.histogram(
                df,
                x='Soil_Moisture_Percentage',
                color='Landslide',
                title='Soil Moisture Distribution',
                labels={'Soil_Moisture_Percentage': 'Soil Moisture (%)', 'count': 'Frequency'},
                color_discrete_map={0: 'green', 1: 'red'}
            )
            st.plotly_chart(fig_moisture, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #666; padding: 2rem;">'
        'Developed by <strong>Shoeb Alam</strong> | '
        '🌋 Landslide Prediction Dashboard | '
        'Powered by Machine Learning'
        '</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
