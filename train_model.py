"""
Landslide Prediction Model Training Script
==========================================

This script trains a Support Vector Machine (SVM) model to predict landslide risk
based on rainfall and soil moisture data.

Author: Shoeb Alam
Date: 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_and_prepare_data():
    """Load and prepare the dataset for training."""
    print("Loading dataset...")
    
    # Load the dataset
    df = pd.read_csv('data/soil_rainfall_landslide_dataset.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nDataset Info:")
    print(df.info())
    print("\nDataset Statistics:")
    print(df.describe())
    
    # Check for missing values
    missing_values = df.isnull().sum()
    print(f"\nMissing values:\n{missing_values}")
    
    if missing_values.sum() > 0:
        print("Handling missing values...")
        df = df.dropna()
    
    return df

def train_svm_model(df):
    """Train the SVM model."""
    print("\nPreparing features and target...")
    
    # Prepare features (X) and target (y)
    # Using the correct column names from the dataset
    X = df[['Rainfall_mm', 'Soil_Moisture_Percentage']]
    y = df['Landslide']
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # Split the data into training and testing sets
    print("\nSplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Standardize the features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the SVM model
    print("\nTraining SVM model...")
    model = SVC(
        probability=True,  # Enable probability estimates
        kernel='rbf',      # Radial Basis Function kernel
        random_state=42,
        C=1.0,            # Regularization parameter
        gamma='scale'      # Kernel coefficient
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Landslide', 'Landslide']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model, scaler, accuracy

def save_model_and_scaler(model, scaler, accuracy):
    """Save the trained model and scaler."""
    print("\nSaving model and scaler...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    model_path = 'models/landslide_model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save the scaler
    scaler_path = 'models/feature_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")
    
    # Save model info
    model_info = {
        'accuracy': accuracy,
        'features': ['Rainfall_mm', 'Soil_Moisture_Percentage'],
        'target': 'Landslide',
        'model_type': 'SVM',
        'kernel': 'rbf'
    }
    
    info_path = 'models/model_info.pkl'
    joblib.dump(model_info, info_path)
    print(f"Model info saved to: {info_path}")
    
    return model_path, scaler_path, info_path

def main():
    """Main function to orchestrate the training process."""
    print("🌋 Landslide Prediction Model Training")
    print("=" * 50)
    
    try:
        # Load and prepare data
        df = load_and_prepare_data()
        
        # Train the model
        model, scaler, accuracy = train_svm_model(df)
        
        # Save the model and scaler
        model_path, scaler_path, info_path = save_model_and_scaler(model, scaler, accuracy)
        
        print("\n" + "=" * 50)
        print("✅ Training completed successfully!")
        print(f"📊 Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"💾 Model saved to: {model_path}")
        print(f"🔧 Scaler saved to: {scaler_path}")
        print(f"📋 Model info saved to: {info_path}")
        print("\n🚀 Ready to use the model in the Streamlit app!")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
