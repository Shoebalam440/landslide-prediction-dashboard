# 🌋 Landslide Prediction Dashboard

A comprehensive machine learning application that predicts landslide risk using Support Vector Machine (SVM) based on rainfall and soil moisture data. The application features an interactive Streamlit dashboard with real-time risk assessment and data visualization.

## 🎯 Features

- **Real-time Risk Prediction**: Input rainfall and soil moisture values to get instant landslide risk assessment
- **Risk Classification**: Categorizes risk into Low, Medium, and High levels based on prediction probability
- **Interactive Dashboard**: Beautiful and responsive Streamlit interface with data visualizations
- **Data Visualization**: Comprehensive charts showing dataset distribution and patterns
- **Model Performance**: Displays model accuracy and performance metrics
- **Responsive Design**: Works seamlessly on desktop and mobile devices

## 🧠 Machine Learning Model

- **Algorithm**: Support Vector Machine (SVM) with RBF kernel
- **Features**: Rainfall (mm) and Soil Moisture (%)
- **Target**: Binary classification (Landslide: 1, No Landslide: 0)
- **Risk Levels**:
  - 🟢 **Low Risk**: < 40% probability
  - 🟡 **Medium Risk**: 40-70% probability  
  - 🔴 **High Risk**: > 70% probability

## 📁 Project Structure

```
landslide_prediction_app/
│
├── data/
│   └── soil_rainfall_landslide_dataset.csv    # Training dataset
│
├── models/
│   ├── landslide_model.pkl                    # Trained SVM model
│   ├── feature_scaler.pkl                     # Feature scaler
│   └── model_info.pkl                         # Model metadata
│
├── train_model.py                             # Model training script
├── app.py                                     # Streamlit dashboard
├── requirements.txt                           # Python dependencies
└── README.md                                  # Project documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Shoebalam440/landslide-prediction-dashboard.git
   cd landslide-prediction-dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**
   ```bash
   python train_model.py
   ```

4. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, manually navigate to the URL shown in the terminal

## 📊 Usage

### Training the Model

Run the training script to create the machine learning model:

```bash
python train_model.py
```

This will:
- Load and analyze the dataset
- Split data into training and testing sets
- Train the SVM model with optimal parameters
- Evaluate model performance
- Save the trained model and scaler to the `models/` directory

### Using the Dashboard

1. **Input Parameters**:
   - Enter rainfall amount in millimeters (0-1000 mm)
   - Enter soil moisture percentage (0-100%)

2. **Get Prediction**:
   - Click the "🔮 Predict Risk" button
   - View the risk level classification
   - See the probability percentage

3. **Explore Data**:
   - View dataset statistics and distributions
   - Analyze rainfall vs soil moisture patterns
   - Understand model performance metrics

## 📈 Model Performance

The SVM model achieves high accuracy in predicting landslide risk:

- **Algorithm**: Support Vector Machine with RBF kernel
- **Cross-validation**: Stratified train-test split (80/20)
- **Feature Scaling**: StandardScaler for optimal performance
- **Hyperparameters**: Optimized for best accuracy

## 🛠️ Technical Details

### Dependencies

- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning library
- **joblib**: Model serialization
- **plotly**: Interactive visualizations

### Model Architecture

```python
SVM(
    kernel='rbf',
    probability=True,
    C=1.0,
    gamma='scale',
    random_state=42
)
```

### Feature Engineering

- **Standardization**: Features are scaled using StandardScaler
- **Feature Selection**: Rainfall and Soil Moisture as primary predictors
- **Data Validation**: Missing value handling and outlier detection

## 🌐 Deployment

### Streamlit Cloud Deployment

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Select the main branch and `app.py` as the main file
   - Deploy!

### Local Deployment

For local deployment with custom configurations:

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## 📊 Dataset Information

The dataset contains historical landslide data with the following features:

- **Rainfall_mm**: Rainfall amount in millimeters
- **Soil_Moisture_Percentage**: Soil moisture content as percentage
- **Landslide**: Binary target variable (1: Landslide occurred, 0: No landslide)

## 🔧 Customization

### Adding New Features

To add new features to the model:

1. Update the dataset with new columns
2. Modify `train_model.py` to include new features
3. Update the input fields in `app.py`
4. Retrain the model

### Modifying Risk Thresholds

Adjust risk classification thresholds in `app.py`:

```python
def classify_risk_level(probability):
    if probability < 0.3:  # Adjust threshold
        return "Low", "🟢 Low Risk - Safe"
    elif 0.3 <= probability < 0.6:  # Adjust threshold
        return "Medium", "🟡 Medium Risk - Moderate, Stay Alert"
    else:
        return "High", "🔴 High Risk - ⚠️ High Landslide Probability! Take Precautions."
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Shoeb Alam**
- GitHub: [@Shoebalam440](https://github.com/Shoebalam440)
- LinkedIn: [Shoeb Alam](https://www.linkedin.com/in/shoeb-alam-6079b0257)

## 🙏 Acknowledgments

- Scikit-learn team for the excellent machine learning library
- Streamlit team for the amazing web app framework
- Plotly team for interactive visualizations
- The open-source community for continuous support

## 📞 Support

If you have any questions or need help with the application, please:

1. Check the [Issues](https://github.com/Shoebalam440/landslide-prediction-dashboard/issues) page
2. Create a new issue with detailed description
3. Contact the author via LinkedIn

---

⭐ **Star this repository if you found it helpful!**
