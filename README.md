# Air Quality Impact on Productivity

A machine learning project that analyzes and predicts the impact of air quality on workplace productivity using regression models.

##  Project Overview

This project explores the relationship between air quality parameters (PM2.5, PM10, CO, NO₂, O₃) and workplace productivity scores. It uses multiple regression models to predict productivity based on environmental conditions.

##  Features

- **Data Generation**: Synthetic dataset with realistic air quality patterns
- **Exploratory Data Analysis**: Comprehensive visualizations and statistical analysis
- **Multiple ML Models**: Linear Regression, Random Forest, Gradient Boosting
- **Feature Engineering**: Composite AQI, comfort scores, temporal features
- **Interactive Dashboard**: Streamlit app with real-time predictions
- **Visual Analytics**: Gauge charts, comparisons, and insights


## 📁 Project Structure

air-quality-productivity/
├── air_quality.ipynb              # Jupyter notebook for data analysis & model training
├── air.py                         # Streamlit deployment application
├── README.md                      # Project documentation
├── air_quality_productivity.csv   # Generated dataset (after running notebook)
├── productivity_model.pkl         # Trained model (after running notebook)
├── scaler.pkl                     # Feature scaler (after running notebook)
└── feature_columns.pkl            # Feature list (after running notebook)


##  Dataset Features

### Air Quality Parameters
- **PM2.5**: Fine particulate matter (µg/m³)
- **PM10**: Coarse particulate matter (µg/m³)
- **CO**: Carbon Monoxide (ppm)
- **NO₂**: Nitrogen Dioxide (ppb)
- **O₃**: Ozone (ppb)

### Environmental Parameters
- **Temperature**: Ambient temperature (°C)
- **Humidity**: Relative humidity (%)
- **Work Hours**: Daily work duration

### Temporal Features
- **Day of Week**: Monday-Friday
- **Season**: Spring, Summer, Fall, Winter

### Target Variable
- **Productivity Score**: 0-100 scale

##  Machine Learning Models

### Models Implemented
1. **Linear Regression**
   - Baseline model
   - Fast predictions
   - Interpretable coefficients

2. **Random Forest Regressor**
   - Handles non-linear relationships
   - Feature importance analysis
   - Robust to outliers

3. **Gradient Boosting Regressor**
   - Best performance
   - Sequential learning
   - High accuracy

### Model Evaluation Metrics
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R² Score** (Coefficient of Determination)

##  Model Performance

Typical performance metrics (varies with random seed):

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| Linear Regression | ~4.5 | ~3.5 | ~0.75 |
| Random Forest | ~3.8 | ~2.9 | ~0.82 |
| Gradient Boosting | ~3.5 | ~2.7 | ~0.85 |



