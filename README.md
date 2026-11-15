# 🌍 Air Quality and Health Impact Analysis

> A comprehensive Data Science project that investigates the relationship between **air pollution** and **public health outcomes**.  
> The project performs exploratory data analysis, hypothesis testing, multi-algorithm machine learning modeling, and deployment of a web application to predict the **Health Impact Score** across the full spectrum of environmental and health conditions.

---

## 🧭 Quick Navigation

| Section | Description |
|---------|-------------|
| [📂 Project Structure](#-project-structure) | Overview of project directories and files |
| [🎯 Objective](#-objective) | Goals and target variables |
| [📊 Dataset Description](#-dataset-description) | Full dataset attributes and features |
| [🔬 ML Pipeline & Modeling](#-ml-pipeline--modeling) | Algorithms, preprocessing, and leakage avoidance |
| [📈 Model Evaluation Results](#-model-evaluation-results) | Performance metrics (R², MAE, MSE, RMSE) |
| [💻 Web Application & API](#-web-application--api) | Flask app setup, UI features, and prediction endpoints |
| [🚀 Quick Start Guide](#-quick-start-guide) | Step-by-step setup and execution |

---

## 📂 Project Structure

```
Air-Pollution-and-Health-Analysis-main/
│
├── data/
│   └── processed/
│       ├── health_impact_cleaned.csv         # Complete processed health impact dataset (5,811 rows)
│       ├── city_day_cleaned.csv              # Cleaned city daily data
│       └── city_day_feature_engineered.csv   # Feature engineered city dataset
│
├── models/
│   └── health_impact_model.pkl               # Saved scikit-learn Pipeline (Scaler + LightGBM Regressor)
│
├── static/
│   ├── style.css                             # Modern responsive UI stylesheet
│   ├── feature_importance.png                # Relative feature importance plot
│   ├── actual_vs_predicted.png               # Actual vs Predicted fit and error distribution
│   └── model_comparison.png                  # Multi-model evaluation comparison chart
│
├── templates/
│   └── index.html                            # Flask UI with sample data loader, validation & error cards
│
├── app.py                                    # Flask web application & REST API
├── train_model.py                            # Standalone end-to-end model training script
├── script.py                                 # Original exploration script
├── notebook.ipynb                            # Jupyter analysis notebook
├── requirements.txt                          # Python dependencies
└── README.md                                 # Documentation
```

---

## 🎯 Objective & Problem Definition

- **Problem Type:** Supervised Machine Learning – Regression
- **Target Variable:** `HealthImpactScore` (Continuous, range ~20 to 100)
- **Features (12):**
  1. `AQI`: Air Quality Index
  2. `PM10`: Particulate Matter ≤ 10μm (μg/m³)
  3. `PM2_5`: Particulate Matter ≤ 2.5μm (μg/m³)
  4. `NO2`: Nitrogen Dioxide concentration (μg/m³)
  5. `SO2`: Sulfur Dioxide concentration (μg/m³)
  6. `O3`: Ozone concentration (μg/m³)
  7. `Temperature`: Ambient temperature (°C)
  8. `Humidity`: Relative humidity (%)
  9. `WindSpeed`: Average wind speed (m/s)
  10. `RespiratoryCases`: Reported respiratory disease cases
  11. `CardiovascularCases`: Reported cardiovascular cases
  12. `HospitalAdmissions`: Hospital admission counts

---

## 🔬 ML Pipeline & Modeling

To ensure optimal generalization and avoid data leakage:
1. **Target & Identity Exclusion:** Non-predictive identifier `RecordID` and categorical label `HealthImpactClass` are removed from input features.
2. **Train/Test Split:** Data is split into **80% training** (4,648 rows) and **20% testing** (1,163 rows) with `random_state=42`.
3. **Leakage-Free Preprocessing:** `StandardScaler` is wrapped together with the regression model in a scikit-learn `Pipeline`, ensuring that scaling parameters are learned strictly on `X_train`.
4. **Compared Algorithms:**
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
   - Gradient Boosting Regressor
   - XGBoost Regressor
   - LightGBM Regressor

---

## 📈 Model Evaluation Results

| Model | Train R² | Test R² | Test MAE | Test MSE | Test RMSE |
|:------|:--------:|:-------:|:--------:|:--------:|:---------:|
| **LightGBM** ⭐ | **0.9970** | **0.9766** | **1.0758** | **4.3862** | **2.0943** |
| **Gradient Boosting** | 0.9954 | 0.9702 | 1.2171 | 5.5775 | 2.3617 |
| **XGBoost** | 0.9947 | 0.9692 | 1.2463 | 5.7678 | 2.4016 |
| **Random Forest** | 0.9905 | 0.9471 | 1.5276 | 9.9076 | 3.1476 |
| **Decision Tree** | 0.9614 | 0.8693 | 2.2860 | 24.4826 | 4.9480 |
| **Linear Regression** | 0.4902 | 0.5053 | 7.3541 | 92.6829 | 9.6272 |

> ⭐ **Best Model Selected:** `LightGBM Regressor` achieved the lowest Test RMSE (`2.0943`), lowest Test MAE (`1.0758`), and highest Test R² (`0.9766`), demonstrating outstanding predictive accuracy and generalization across all ranges of pollution.

### 🌐 Behavior Across External Data Scenarios
Unlike previous truncated models that capped at 100, the updated pipeline reliably differentiates severity:
- **Clean Air / Low Risk:** Score **19.71** (Healthy, mild impact)
- **Moderate Pollution / Average Risk:** Score **68.36** (Noticeable impact on sensitive groups)
- **Severe Air Pollution / Hospital Distress:** Score **99.93** (Critical health risk)

---

## 💻 Web Application & API

The Flask application (`app.py`) provides both an accessible web UI and a JSON REST API for predictions.

### 1. Web Interface
Access `http://127.0.0.1:5000/` in your browser. Enter the 12 features (or click **"📋 Fill Sample Data"**) and click **Predict** to view the calculated score.

### 2. REST API Endpoint (`/predict`)
Send a `POST` request with JSON payload:
```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "AQI": 120.0,
       "PM10": 75.0,
       "PM25": 45.0,
       "NO2": 50.0,
       "SO2": 25.0,
       "O3": 75.0,
       "Temperature": 18.0,
       "Humidity": 55.0,
       "WindSpeed": 6.0,
       "RespiratoryCases": 8,
       "CardiovascularCases": 3,
       "HospitalAdmissions": 1
     }'
```

**Response:**
```json
{
  "health_impact_score": 68.36,
  "raw_score": 68.3582,
  "success": true
}
```

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Machine Learning Pipeline
```bash
python train_model.py
```
This trains all models, selects the best performer, saves `models/health_impact_model.pkl`, and writes evaluation figures to `static/`.

### 3. Run the Flask Web Application
```bash
python app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.
