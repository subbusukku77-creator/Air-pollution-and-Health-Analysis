import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Base directory for reliable path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'health_impact_model.pkl')

# Define expected training feature names and order
FEATURE_COLUMNS = [
    'AQI', 'PM10', 'PM2_5', 'NO2', 'SO2', 'O3',
    'Temperature', 'Humidity', 'WindSpeed',
    'RespiratoryCases', 'CardiovascularCases', 'HospitalAdmissions'
]

# Form field to training column mapping (HTML form uses 'PM25')
FORM_TO_FEATURE_MAP = {
    'AQI': 'AQI',
    'PM10': 'PM10',
    'PM25': 'PM2_5',
    'NO2': 'NO2',
    'SO2': 'SO2',
    'O3': 'O3',
    'Temperature': 'Temperature',
    'Humidity': 'Humidity',
    'WindSpeed': 'WindSpeed',
    'RespiratoryCases': 'RespiratoryCases',
    'CardiovascularCases': 'CardiovascularCases',
    'HospitalAdmissions': 'HospitalAdmissions'
}

# Physical validity ranges for input features
VALIDATION_RANGES = {
    'AQI': (0, 1000, "AQI must be between 0 and 1000"),
    'PM10': (0, 1000, "PM10 must be between 0 and 1000 μg/m³"),
    'PM25': (0, 1000, "PM2.5 must be between 0 and 1000 μg/m³"),
    'NO2': (0, 1000, "NO2 must be between 0 and 1000 μg/m³"),
    'SO2': (0, 1000, "SO2 must be between 0 and 1000 μg/m³"),
    'O3': (0, 1000, "O3 must be between 0 and 1000 μg/m³"),
    'Temperature': (-60, 70, "Temperature must be between -60°C and 70°C"),
    'Humidity': (0, 100, "Humidity must be between 0% and 100%"),
    'WindSpeed': (0, 150, "Wind Speed must be between 0 and 150 m/s"),
    'RespiratoryCases': (0, 10000, "Respiratory cases must be non-negative"),
    'CardiovascularCases': (0, 10000, "Cardiovascular cases must be non-negative"),
    'HospitalAdmissions': (0, 10000, "Hospital admissions must be non-negative")
}

# Load the trained machine learning pipeline
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"✓ Model successfully loaded from '{MODEL_PATH}'")
    else:
        print(f"⚠️ Model file not found at '{MODEL_PATH}'. Run train_model.py first.")
except Exception as e:
    print(f"⚠️ Failed to load model: {e}")


def get_risk_tier(score):
    """
    Map predicted numerical score to risk tier, badge, theme, and guidance.
    """
    if score < 40.0:
        return {
            'tier': 'Low',
            'badge': '🟢 LOW RISK (Clean / Mild)',
            'theme': 'theme-low',
            'color': '#10b981',
            'title': 'Favorable Public Health Conditions',
            'desc': 'Air quality meets clean ambient standards. Population can engage in normal outdoor activities with minimal health impact.'
        }
    elif score < 75.0:
        return {
            'tier': 'Moderate',
            'badge': '🟡 MODERATE RISK (Noticeable Impact)',
            'theme': 'theme-mod',
            'color': '#eab308',
            'title': 'Elevated Risk for Sensitive Populations',
            'desc': 'Elevated particulate or pollutant exposure. Asthmatics, elderly individuals, and young children should monitor respiratory comfort.'
        }
    elif score < 95.0:
        return {
            'tier': 'High',
            'badge': '🟠 HIGH RISK (Substantial Impact)',
            'theme': 'theme-high',
            'color': '#f97316',
            'title': 'Unhealthy Environmental Exposure',
            'desc': 'Significant air pollution correlating with increased clinic visits, outpatient hospitalizations, and cardiopulmonary distress.'
        }
    else:
        return {
            'tier': 'Critical',
            'badge': '🔴 CRITICAL RISK (Severe Emergency)',
            'theme': 'theme-crit',
            'color': '#ef4444',
            'title': 'Hazardous Environmental Health Crisis',
            'desc': 'Acute public health emergency. Heavy atmospheric toxic saturation and emergency hospital admissions. Stay indoors and run air filtration.'
        }


def validate_and_parse_inputs(data_source):
    """
    Validate input fields: presence, numeric type, and physical bounds.
    Returns:
        parsed_dict: Dictionary of cleaned floats matching FEATURE_COLUMNS
        error_msg: String error message if validation fails, else None
    """
    parsed_features = {}

    for form_key, feat_name in FORM_TO_FEATURE_MAP.items():
        val_str = None
        if form_key in data_source:
            val_str = data_source[form_key]
        elif feat_name in data_source:
            val_str = data_source[feat_name]

        if val_str is None or str(val_str).strip() == "":
            return None, f"Missing required field: '{form_key}'"

        try:
            val = float(val_str)
        except (ValueError, TypeError):
            return None, f"Invalid numeric value for '{form_key}': '{val_str}'"

        min_v, max_v, range_err = VALIDATION_RANGES[form_key]
        if not (min_v <= val <= max_v):
            return None, range_err

        parsed_features[feat_name] = val

    return parsed_features, None


@app.route('/')
def home():
    """Render home landing page with the input form."""
    return render_template('index.html', form_data={})


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Handle prediction requests.
    Supports both HTML form submission and JSON API payloads.
    """
    if request.method == 'GET':
        return render_template('index.html', form_data={})

    is_json = request.is_json
    data_source = request.get_json(silent=True) if is_json else request.form

    if not data_source:
        error_msg = "No input data provided."
        if is_json:
            return jsonify({'success': False, 'error': error_msg}), 400
        return render_template('index.html', error_text=error_msg, form_data={})

    # Validate inputs
    parsed_features, validation_error = validate_and_parse_inputs(data_source)
    if validation_error:
        if is_json:
            return jsonify({'success': False, 'error': validation_error}), 400
        return render_template('index.html', error_text=validation_error, form_data=data_source)

    if model is None:
        error_msg = "Model is not loaded. Please ensure 'models/health_impact_model.pkl' exists."
        if is_json:
            return jsonify({'success': False, 'error': error_msg}), 500
        return render_template('index.html', error_text=error_msg, form_data=data_source)

    try:
        # Construct single-row DataFrame with exact training column order
        input_df = pd.DataFrame([parsed_features])[FEATURE_COLUMNS]

        # Generate prediction via pipeline (handles scaling and regressor)
        prediction = model.predict(input_df)[0]
        # Bounded between 0 and 100
        prediction_clipped = float(np.clip(prediction, 0.0, 100.0))

        # Risk tier metadata
        risk_info = get_risk_tier(prediction_clipped)

        if is_json:
            return jsonify({
                'success': True,
                'health_impact_score': round(prediction_clipped, 2),
                'raw_score': round(float(prediction), 4),
                'risk_tier': risk_info['tier'],
                'risk_badge': risk_info['badge'],
                'risk_theme': risk_info['theme'],
                'risk_color': risk_info['color'],
                'risk_title': risk_info['title'],
                'risk_desc': risk_info['desc']
            })

        return render_template(
            'index.html',
            prediction_text=f"{prediction_clipped:.2f}",
            risk_tier=risk_info['tier'],
            risk_badge=risk_info['badge'],
            risk_theme=risk_info['theme'],
            risk_color=risk_info['color'],
            risk_title=risk_info['title'],
            risk_desc=risk_info['desc'],
            form_data=data_source
        )

    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        if is_json:
            return jsonify({'success': False, 'error': error_msg}), 500
        return render_template('index.html', error_text=error_msg, form_data=data_source)


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
