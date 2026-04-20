import os
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load Preprocessors
try:
    scaler = joblib.load('Robust-Scaler.bin')
    encoders = joblib.load('Label-Encoder.bin')
except Exception as e:
    print(f"Warning: Preprocessors not found. Ensure .bin files are present. Error: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # 1. Get Selected Model Category and Name
            model_selection = request.form['model_selection']
            model_type, model_name = model_selection.split('|')
            
            # 2. Gather form data (14 features)
            data = {
                'device_age_months': float(request.form['device_age_months']),
                'battery_capacity_mah': float(request.form['battery_capacity_mah']),
                'avg_screen_on_hours_per_day': float(request.form['avg_screen_on_hours_per_day']),
                'avg_charging_cycles_per_week': float(request.form['avg_charging_cycles_per_week']),
                'avg_battery_temp_celsius': float(request.form['avg_battery_temp_celsius']),
                'fast_charging_usage_percent': float(request.form['fast_charging_usage_percent']),
                'overnight_charging_freq_per_week': float(request.form['overnight_charging_freq_per_week']),
                'gaming_hours_per_week': float(request.form['gaming_hours_per_week']),
                'video_streaming_hours_per_week': float(request.form['video_streaming_hours_per_week']),
                'background_app_usage_level': request.form['background_app_usage_level'],
                'signal_strength_avg': request.form['signal_strength_avg'],
                'charging_habit_score': float(request.form['charging_habit_score']),
                'usage_intensity_score': float(request.form['usage_intensity_score']),
                'thermal_stress_index': float(request.form['thermal_stress_index'])
            }
            
            # 3. Create DataFrame
            df = pd.DataFrame([data])
            
            # 4. Encode categorical variables
            for col in ['background_app_usage_level', 'signal_strength_avg']:
                df[col] = encoders[col].transform(df[col])
                
            # 5. Scale features
            X_scaled = scaler.transform(df)
            
            # 6. Load Model and Predict
            if model_type == 'ML':
                # Load Machine Learning Model
                model = joblib.load(f"{model_name}.pkl")
                prediction = model.predict(X_scaled)[0]
            else:
                # Load Deep Learning Model
                model = load_model(f"{model_name}.keras")
                prediction = model.predict(X_scaled, verbose=0)[0][0]

            prediction_value = round(float(prediction), 2)
            
            # 7. Rule-Based Recommendation Engine
            if prediction_value >= 75:
                recommendation = "Keep Using"
                status_class = "status-good"
                icon = "🔋"
            elif prediction_value >= 50:
                recommendation = "Replace Battery"
                status_class = "status-warning"
                icon = "⚠️"
            else:
                recommendation = "Change Phone"
                status_class = "status-danger"
                icon = "📱"

            return render_template('predict.html', 
                                   prediction=prediction_value, 
                                   recommendation=recommendation,
                                   status_class=status_class,
                                   icon=icon,
                                   model_name=model_name.replace('_', ' '))
                                   
        except Exception as e:
            return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)