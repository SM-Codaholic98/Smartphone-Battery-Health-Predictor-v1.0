# ⚡ Smartphone Battery Health Predictor v1.0

**An Advanced Machine Learning & Deep Learning Pipeline + Web Application for Predictive Battery Degradation Analysis**
-----

## 📖 Project Overview

The **Smartphone Battery Health Predictor** is a comprehensive predictive modeling project designed to estimate the current battery health percentage of a smartphone based on user behavior, charging habits, and device stress factors.

This repository contains the complete lifecycle of the project: from Exploratory Data Analysis (EDA) and rigorous model training (using highly optimized Hybrid Deep Learning architectures) to a fully deployed, responsive **Flask Web Application**. Based on the predicted battery health, the app intelligently prescribes actionable recommendations (Keep Using, Replace Battery, or Change Phone).

-----

## 💻 Tech Stack

  * **Core Language:** Python 3.10+
  * **Data Processing & EDA:** Pandas, NumPy, Matplotlib, Seaborn
  * **Machine Learning:** Scikit-Learn, XGBoost, LightGBM, CatBoost
  * **Deep Learning:** TensorFlow / Keras
  * **Web Development (Backend):** Flask, Werkzeug
  * **Web Development (Frontend):** HTML5, CSS3 (Custom Glassmorphic & Responsive UI)

-----

## 📊 Dataset & Features

The model learns from a robust dataset of smartphone usage patterns. It utilizes the following **14 independent variables** to predict the `current_battery_health_percent`:

  * **Device Specs:** `device_age_months`, `battery_capacity_mah`
  * **Usage Patterns:** `avg_screen_on_hours_per_day`, `gaming_hours_per_week`, `video_streaming_hours_per_week`
  * **Charging Habits:** `avg_charging_cycles_per_week`, `fast_charging_usage_percent`, `overnight_charging_freq_per_week`, `charging_habit_score`
  * **Device Stress:** `avg_battery_temp_celsius`, `thermal_stress_index`, `usage_intensity_score`
  * **Categorical Factors:** `background_app_usage_level` (Low, Medium, High), `signal_strength_avg` (Poor, Moderate, Good)

-----

## ⚙️ Model Architecture & Methodology

### 1\. Data Preprocessing

  * **Encoding & Scaling:** Categorical features are transformed using `LabelEncoder`, and numerical stability is achieved via `RobustScaler` (saved as `.bin` artifacts for seamless web deployment).
  * **Splitting:** A strict 70/30 Train-Test split ensures accurate generalization testing.

### 2\. Machine Learning Pipeline

The project evaluates seven powerful regression algorithms, optimized with multi-threading to serve as rapid baselines:

  * Random Forest | AdaBoost | Gradient Boosting | XGBoost | XGBoost RF-Booster | LightGBM | CatBoost

### 3\. Optimized Deep Learning Pipeline

To capture complex, non-linear, and sequential relationships within the tabular data, the project implements a custom, highly optimized **3-block Hybrid Architecture**.

  * *Note: The architecture avoids redundant Dense layers (outputting directly to the single prediction unit from the final recurrent block) to prevent parameter bloat and maximize test-set generalization.*
  * **Base DNN:** A streamlined feed-forward network with a `256 -> 128 -> 64 -> 1` unit hierarchy, paired with Batch Normalization and Dropout.
  * **Hybrid DL Models:** Features are dynamically reshaped into 3D tensors `(Batch, Timesteps, Features)` for 1D Convolutions before passing through Recurrent layers:
      * **CNN-LSTM** & **CNN-BiLSTM**
      * **CNN-GRU** & **CNN-BiGRU**

-----

## 🌐 Web Application Features

The project includes a sleek, modern, and user-friendly Web Interface built with **Flask**.

  * **Dynamic Model Selection:** Users can seamlessly choose between any of the 7 ML models or 5 DL models from an intelligently grouped dropdown menu.
  * **Responsive UI:** Built with custom CSS Grid and Flexbox, ensuring a perfect layout across desktop monitors, tablets, and mobile devices.
  * **Rule-Based Recommendation Engine:** \* 🔋 **Keep Using:** Battery ≥ 75% (Status: Green)
      * ⚠️ **Replace Battery:** Battery 50–74% (Status: Orange/Warning)
      * 📱 **Change Phone:** Battery \< 50% (Status: Red/Danger)
  * **Real-time Processing:** The backend dynamically scales input features using the saved `Robust-Scaler.bin` and routes data to the selected `.pkl` or `.keras` model.

-----

## 📁 Repository Structure

```text
Smartphone-Battery-Health-Predictor/
│
├── app.py                            # Flask Backend Application
├── Label-Encoder.bin                 # Saved label encoder for inference
├── Robust-Scaler.bin                 # Saved scaler for inference
├── (Model Files)                     # ML (.pkl) and DL (.keras) saved models
│
├── Notebooks/
│   └── SBHP_EDA_ML_&_DL_Models.ipynb # Training, EDA, and Architecture definitions
│
├── static/
│   └── style.css                     # Custom styling for the Web App
│
└── templates/
    ├── index.html                    # Input form and model selection UI
    └── predict.html                  # Dynamic prediction results page
```

-----

## 🚀 How to Run Locally

**1. Clone the repository:**

```bash
git clone https://github.com/yourusername/Smartphone-Battery-Health-Predictor.git
cd Smartphone-Battery-Health-Predictor-v1.0
```

**2. Install dependencies:**

```bash
pip install flask pandas numpy scikit-learn xgboost lightgbm catboost tensorflow joblib
```

**3. Ensure Models are Present:**
*Run the Jupyter Notebook first if you haven't generated the `.pkl`, `.keras`, and `.bin` files yet, and ensure they are placed in the root directory.*

**4. Start the Flask Server:**

```bash
python app.py
```

**5. Access the Web App:**
Open your browser and navigate to `http://127.0.0.1:5000/`

-----

## 👨‍💻 About the Author

**Soham Maity** *AI/ML Engineer* Passionate about building highly optimized, clean, and scalable machine learning systems, advanced deep learning architectures, and full-stack AI web applications.
