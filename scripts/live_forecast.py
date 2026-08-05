import joblib, requests, os
import pandas as pd
import numpy as np
from scripts.pipeline import data_engineering, classification_engineering
from huggingface_hub import hf_hub_download
import streamlit as st
hf_repo_id = "Adarsh-0704/green-h2-forecast"

@st.cache_resource
def load_hf_model(file):
    model = hf_hub_download(repo_id=hf_repo_id, filename=file)
    return joblib.load(model)

def forecast(lat=24.11, lon=69.35):
    # Open Meteo API endpoint
    url = 'https://api.open-meteo.com/v1/forecast'
    parameters = {
        'latitude' : lat,
        'longitude' : lon,
        'hourly' : 'windspeed_100m,shortwave_radiation',
        'timezone' : 'auto',
        'forecast_days' : 7
    }

    response = requests.get(url, params=parameters, timeout=(5, 10))
    response.raise_for_status()

    data = response.json()['hourly']

    df = pd.DataFrame({'Timestamp' : pd.to_datetime(data['time']),
                       'GHI(W/m2)' : data['shortwave_radiation'],
                       'Windspeed(m/s)' : data['windspeed_100m']
                      })
    # km/h to m/s
    df['Windspeed(m/s)'] = df['Windspeed(m/s)'] / 3.6

    # Loading our trained models
    spline = load_hf_model('spline_classification.joblib')
    scaler = load_hf_model('classi_scaler.joblib')
    best_rf = load_hf_model('random_forest_regressor.joblib')
    best_svm = load_hf_model('svm_shutdown_classification.joblib')
    chosen_threshold = load_hf_model('svm_threshold.joblib')

    # Part 1 for Random Forest
    df_engineered, _ = data_engineering(df.copy(), fit_spline=spline)

    rf_features = ['GHI(W/m2)', 'Windspeed(m/s)', 'Stored Energy(MWh)', 'Mon',
                    'Day', 'spline_hr_1','spline_hr_2', 'spline_hr_3',
                    'spline_hr_4', 'Windspeed_mean_3h','Windspeed_std_3h',
                    'GHI_mean_3h', 'Windspeed_lag_1hr','GHI_lag_1hr'
                    ]
    X_test = df_engineered[rf_features]
    rf_yield = best_rf.predict(X_test)

    # Part 2 for SVM
    X_scaled, y, _, _ = classification_engineering(df.copy(), scaler=scaler, spline=spline)
    decision_scores = best_svm.decision_function(X_scaled)

    # Using our threshold where 1 = Shutdown, 0 = Operating
    df_engineered['Predicted_Shutdown'] = (decision_scores >= chosen_threshold).astype(int)
    # Whenever Electrolyzer is predicted shutdown drop yield to 0 kg
    df_engineered['Predicted_Hydrogen_yield(kg)'] = np.where(df_engineered['Predicted_Shutdown'] == 1, 0, rf_yield)

    # Extracting date to dynamically refresh every midnight
    today = pd.Timestamp.now().date()
    df_engineered = df_engineered[df_engineered.index.date >= today].copy()
    # Labeling days with (0, 1, .. ,6)
    df_engineered['Relative Day'] = (df_engineered.index - pd.Timestamp(today)).days

    return df_engineered