# --- File: model/model_utils.py ---
# This file contains the functions for loading data and the model,
# and for preparing the data for prediction.
# -----------------------------------------------------------------

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

@st.cache_data
def load_data(data_path):
    """Loads the main dataframe from a parquet file."""
    #breakpoint()
    if not os.path.exists(data_path):
        st.error(f"Data file '{data_path}' not found. Please upload it to the project's root directory.")
        return None
    try:
        df = pd.read_parquet(data_path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_model(model_path):
    """Loads the trained model from a pickle file."""
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please upload it to the project's root directory.")
        return None
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_input_data(store_id, item_id, forecast_dates, df_complete):
    """
    Finds the correct rows from the pre-engineered dataframe and prepares them for prediction.
    """
    MODEL_FEATURES = [
        'unit_sales_rolling_mean_7', 'unit_sales_rolling_mean_14', 'z_score',
        'unit_sales_rolling_std_7', 'unit_sales_rolling_mean_30', 'unit_sales_lag_1',
        'is_weekend', 'unit_sales_rolling_std_14', 'unit_sales_rolling_std_30',
        'day_of_week', 'cluster_17', 'month', 'item_nbr', 'unit_sales_lag_14',
        'store_type_D', 'locale_National', 'year', 'class', 'unit_sales_lag_7',
        'transactions'
    ]

    input_dates = pd.to_datetime(forecast_dates)

    input_rows = df_complete[
        (df_complete['store_nbr'] == store_id) &
        (df_complete['item_nbr'] == item_id) &
        (df_complete['date'].isin(input_dates))
    ].copy()

    if input_rows.empty:
        st.warning("No data found for the selected store, item, and date combination.")
        return None

    for feature in MODEL_FEATURES:
        if feature not in input_rows.columns:
            input_rows[feature] = 0

    try:
        features = input_rows[MODEL_FEATURES]
    except KeyError as e:
        st.error(f"Error selecting features: {e}. There might be a mismatch between the data and the model's expected features.")
        return None

    return features

def predict(model, input_data):
    """Makes a prediction using the loaded model """
    if input_data is not None and not input_data.empty:
        prediction = model.predict(input_data)
        return prediction
    return np.array([0.0])


