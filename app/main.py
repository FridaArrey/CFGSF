# --- File: app/main.py ---
# Main application script that runs the Streamlit web app.
# Imports the necessary functions from model/model_utils.py.
# --------------------------------------------------------------------
print("Creating 'app/main.py'...")

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import sys

# Add the parent directory to the Python path to allow importing from 'model'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model_utils import load_data, load_model, preprocess_input_data, predict

# --- Main App Function ---
def main():
    st.title("Corporación Favorita Sales Forecasting")
    st.markdown("---")

    # Load data and model
    st.info("Loading model and data...")
    data_filename = 'df_complete_one.parquet'
    model_filename = 'final_optimized_xgboost_model.pkl'
    breakpoint()
    # Place these files in the project's root directory.
    df_complete = load_data(data_filename)
    model = load_model(model_filename)
    #breakpoint()
    # Stop the app if files are missing
    if df_complete is None or model is None:
        return

    st.success("Files loaded successfully!")

    # --- UI components for inputs ---
    st.subheader("Select Store and Item")
    st.markdown("---")
    
    unique_stores = sorted(df_complete['store_nbr'].unique().tolist())
    unique_items = sorted(df_complete['item_nbr'].unique().tolist())
    
    default_store = 44
    default_item = 1047679
    default_store_index = unique_stores.index(default_store) if default_store in unique_stores else 0
    default_item_index = unique_items.index(default_item) if default_item in unique_items else 0

    store_id = st.selectbox("Select a Store", unique_stores, index=default_store_index)
    item_id = st.selectbox("Select an Item", unique_items, index=default_item_index)
    
    st.write(f"**Selected Store:** {store_id}")
    st.write(f"**Selected Item:** {item_id}")

    min_date = df_complete['date'].min().date()
    max_date = df_complete['date'].max().date()

    forecast_date = st.date_input("Forecast Date", value=max_date, min_value=min_date, max_value=max_date)

    st.markdown("---")
    st.subheader("Forecast mode")
    forecast_mode = st.radio(" ", ("Single day", "Next N days"), key="forecast_mode")
    
    n_days = 1
    if forecast_mode == "Next N days":
        n_days = st.slider("N days", 1, 30, 7)
    
    st.markdown("---")
    
    if st.button("Get Forecast"):
        with st.spinner("Generating forecast..."):
            
            if forecast_mode == "Single day":
                forecast_dates = [forecast_date]
            else:
                forecast_dates = [forecast_date + datetime.timedelta(days=i) for i in range(1, n_days + 1)]
                
            input_data = preprocess_input_data(store_id, item_id, forecast_dates, df_complete)
            
            if input_data is not None:
                prediction = predict(model, input_data)
                
                forecast_df = pd.DataFrame({
                    'date': forecast_dates,
                    'prediction': prediction
                })

                historical_data = df_complete[
                    (df_complete['store_nbr'] == store_id) &
                    (df_complete['item_nbr'] == item_id) &
                    (df_complete['date'] < pd.to_datetime(forecast_df['date'].min()))
                ][['date', 'unit_sales']].copy()

                plot_data = pd.DataFrame(columns=['date', 'Actual (history)', 'Forecast'])
                plot_data['date'] = pd.to_datetime(historical_data['date'].tolist() + forecast_df['date'].tolist())
                plot_data['Actual (history)'] = historical_data['unit_sales'].tolist() + [np.nan] * len(forecast_df)
                plot_data['Forecast'] = [np.nan] * len(historical_data) + forecast_df['prediction'].tolist()
                
                plot_data_start_date = plot_data['date'].max() - pd.Timedelta(days=180)
                plot_data_filtered = plot_data[plot_data['date'] >= plot_data_start_date]

                st.subheader("Forecast Results")
                if forecast_mode == "Single day":
                     st.write(f"Predicted sales for {forecast_dates[0]}: **{prediction[0]:.2f}**")
                else:
                    st.write(f"Predicted {n_days} days: {forecast_dates[0]} → {forecast_dates[-1]}.")

                st.line_chart(plot_data_filtered.set_index('date'))

                st.subheader("Forecast Table")
                st.dataframe(forecast_df, use_container_width=True)

                csv = forecast_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download forecast as CSV",
                    data=csv,
                    file_name=f'forecast_{store_id}_{item_id}_{forecast_date}_{n_days}days.csv',
                    mime='text/csv'
                )
            else:
                st.error("Could not generate a forecast. The selected combination of store, item, and date does not exist in the dataset.")

if __name__ == "__main__":
    main()

print("\n# --- End of app/main.py ---")
