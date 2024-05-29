import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
import yaml
import json
import subprocess
import time
from datetime import timedelta

# Function to start Uvicorn server
def start_uvicorn_server():
    try:
        response = requests.get("http://localhost:8000")
        if response.status_code == 200:
            st.write("Uvicorn server is already running.")
            return None
    except requests.exceptions.RequestException:
        pass

    st.write("Starting Uvicorn server...")
    process = subprocess.Popen(["uvicorn", "appserver:app", "--reload"])

    for _ in range(10):
        try:
            response = requests.get("http://localhost:8000")
            if response.status_code == 200:
                st.write("Uvicorn server started successfully.")
                return process
        except requests.exceptions.RequestException:
            time.sleep(1)

    st.write("Failed to start Uvicorn server.")
    process.terminate()
    return None

# Function to stop Uvicorn server
def stop_uvicorn_server(process):
    if process and isinstance(process, subprocess.Popen):
        process.terminate()
        st.write("Uvicorn server stopped.")

# Function to split dataset into train and test sets
def split_dataset(df, time_col, target_col):
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    return train_df, test_df

# Function to prepare dataset for forecasting
def prepare_dataset_forecast(df, time_col, target_col):
    df[time_col] = pd.to_datetime(df[time_col])
    df[target_col] = df[target_col].astype(float)
    data = df[[time_col, target_col]].copy()
    data[time_col] = data[time_col].astype(str)
    return data.values.tolist()

# Function to generate next 10 dates
def generate_next_dates(last_date, periods):
    next_dates = [last_date + timedelta(days=i) for i in range(1, periods + 1)]
    return next_dates

# Start Uvicorn server
server_process = start_uvicorn_server()

# Streamlit UI
st.title("Climate Data Forecasting")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)
    st.write("Dataset Shape:", dataset.shape)
    st.write("Dataset Columns:", dataset.columns)
    st.write("Dataset Head:", dataset.head())
    st.write("Dataset Tail:", dataset.tail())

    time_col_index = st.selectbox("Select Time Column Index", range(len(dataset.columns)), format_func=lambda x: dataset.columns[x])
    target_col_index = st.selectbox("Select Target Column Index", range(len(dataset.columns)), format_func=lambda x: dataset.columns[x])

    time_col = dataset.columns[time_col_index]
    target_col = dataset.columns[target_col_index]

    train_df, test_df = split_dataset(dataset, time_col, target_col)
    train_data = prepare_dataset_forecast(train_df, time_col, target_col)
    test_data = prepare_dataset_forecast(test_df, time_col, target_col)

    model_option = st.radio("Select Model Option", ("Default Model", "Upload YAML", "Upload JSON"))

    if model_option == "Upload YAML":
        uploaded_yaml = st.file_uploader("Upload YAML file for model request", type="yaml")
        if uploaded_yaml is not None:
            model_request = yaml.safe_load(uploaded_yaml)
        else:
            st.warning("Please upload a YAML file.")
            st.stop()
    elif model_option == "Upload JSON":
        uploaded_json = st.file_uploader("Upload JSON file for model request", type="json")
        if uploaded_json is not None:
            model_request = json.load(uploaded_json)
        else:
            st.warning("Please upload a JSON file.")
            st.stop()
    else:
        model_request = {
            'type': 'meta_lr',
            'scorers': ['smape', 'mape'],
            'params': {
                'preprocessors': [
                    {'type': 'dartsimputer'},
                    {'type': 'minmaxscaler'},
                ],
                'base_models': [
                    {'type': 'darts_naive'},
                    {'type': 'darts_seasonalnaive'},
                ],
            },
        }

    with st.spinner("Training model..."):
        api_json = {
            'data': train_data,
            'model': model_request
        }

        with open('url.yaml', 'r') as file:
            url_dict = yaml.safe_load(file)

        protocol = url_dict['protocol']
        host = url_dict['host']
        port = url_dict['port']
        endpoint = 'train'

        url = f'{protocol}://{host}:{port}/{endpoint}'

        response = requests.post(url, json=api_json)
        train_json_out = response.json()

    st.success("Model trained successfully!")

    model = train_json_out['model']

    with st.spinner("Forecasting..."):
        api_json = {
            'model': model,
            'data': test_data
        }

        endpoint = 'forecast'
        url = f'{protocol}://{host}:{port}/{endpoint}'

        response = requests.post(url, json=api_json)
        forecast_json_out = response.json()

    forecast_df = pd.DataFrame(
        data=forecast_json_out['data'],
        columns=[time_col, target_col],
    )

    st.write("Forecast DataFrame Shape:", forecast_df.shape)
    st.write("Forecast DataFrame Columns:", forecast_df.columns)
    st.write("Forecast DataFrame Head:", forecast_df.head())
    st.write("Forecast DataFrame Tail:", forecast_df.tail())

    # Convert time column back to datetime
    forecast_df[time_col] = pd.to_datetime(forecast_df[time_col])

    # Plot the results
    figsize = (16, 8)
    fig, ax = plt.subplots(figsize=figsize)
    train_df.set_index(time_col)[target_col].plot(ax=ax, color='blue', label='Train')
    test_df.set_index(time_col)[target_col].plot(ax=ax, color='orange', label='Test')
    forecast_df.set_index(time_col)[target_col].plot(ax=ax, color='red', label='Forecast')
    ax.legend()
    st.pyplot(fig)

    combined_df = pd.concat([dataset, forecast_df]).reset_index(drop=True)
    st.write("Combined DataFrame Shape:", combined_df.shape)
    st.write("Combined DataFrame Head:", combined_df.head())
    st.write("Combined DataFrame Tail:", combined_df.tail())

    # Provide download option for detailed forecast CSV
    csv = combined_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Detailed Forecast CSV",
        data=csv,
        file_name='detailed_forecast.csv',
        mime='text/csv',
    )

    # Generate next 10 dates for future prediction
    last_date = pd.to_datetime(dataset[time_col].iloc[-1])
    next_dates = generate_next_dates(last_date, 10)
    future_df = pd.DataFrame(next_dates, columns=[time_col])
    st.write("Future Dates for Prediction:")
    st.write(future_df)

    # Provide download option for future dates CSV
    future_csv = future_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Future Dates CSV",
        data=future_csv,
        file_name='future_dates.csv',
        mime='text/csv',
    )

    # Option to upload a new CSV for future prediction
    future_file = st.file_uploader("Upload a new CSV for future prediction", type="csv")
    if future_file is not None:
        future_dataset = pd.read_csv(future_file)
        st.write("Future Dataset Shape:", future_dataset.shape)
        st.write("Future Dataset Columns:", future_dataset.columns)
        st.write("Future Dataset Head:", future_dataset.head())
        st.write("Future Dataset Tail:", future_dataset.tail())

        future_time_col_index = st.selectbox("Select Future Time Column Index", range(len(future_dataset.columns)), format_func=lambda x: future_dataset.columns[x])

        future_test_data = future_dataset.iloc[:, [future_time_col_index]].copy()
        future_test_data[future_test_data.columns[0]] = pd.to_datetime(future_test_data[future_test_data.columns[0]]).astype(str).tolist()

        with st.spinner("Forecasting future data..."):
            api_json = {
                'model': model,
                'data': future_test_data.values.tolist()
            }

            endpoint = 'forecast'
            url = f'{protocol}://{host}:{port}/{endpoint}'

            response = requests.post(url, json=api_json)
            future_forecast_json_out = response.json()

        future_forecast_df = pd.DataFrame(
            data=future_forecast_json_out['data'],
            columns=[future_dataset.columns[future_time_col_index], target_col],
        )

        st.write("Future Forecast DataFrame Shape:", future_forecast_df.shape)
        st.write("Future Forecast DataFrame Columns:", future_forecast_df.columns)
        st.write("Future Forecast DataFrame Head:", future_forecast_df.head())
        st.write("Future Forecast DataFrame Tail:", future_forecast_df.tail())

        # Convert future time column back to datetime
        future_forecast_df[future_dataset.columns[future_time_col_index]] = pd.to_datetime(future_forecast_df[future_dataset.columns[future_time_col_index]])

        # Plot the future forecast
        fig, ax = plt.subplots(figsize=figsize)
        future_forecast_df.set_index(future_dataset.columns[future_time_col_index])[target_col].plot(ax=ax, color='green', label='Future Forecast')
        ax.legend()
        st.pyplot(fig)

# Stop Uvicorn server when Streamlit app is closed
if st.button("Stop Uvicorn Server"):
    stop_uvicorn_server(server_process)
