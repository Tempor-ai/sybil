import streamlit as st
import json
import yaml
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import time

sns.set()


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


def stop_uvicorn_server(process):
    if process and isinstance(process, subprocess.Popen):
        process.terminate()
        st.write("Uvicorn server stopped.")


# Start Uvicorn server at the beginning
uvicorn_process = start_uvicorn_server()


# Function to load dataset
def load_dataset(file_path):
    dataset = pd.read_csv(file_path)
    return dataset


# Function to split dataset
def split_dataset(dataset, train_size=0.8):
    train_points = int(train_size * len(dataset))
    train_df = dataset.iloc[:train_points]
    test_df = dataset.iloc[train_points:]
    return train_df, test_df


# Function to prepare data for API
def prepare_data(df):
    data = []
    for value in df.values:
        data.append(list(value))
    return data


# Function to plot data
def plot_data(df, time_col, target_col, title, figsize=(16, 8), color='b'):
    plt.figure(figsize=figsize)
    df.set_index(time_col)[target_col].plot(color=color)
    plt.title(title, fontweight='bold', fontsize=20)
    plt.ylabel(f'Temperature Anomaly ({u"N{DEGREE SIGN}" + "C"})')
    st.pyplot(plt)


# Function to make API request
def make_api_request(url, data):
    response = requests.post(url, json=data)
    return response.json()


# Streamlit App
st.title("Climate Data Forecasting")

# Step 1: Upload Dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")
if uploaded_file:
    dataset = load_dataset(uploaded_file)
    st.write("Dataset Loaded Successfully!")
    st.write(dataset.head())

    # Step 2: Show Training Plot
    time_col = dataset.columns[0]
    target_col = dataset.columns[-1]
    dataset[time_col] = dataset[time_col].astype(str)

    train_df, test_df = split_dataset(dataset)
    plot_data(train_df, time_col, target_col, "Training Data Plot")

    # Step 3: Select Model Type
    st.write("Select Model Option:")
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
            'type': 'meta_lr',  # 'meta_wa'
            'scorers': ['smape', 'mape'],
            'params': {
                'preprocessors': [
                    {'type': 'dartsimputer', 'params': {'strategy': 'mean'}},
                    {'type': 'minmaxscaler'},
                ],
                'base_models': [
                    {'type': 'darts_naive'},
                    {'type': 'darts_seasonalnaive'},
                    {'type': 'darts_autotheta'},
                    {'type': 'darts_autoets'},
                    {'type': 'darts_autoarima'},
                    {'type': 'darts_tbats'},
                    {'type': 'darts_lightgbm',
                     'params': {
                         'lags': 12,
                         'lags_future_covariates': [0, 1, 2],
                         'output_chunk_length': 6,
                         'verbose': -1
                     }
                     },
                    {'type': 'darts_rnn',
                     'params': {
                         'model': 'LSTM',
                         'hidden_dim': 10,
                         'n_rnn_layers': 3
                     }
                     },
                ],
            },
        }

    train_data = prepare_data(train_df)
    api_json = {'data': train_data, 'model': model_request}

    with open('url.yaml', 'r') as file:
        url_dict = yaml.safe_load(file)

    protocol = url_dict['protocol']
    host = url_dict['host']
    port = url_dict['port']
    endpoint = 'train'
    url = f'{protocol}://{host}:{port}/{endpoint}'

    # Step 4: Train Model
    if st.button("Train Model"):
        st.write("Training the model...")
        train_json_out = make_api_request(url, api_json)
        st.write("Model Trained Successfully!")

        model = train_json_out['model']

        # Step 5: Forecast
        test_data = prepare_data(test_df.drop(columns=target_col))
        api_json = {'model': model, 'data': test_data}
        endpoint = 'forecast'
        url = f'{protocol}://{host}:{port}/{endpoint}'

        if st.button("Forecast"):
            st.write("Forecasting...")
            forecast_json_out = make_api_request(url, api_json)
            st.write("Forecast Completed!")

            forecast_df = pd.DataFrame(
                data=forecast_json_out['data'],
                columns=[time_col, target_col],
            )

            train_df['color'] = 'b'
            forecast_df['color'] = 'r'
            df = pd.concat([train_df, forecast_df]).reset_index(drop=True)

            plot_data(df, time_col, target_col, "Training and Forecast Data")

            plot_data(dataset, time_col, target_col, "Full Dataset with Forecast")

# Stop Uvicorn server when the app closes
if uvicorn_process:
    stop_uvicorn_server(uvicorn_process)
