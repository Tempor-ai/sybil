import streamlit as st
import pandas as pd
import json
import yaml
import requests
import time
import subprocess

# CSS for additional styling
# def local_css(file_name):
#     with open(file_name) as f:
#         st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
#
# # Load CSS
# local_css("style.css")

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

def prepare_dataset_forecast(df, time_col_index, target_col_index, test_period):
    time_col = df.columns[time_col_index]
    target_col = df.columns[target_col_index]
    df[time_col] = df[time_col].astype(str)
    df[target_col] = df[target_col].astype(float)
    train_data = df.iloc[:, [time_col_index, target_col_index]].values.tolist()
    return train_data[:-test_period], train_data[-test_period:]

server_process = start_uvicorn_server()

if server_process is None:
    st.warning("Using already running Uvicorn server.")
else:
    st.success("Started a new Uvicorn server.")

st.title("Sybil - Time Series Forecasting")
st.write("""
Upload your time series data in CSV format, select a forecasting model, 
specify the prediction period, and get your forecast results.
""")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataframe Preview:")
    st.dataframe(df, use_container_width=True)

    st.write("Line Chart:")
    st.line_chart(df.iloc[:, -1])

    st.write("Select Model:")
    model_option = st.selectbox("Choose a model option:",
                                ("Select default model", "Upload YAML file for custom model",
                                 "Upload JSON file for custom model"))

    if model_option == "Select default model":
        model_request = {
            "type": "meta_lr",
            "scorers": ["smape", "mape"],
            "params": {
                "preprocessors": [

                    {"type": "minmaxscaler"},
                ],
                "base_models": [
                    {"type": "darts_naive"},
                    {"type": "darts_seasonalnaive"},
                    {"type": "darts_autotheta"},
                    {"type": "darts_autoets"},
                    {"type": "darts_autoarima"},
                ],
            },
        }
    elif model_option == "Upload YAML file for custom model":
        yaml_file = st.file_uploader("Upload YAML file", type="yaml")
        if yaml_file is not None:
            model_request = yaml.safe_load(yaml_file)
    elif model_option == "Upload JSON file for custom model":
        json_file = st.file_uploader("Upload JSON file", type="json")
        if json_file is not None:
            model_request = json.load(json_file)

    test_period = st.number_input("Select number of days to predict", min_value=1, max_value=365, value=7, step=1)

    if st.button("Run Forecast"):
        with st.spinner("Processing..."):
            train_data, test_data = prepare_dataset_forecast(df, 0, -1, test_period)
            api_json = {
                "data": train_data,
                "model": model_request,
            }

            try:
                with open("url.yaml", "r") as file:
                    url_dict = yaml.safe_load(file)

                protocol = url_dict["protocol"]
                host = url_dict["host"]
                port = url_dict["port"]
                train_endpoint = "train"
                forecast_endpoint = "forecast"
                train_url = f"{protocol}://{host}:{port}/{train_endpoint}"

                start_time = time.time()
                response = requests.post(train_url, json=api_json)
                response.raise_for_status()
                exc_time = time.time() - start_time
                response_data = response.json()

                st.write("Response Data:")
                st.write(response_data)

                if 'model' in response_data:
                    model = response_data['model']
                    dates = [data[0] for data in test_data]
                    forecast_api_json = {
                        'model': model,
                        'data': dates
                    }
                    forecast_url = f"{protocol}://{host}:{port}/{forecast_endpoint}"

                    forecast_response = requests.post(forecast_url, json=forecast_api_json)
                    forecast_response.raise_for_status()
                    forecast_data = forecast_response.json()

                    if 'data' in forecast_data:
                        forecast_dates = [item[0] for item in forecast_data['data']]
                        forecast_values = [item[1] for item in forecast_data['data']]

                        st.write("Forecast Results:")
                        result_df = pd.DataFrame({
                            "Date": forecast_dates,
                            "Forecast": forecast_values,
                            "Actual": [data[1] for data in test_data]
                        })
                        st.line_chart(result_df.set_index("Date"))

                        csv = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download Forecast Data as CSV",
                            data=csv,
                            file_name='forecast_results.csv',
                            mime='text/csv',
                        )
                    else:
                        st.error("Forecast response does not contain 'data' key. Please check the server response.")
                else:
                    st.error("Train response does not contain 'model' key. Please check the server response.")

            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred: {e}")
                if e.response:
                    st.error(f"Server response: {e.response.text}")
                st.error("Please ensure the Sybil AWS service is running and the URL is correct.")

stop_uvicorn_server(server_process)
