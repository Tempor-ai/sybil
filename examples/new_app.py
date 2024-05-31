import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
import yaml
import json
import subprocess
import time
import plotly.graph_objects as go
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
    train_size = int(len(df) * 0.8)  # Calculate 80% of the dataset length
    train_df = df.iloc[:train_size].copy()  # Take the first 80% of the data as training data
    test_df = df.iloc[train_size:].copy()  # Take the remaining 20% of the data as testing data
    return train_df, test_df

# Function to prepare dataset for forecasting
def prepare_dataset_forecast(df, time_col, target_col):
    df[time_col] = pd.to_datetime(df[time_col])
    df[target_col] = df[target_col].astype(float)
    data = df[[time_col, target_col]].copy()
    data[time_col] = data[time_col].astype(str)
    return data.values.tolist()

# Start Uvicorn server
server_process = start_uvicorn_server()

# Load URL configuration
with open('url.yaml', 'r') as file:
    url_dict = yaml.safe_load(file)

protocol = url_dict['protocol']
host = url_dict['host']
port = url_dict['port']

# Streamlit UI
st.title("SYBIL General-Purpose Forecaster")

# Initialize session state for model
if 'model' not in st.session_state:
    st.session_state.model = None

# Initialize session state for dataset and time_col
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'time_col' not in st.session_state:
    st.session_state.time_col = None

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    st.session_state.dataset = pd.read_csv(uploaded_file)
    dataset = st.session_state.dataset
    st.write("### Dataset Preview")
    st.dataframe(dataset)  # Display entire dataset with option to scroll

    st.session_state.time_col = dataset.columns[0]  # Default to first column for time
    target_col = dataset.columns[1]  # Default to second column for target

    # static plot
    # st.write("### Dataset Line Plot")
    # fig, ax = plt.subplots()
    # ax.plot(pd.to_datetime(dataset[st.session_state.time_col]), dataset[target_col], label='Data', marker='o')
    # ax.set_xlabel("Date")
    # ax.set_ylabel("Value")
    # ax.legend()
    # st.pyplot(fig)

    st.session_state.time_col = dataset.columns[0]
    st.session_state.target_col = dataset.columns[1]
    time_col = st.session_state.time_col
    target_col = st.session_state.target_col

    # Dynamic plot
    st.write("### Dynamic Plot")

    # Ensure the time_col and target_col exist in the dataset
    if time_col not in dataset.columns or target_col not in dataset.columns:
        st.error(f"Columns {time_col} or {target_col} not found in the dataset")
    else:
        # Creating a Plotly figure
        fig = go.Figure()

        # Adding the time series data to the plot
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(dataset[time_col]),
            y=dataset[target_col],
            mode='lines+markers',
            name='Data',
            marker=dict(size=5),
            line=dict(width=2)
        ))

        # Enhancements
        fig.update_layout(
            title='Time Series visualisation',
            xaxis_title='Date',
            yaxis_title='Value',
            template='plotly_white',
            xaxis=dict(
                showgrid=True,
                gridcolor='lightgrey',
                tickformat='%b %Y',
                rangeslider=dict(visible=True)
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='lightgrey'
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)
    model_option = st.radio("Select Model Option", ("Default Model", "Upload YAML", "Upload JSON"))

    model_request = None
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

    if st.button("Train Model"):
        if model_request is None:
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
                    ],
                },
            }

        # Train model with 80-20 split
        train_df, test_df = split_dataset(st.session_state.dataset, st.session_state.time_col, target_col)
        train_data = prepare_dataset_forecast(train_df, st.session_state.time_col, target_col)

        with st.spinner("Training model..."):
            api_json = {
                'data': train_data,
                'model': model_request
            }

            endpoint = 'train'
            url = f'{protocol}://{host}:{port}/{endpoint}'

            response = requests.post(url, json=api_json)
            train_json_out = response.json()

        st.success("Model trained successfully!")
        st.write("### Model Response")
        st.json(train_json_out)

        # Save model to session state
        st.session_state.model = train_json_out['model']

    st.write("### Future Forecast")
    future_file = st.file_uploader("Upload future data CSV", type="csv")
    if future_file is not None:
        future_dataset = pd.read_csv(future_file)
        st.write("### Future Dataset Preview")
        st.dataframe(future_dataset)

        future_time_col = future_dataset.columns[0]
        future_target_col = future_dataset.columns[1] if len(future_dataset.columns) > 1 else None

        # Retrain model on the entire dataset before making future forecasts
        if st.button("Forecast Future Data"):
            if st.session_state.model is not None:
                # Retrain model with 100% of the data
                train_data = prepare_dataset_forecast(st.session_state.dataset, st.session_state.time_col, target_col)

                with st.spinner("Prediction in progress..."):
                    api_json = {
                        'data': train_data,
                        'model': model_request
                    }

                    endpoint = 'train'
                    url = f'{protocol}://{host}:{port}/{endpoint}'

                    response = requests.post(url, json=api_json)
                    train_json_out = response.json()

                # st.success("Model retrained successfully!")
                # st.write("### Retrained Model Response")
                # st.json(train_json_out)

                # Update model in session state
                st.session_state.model = train_json_out['model']

                future_test_data = future_dataset[[future_time_col]].copy()
                future_test_data[future_time_col] = pd.to_datetime(future_test_data[future_time_col]).astype(str)

                future_test_data_list = future_test_data.values.tolist()

                with st.spinner("Forecasting future data..."):
                    api_json = {
                        'model': st.session_state.model,
                        'data': future_test_data_list
                    }

                    endpoint = 'forecast'
                    url = f'{protocol}://{host}:{port}/{endpoint}'

                    response = requests.post(url, json=api_json)
                    future_forecast_json_out = response.json()

                future_forecast_df = pd.DataFrame(
                    data=future_forecast_json_out['data'],
                    columns=[future_time_col, 'Forecast'],
                )

                st.write("### Future Forecast Results")
                future_forecast_df[future_time_col] = pd.to_datetime(future_forecast_df[future_time_col])
                st.dataframe(future_forecast_df)

                # Plot training data and future forecast
                fig, ax = plt.subplots()
                ax.plot(pd.to_datetime(st.session_state.dataset[st.session_state.time_col]), st.session_state.dataset[target_col], label='Training Data', marker='o')
                future_forecast_df.set_index(future_time_col)['Forecast'].plot(ax=ax, color='green', label='Future Forecast')
                ax.axvline(x=pd.to_datetime(st.session_state.dataset[st.session_state.time_col].iloc[-1]), color='black', linestyle='--', label='End of Training')
                ax.set_xlabel("Date")
                ax.set_ylabel("Value")
                ax.legend()
                st.pyplot(fig)
                #
                # # Assuming necessary session state variables are already set
                # if 'dataset' not in st.session_state:
                #     st.session_state.dataset = None
                # if 'time_col' not in st.session_state:
                #     st.session_state.time_col = None
                # if 'target_col' not in st.session_state:
                #     st.session_state.target_col = None
                #
                # # Example future_forecast_df for demonstration
                # # You would typically have this from your forecasting model
                # # future_forecast_df = pd.DataFrame({
                # #     'Date': pd.date_range(start='2023-01-02', periods=30, freq='D'),
                # #     'Forecast': [x + (x * 0.1) for x in range(30)]
                # # })
                #
                # # Assuming future_time_col is the column in the forecast DataFrame that represents time
                # future_time_col = 'Date'
                #
                # # Streamlit interface for displaying future forecast results
                # st.write("### Future Forecast Results")
                # future_forecast_df[future_time_col] = pd.to_datetime(future_forecast_df[future_time_col])
                # st.dataframe(future_forecast_df)
                #
                # # Ensure the time_col and target_col exist in the dataset
                # time_col = st.session_state.time_col
                # target_col = st.session_state.target_col
                #
                # if time_col not in st.session_state.dataset.columns or target_col not in st.session_state.dataset.columns:
                #     st.error(f"Columns {time_col} or {target_col} not found in the dataset")
                # else:
                #     # Creating a Plotly figure
                #     fig = go.Figure()
                #
                #     # Adding training data
                #     fig.add_trace(go.Scatter(
                #         x=pd.to_datetime(st.session_state.dataset[time_col]),
                #         y=st.session_state.dataset[target_col],
                #         mode='lines+markers',
                #         name='Training Data',
                #         marker=dict(size=5),
                #         line=dict(width=2)
                #     ))
                #
                #     # Adding future forecast data
                #     fig.add_trace(go.Scatter(
                #         x=pd.to_datetime(future_forecast_df[future_time_col]),
                #         y=future_forecast_df['Forecast'],
                #         mode='lines+markers',
                #         name='Future Forecast',
                #         marker=dict(size=5, color='green'),
                #         line=dict(width=2, color='green')
                #     ))
                #
                #     # Adding a vertical line for the end of training data
                #     end_of_training_date = pd.to_datetime(st.session_state.dataset[time_col].iloc[-1])
                #     fig.add_shape(
                #         dict(
                #             type="line",
                #             x0=end_of_training_date,
                #             y0=0,
                #             x1=end_of_training_date,
                #             y1=max(future_forecast_df['Forecast'].max(), st.session_state.dataset[target_col].max()),
                #             line=dict(color="black", dash="dash"),
                #         )
                #     )
                #
                #     fig.update_layout(
                #         title='Training Data and Future Forecast',
                #         xaxis_title='Date',
                #         yaxis_title='Value',
                #         template='plotly_white',
                #         xaxis=dict(
                #             showgrid=True,
                #             gridcolor='lightgrey',
                #             tickformat='%b %Y',
                #             rangeslider=dict(visible=True)
                #         ),
                #         yaxis=dict(
                #             showgrid=True,
                #             gridcolor='lightgrey'
                #         ),
                #         legend=dict(
                #             yanchor="top",
                #             y=0.99,
                #             xanchor="left",
                #             x=0.01
                #         )
                #     )
                #
                #     # Display the plot in Streamlit
                #     st.plotly_chart(fig)

                # If the future dataset contains actual values, plot them
                if future_target_col is not None:
                    st.write("### Future Forecast with Actual Values")
                    future_dataset[future_time_col] = pd.to_datetime(future_dataset[future_time_col])
                    fig, ax = plt.subplots()
                    ax.plot(pd.to_datetime(st.session_state.dataset[st.session_state.time_col]), st.session_state.dataset[target_col], label='Training Data', marker='o')
                    ax.plot(future_dataset.set_index(future_time_col).index, future_dataset[future_target_col], label='Actual Values', marker='o', color='blue')
                    future_forecast_df.set_index(future_time_col)['Forecast'].plot(ax=ax, color='green', label='Future Forecast')
                    ax.axvline(x=pd.to_datetime(st.session_state.dataset[st.session_state.time_col].iloc[-1]), color='black', linestyle='--', label='End of Training')
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Value")
                    ax.legend()
                    st.pyplot(fig)
                    st.dataframe(future_dataset)

            else:
                st.warning("Please train the model first.")

# Stop Uvicorn server when Streamlit app is closed
if st.button("Stop Uvicorn Server"):
    stop_uvicorn_server(server_process)
