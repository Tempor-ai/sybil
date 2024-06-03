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
st.image("raphael_sibyls.jpg", caption="\"Sybils\" fresco painting by Renaissance artist Raphael (Source: Wikipedia)", use_column_width=True)
st.title(":violet[SYBIL General-Purpose Forecaster]")

st.write("### :blue[Overview]")
st.write("Welcome to **SYBIL**, the *general-purpose* and *domain-agnostic* forecaster! This is the simple UI webapp to demostrate how use the SYBIL service via HTTP API. You can also find SYBIL deployed on the SingularityNET (SNET) marketplace [here](https://beta.singularitynet.io/servicedetails/org/temporai/service/sybil).")
st.write("Below is the image of the SYBIL API Service schema. You can find more details about SYBIL's architecture in our comprehensive design report [here](https://bit.ly/sybil-design-report).")
st.image("api_document.png", caption="SYBIL API Service schema visualization (Source: Temporai)", use_column_width=True)
st.write("As mentioned in the design report, SYBIL contains one service with two API functions: **Train** and **Forecast**. Here are their descriptions:")
st.write("- **Train Function:** user inputs time-series train data (.csv) and optional custom model parameters (.yaml or .json), function outputs fitted model object in serialized format with evaluation metrics.")
st.write("- **Forecast Function:** users inputs forecasted datetimes and optional actual values (for evaluation purposes) with the fitted model object in serialized format, functions outputs those datetimes with the fitted model's forecasted values.")
st.write("### :blue[Train Function]")
st.write("Welcome to the **Train Function**! Here, you can effortlessly train your model on historical data with just a single click. Let's get your model up and running!")

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
    st.write("##### Dataset Preview")
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
    st.write("##### Dynamic Plot")

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
            line=dict(color='black')
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
        # if model_request is None:
        #     model_request = {
        #         "type": "meta_lr",
        #         "scorers": ["smape", "mape"],
        #         "params": {
        #             "preprocessors": [
        #                 {"type": "minmaxscaler"},
        #             ],
        #             "base_models": [
        #                 {"type": "darts_naive"},
        #                 {"type": "darts_seasonalnaive"},
        #             ],
        #         },
        #     }

        # Train model with 80-20 split
        train_df, test_df = split_dataset(st.session_state.dataset, st.session_state.time_col, target_col)
        train_data = prepare_dataset_forecast(train_df, st.session_state.time_col, target_col)
        # st.write(model_request)
        with st.spinner("Training model..."):
            api_json = {
                'data': train_data,
                'model': model_request
            }

            endpoint = 'train'
            url = f'{protocol}://{host}:{port}/{endpoint}'

            response = requests.post(url, json=api_json)
            # st.write(model_request)
            train_json_out = response.json()
        st.session_state.train_json_out = train_json_out
        st.session_state.model = train_json_out['model']
        # st.success("Model trained successfully!")
        # st.write("### Model Response")
        # st.json(train_json_out)
        st.success("Model trained successfully!")

        # Display Model Response
        st.write("#### Model Response")

        # Convert JSON to string for display
        if 'train_response' not in st.session_state:
            st.session_state.train_response = train_json_out
        json_string = json.dumps(st.session_state.train_response, indent=4)
        st.write(f"```json\n{json_string}\n```")


    # st.write(model_request)
    st.write("### :blue[Forecast Function]")
    st.write("Welcome to the **Forecast Function**! Here, you can make forecasts on future data using the trained model. Let's get started!")
    future_file = st.file_uploader("Upload future data CSV", type="csv")
    if future_file is not None:
        future_dataset = pd.read_csv(future_file)
        st.write("## Future Dataset Preview")
        st.dataframe(future_dataset)

        future_time_col = future_dataset.columns[0]
        future_target_col = future_dataset.columns[1] if len(future_dataset.columns) > 1 else None
        st.write("#### Training Model Response")
        json_string = json.dumps(st.session_state.train_response, indent=4)
        st.write(f"```json\n{json_string}\n```")
        # Retrain model on the entire dataset before making future forecasts
        # st.write(model_request)
        if st.button("Forecast Future Data"):
            # st.write(model_request)
            if st.session_state.model is not None:
                # Retrain model with 100% of the data
                train_data = prepare_dataset_forecast(st.session_state.dataset, st.session_state.time_col, target_col)
                # st.write(model_request)
                with st.spinner("Prediction in progress..."):
                    api_json = {
                        'data': train_data,
                        'model': model_request
                    }

                    endpoint = 'train'
                    url = f'{protocol}://{host}:{port}/{endpoint}'

                    response = requests.post(url, json=api_json)
                    train_json_out = response.json()



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
                #
                # st.write("### Future Forecast Results")
                # future_forecast_df[future_time_col] = pd.to_datetime(future_forecast_df[future_time_col])
                # # future_dataset = pd.concat([future_dataset, future_forecast_df['Forecast']], axis=1)
                # future_forecast_df[future_time_col]= future_dataset[future_time_col]
                # st.dataframe(future_forecast_df)
                #
                # # Create Plotly plot
                # fig = go.Figure()
                #
                # # Add training data trace
                # fig.add_trace(go.Scatter(
                #     x=pd.to_datetime(st.session_state.dataset[st.session_state.time_col]),
                #     y=st.session_state.dataset[target_col],
                #     mode='lines+markers',
                #     name='Training Data',line=dict(color='black')
                # ))
                #
                # # Add future forecast trace
                # st.write("debug dataframe")
                # st.dataframe(future_forecast_df)
                # st.write(future_forecast_df[future_time_col])
                # st.write(future_forecast_df['Forecast'])
                # fig.add_trace(go.Scatter(
                #     x=future_forecast_df[future_time_col],
                #     y=future_forecast_df['Forecast'],
                #     mode='lines',
                #     name='Future Forecast',
                #     line=dict(color='green')
                # ))
                #
                # # Add vertical line for the end of training
                # end_of_training_date = pd.to_datetime(st.session_state.dataset[st.session_state.time_col].iloc[-1])
                # fig.add_vline(x=end_of_training_date, line=dict(color='black', dash='dash'), name='End of Training')
                #
                # # Update layout
                # fig.update_layout(
                #     xaxis_title="Date",
                #     yaxis_title="Value",
                #     legend_title="Legend"
                # )
                #
                # st.plotly_chart(fig)
                #
                st.write("#### Future Forecast Results")
                future_forecast_df[future_time_col] = pd.to_datetime(future_forecast_df[future_time_col])
                # future_dataset = pd.concat([future_dataset, future_forecast_df['Forecast']], axis=1)
                future_forecast_df[future_time_col]= future_dataset[future_time_col]
                future_forecast_df[future_time_col] = pd.to_datetime(future_forecast_df[future_time_col])
                st.write("#### Future Forecast")
                st.dataframe(future_forecast_df)
                # st.write("Debugging DataFrame")
                # st.dataframe(future_forecast_df)

                # Create Plotly plot
                fig = go.Figure()

                # Add training data trace
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(st.session_state.dataset[st.session_state.time_col]),
                    y=st.session_state.dataset[target_col],
                    mode='lines+markers',
                    name='Training Data',
                    line=dict(color='black')
                ))

                # Add future forecast trace
                fig.add_trace(go.Scatter(
                    x=future_forecast_df[future_time_col],
                    y=future_forecast_df['Forecast'],
                    mode='lines',
                    name='Future Forecast',
                    line=dict(color='blue')
                ))

                # Add vertical line for the end of training
                end_of_training_date = pd.to_datetime(st.session_state.dataset[st.session_state.time_col].iloc[-1])
                fig.add_shape(type="line",
                              x0=end_of_training_date, y0=0, x1=end_of_training_date, y1=1,
                              xref='x', yref='paper',
                              line=dict(color='black', dash='dash'))

                # Update layout
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Value",
                    legend_title="Legend"
                )

                st.plotly_chart(fig)

                # If the future dataset contains actual values, plot them
                if future_target_col is not None:
                    st.write("#### Future Forecast with Actual Values")

                    # Convert future_time_col to datetime
                    future_dataset[future_time_col] = pd.to_datetime(future_dataset[future_time_col])
                    future_forecast_df[future_time_col] = pd.to_datetime(future_forecast_df[future_time_col])

                    # Create Plotly plot
                    fig = go.Figure()

                    # Add training data trace
                    fig.add_trace(go.Scatter(
                        x=pd.to_datetime(st.session_state.dataset[st.session_state.time_col]),
                        y=st.session_state.dataset[target_col],
                        mode='lines+markers',
                        name='Training Data',
                        line=dict(color='black'),
                    ))

                    # Add actual values trace
                    fig.add_trace(go.Scatter(
                        x=future_dataset[future_time_col],
                        y=future_dataset[future_target_col],
                        mode='lines+markers',
                        name='Actual Values',
                        line=dict(color='black')
                    ))

                    # Add future forecast trace
                    fig.add_trace(go.Scatter(
                        x=future_forecast_df[future_time_col],
                        y=future_forecast_df['Forecast'],
                        mode='lines',
                        name='Future Forecast',
                        line=dict(color='blue')
                    ))

                    # Add vertical line for the end of training
                    end_of_training_date = pd.to_datetime(st.session_state.dataset[st.session_state.time_col].iloc[-1])
                    fig.add_vline(x=end_of_training_date, line=dict(color='black', dash='dash'), name='End of Training')

                    # Update layout
                    fig.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Value",
                        legend_title="Legend"
                    )

                    st.plotly_chart(fig)
                    #concat ds y and actual values


                    future_dataset= pd.concat([future_dataset, future_forecast_df['Forecast']], axis=1)
                    st.write("### Future Dataset with Forecasted and Actual Values")
                    st.dataframe(future_dataset)

            else:
                st.warning("Please train the model first.")

# Stop Uvicorn server when Streamlit app is closed
if st.button("Stop Uvicorn Server"):
    stop_uvicorn_server(server_process)

st.write("&nbsp;&nbsp;&nbsp;&nbsp;")
st.write(":grey[Powered by]")
st.image("temporai_logo.jpg", width=200)