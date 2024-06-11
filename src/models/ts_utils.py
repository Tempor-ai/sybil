import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Defining an Epsilon for handling div/0 edge cases
epsilon=1e-10



def get_frequency(time_series):
    # Check if the index of the time series is a DateTimeIndex
    if isinstance(time_series.index, pd.DatetimeIndex):
        # Infer the frequency of the time series
        frequency = pd.infer_freq(time_series.index)

        if frequency is None:
            return "Frequency not found"
        else:
            return frequency
    else:
        return "DateTimeIndex not found"


def is_seasonal(time_series):
    # Decompose the time series
    decomposition = sm.tsa.seasonal_decompose(time_series)

    # Retrieve the seasonal component
    seasonality = decomposition.seasonal

    # Check if there is seasonality
    if seasonality is not None:
        return True
    else:
        return False


def get_seasonal_period(time_series, plot=False):
    # Calculate the autocorrelation of the time series
    autocorrelation = sm.tsa.acf(time_series, fft=True)

    if plot:
        # Plot the autocorrelation
        plt.figure(figsize=(12, 6))
        plt.stem(range(len(autocorrelation)), autocorrelation)
        plt.xlabel("Lag")
        plt.ylabel("Autocorrelation")
        plt.title("Autocorrelation Plot")
        plt.show()

    # Find the index of the highest peak after the first peak
    seasonal_period = 0
    highest_peak = 0
    for i in range(1, len(autocorrelation) - 1):
        if (
            autocorrelation[i] > autocorrelation[i - 1]
            and autocorrelation[i] > autocorrelation[i + 1]
            and autocorrelation[i] > highest_peak
        ):
            highest_peak = autocorrelation[i]
            seasonal_period = i

    return seasonal_period


def smape(y_true, y_pred, y_train=None):
    """
    Calculate the SMAPE (Symmetric Mean Absolute Percentage Error) between two arrays.
    :param y_true: numpy array or list, representing the true values
    :param y_pred: numpy array or list, representing the predicted values
    :return: float, SMAPE value
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    numerator = np.abs(y_pred - y_true)
    #Adding epsilon
    denominator = ((np.abs(y_true) + np.abs(y_pred)) / 2.0)+epsilon

    smape = np.mean(numerator / denominator) * 100.0

    return smape


def mape(y_true, y_pred, y_train=None):
    """
    Calculate the MAPE (Mean Absolute Percentage Error) between two arrays.
    :param y_true: numpy array or list, representing the true values
    :param y_pred: numpy array or list, representing the predicted values
    :return: float, MAPE value
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    #Adding Epsilon
    percentage_errors = np.abs((y_true - y_pred) / (y_true+epsilon))
    mape = np.mean(percentage_errors) * 100.0

    return mape


def mase(y_true, y_pred, y_train):
    """
    Calculate the MASE (Mean Absolute Scaled Error) between the true values and predicted values.
    :param y_true: numpy array or list, representing the true values
    :param y_pred: numpy array or list, representing the predicted values
    :param y_train: numpy array or list, representing the historical training values
    :return: float, MASE value
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_train = np.array(y_train)

    training_error = np.mean(np.abs(y_train[1:] - y_train[:-1]))
    forecast_error = np.mean(np.abs(y_true - y_pred))

    mase = forecast_error / (training_error+epsilon)

    return mase


def plot_forecast(actual_data, forecast):
    # Plot the actual values and the forecasted values
    plt.plot(actual_data.index, actual_data["value"], label="Actual")
    plt.plot(forecast.index, forecast, label="Forecast")
    plt.xlabel("Date")
    plt.xticks(rotation=45)
    plt.ylabel("Value")
    plt.legend()
    plt.show()
