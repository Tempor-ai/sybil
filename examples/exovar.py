import json
import yaml
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# Local directory
data_dir = '../datasets/climate'
file_name = 'temp_anom_w_forcing'
# data_dir = '../datasets/finance'
# file_name = 'Returns_short_interest_data_train'
file_path = f'{data_dir}/{file_name}.csv'

dataset = pd.read_csv(file_path)

print("Dataset Shape:", dataset.shape)
print("Dataset Columns:", dataset.columns)
print("Dataset Head:", dataset.head())
print("Dataset Tail:", dataset.tail())

time_col = dataset.columns[0]
target_col = dataset.columns[-1]
dataset[time_col] = dataset[time_col].astype(str)

train_size = 0.8
train_points = int(train_size * len(dataset))
train_df = dataset.iloc[:train_points]
test_df = dataset.iloc[train_points:]

figsize = (16, 8)
train_df.set_index(time_col)[target_col].plot(figsize=figsize)
plt.show()

train_data = [list(value) for value in train_df.values]

# Customized model request
model_request = {
    'type': 'meta_lr',
    'scorers': ['smape', 'mape'],
    'params': {
        'preprocessors': [
            # {'type': 'dartsimputer', 'params': {'strategy': 'mean'}},
            {'type': 'minmaxscaler'},
        ],
        'base_models': [
            {'type': 'darts_naive'},
            {'type': 'darts_seasonalnaive'},
            # {'type': 'darts_autotheta'},
            # {'type': 'darts_autoets'},
            # {'type': 'darts_autoarima'},
            # {'type': 'darts_tbats'},
            # {'type': 'darts_lightgbm',
            #  'params': {
            #      'lags': 12,
            #      'lags_future_covariates': [0, 1, 2],
            #      'output_chunk_length': 6,
            #      'verbose': -1
            #  }
            #  },
            # {'type': 'darts_rnn',
            #  'params': {
            #      'model': 'LSTM',
            #      'hidden_dim': 10,
            #      'n_rnn_layers': 3
            #  }
            #  },
        ],
    },
}

api_json = {
    'data': train_data,
    'model': model_request
}

with open('url.yaml', 'r') as file:
    url_dict = yaml.safe_load(file)

# URL to our SYBIL AWS service
protocol = url_dict['protocol']
host = url_dict['host']
port = url_dict['port']
endpoint = 'train'

url = f'{protocol}://{host}:{port}/{endpoint}'

response = requests.post(url, json=api_json)
print("Response:", response)
print()

train_json_out = response.json()
print("Train JSON Output:", train_json_out)

test_data = [list(value) for value in test_df.drop(columns=target_col).values]

model = train_json_out['model']

api_json = {
    'model': model,
    'data': test_data
}

endpoint = 'forecast'
url = f'{protocol}://{host}:{port}/{endpoint}'

response = requests.post(url, json=api_json)
print("Response:", response)
print()

forecast_json_out = response.json()
print("Forecast JSON Output:", forecast_json_out)

forecast_df = pd.DataFrame(
    data=forecast_json_out['data'],
    columns=[time_col, target_col],
)

print("Forecast DataFrame Shape:", forecast_df.shape)
print("Forecast DataFrame Columns:", forecast_df.columns)
print("Forecast DataFrame Head:", forecast_df.head())
print("Forecast DataFrame Tail:", forecast_df.tail())

train_df['color'] = 'b'
train_df.set_index(time_col)[target_col].plot(figsize=figsize, color=train_df['color'])
forecast_df['color'] = 'r'
forecast_df.set_index(time_col)[target_col].plot(figsize=figsize, color=forecast_df['color'])
plt.show()

df = pd.concat([train_df, forecast_df]).reset_index(drop=True)

print("Combined DataFrame Shape:", df.shape)
print("Combined DataFrame Head:", df.head())
print("Combined DataFrame Tail:", df.tail())

df.set_index(time_col)[target_col].plot(figsize=figsize, color='r')
plt.axvline(x=len(train_df), color='black', label='Train/Forecast set cut-off')
plt.text(x=len(train_df) - 9, y=forecast_df[target_col].max(), s='Train', fontweight='bold', fontsize=14)
plt.text(x=len(train_df) + 1, y=forecast_df[target_col].max(), s='Forecast', fontweight='bold', fontsize=14)
plt.ylabel('Temperature Anomaly (' + u'\N{DEGREE SIGN}' + 'C)')
plt.title('Annual Temperature Anomaly (Train: 1850-1979) (Forecast: 1980-2012)', fontweight='bold', fontsize=20)
plt.show()

dataset.set_index(time_col)[target_col].plot(figsize=figsize)
plt.ylabel('Temperature Anomaly (' + u'\N{DEGREE SIGN}' + 'C)')
plt.title('Annual Temperature Anomaly (1850-2012)', fontweight='bold', fontsize=20)
plt.show()

df.set_index(time_col)[target_col].plot(figsize=figsize, color='r')
dataset.set_index(time_col)[target_col].plot(figsize=figsize)
plt.axvline(x=len(train_df), color='black', label='Train/Forecast set cut-off')
plt.text(x=len(train_df) - 9, y=forecast_df[target_col].max(), s='Train', fontweight='bold', fontsize=14)
plt.text(x=len(train_df) + 1, y=forecast_df[target_col].max(), s='Forecast', fontweight='bold', fontsize=14)
plt.ylabel('Temperature Anomaly (' + u'\N{DEGREE SIGN}' + 'C)')
plt.title('Annual Temperature Anomaly (Train: 1850-1979) (Forecast: 1980-2012)', fontweight='bold', fontsize=20)
plt.show()
