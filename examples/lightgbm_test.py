import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import LightGBMModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(1)
n = 100
date_rng = pd.date_range(start='1/1/2000', periods=n, freq='M')
ts_data = 0.5 * np.arange(n) + np.random.normal(scale=5, size=n)
exog_data = 2.0 * np.arange(n) + np.random.normal(scale=5, size=n)

# Create TimeSeries instances
series = TimeSeries.from_dataframe(pd.DataFrame(ts_data, index=date_rng, columns=['value']))
exog = TimeSeries.from_dataframe(pd.DataFrame(exog_data, index=date_rng, columns=['exog_var']))

# Split into training and validation sets
train, val = series.split_after(pd.Timestamp('2006-01-01'))

# Create and train LightGBM model with specified lags and lags_past_covariates
model = LightGBMModel(lags=12, lags_past_covariates=12)
model.fit(train, past_covariates=exog)

# Make predictions
pred = model.predict(n=len(val), past_covariates=exog)

# Plot actual vs. predicted values
series.plot(label='actual')
pred.plot(label='forecast', lw=2)
plt.legend()
plt.show()

# Print error
print('Mean Absolute Percentage Error: {:.2f}%'.format(mape(pred, val)))
