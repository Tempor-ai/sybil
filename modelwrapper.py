import numpy as np
from ts_utils import *
from statsforecast.models import Theta, AutoARIMA, SeasonalNaive


class ModelWrapper:
    def __init__(self, data, model_type="Theta", **kwargs):
        """
        Initialize the wrapper with a specific model type.
        """
        self.data = data
        if is_seasonal(self.data):
            self.season_length = self.seasonal_length()
        else:
            self.season_length = 1
        self.models = {
            "Theta": Theta(season_length=self.season_length, **kwargs),
            "AutoARIMA": AutoARIMA(season_length=self.season_length, **kwargs),
            "SeasonalNaive": SeasonalNaive(season_length=self.season_length, **kwargs),
        }

        if model_type not in self.models:
            raise ValueError(
                f"Model type '{model_type}' not supported. Choose from {list(self.models.keys())}."
            )

        self.model = self.models[model_type]
        # self.ratio = ratio
        # self.h = h
        # self.season_length = seasonal_length()

    def _split_data(self, ratio=0.1):
        """
        Split data based on the given ratio.
        """
        train_set, test_set = np.split(self.data, [int(ratio * len(self.data))])
        return train_set, test_set

    def seasonal_length(self):
        return 24

    def fit_predict(self, ratio):
        """
        Fit the selected model.
        """
        X, y = self._split_data(ratio)
        model = self.model.fit(X.value.to_numpy())
        y_hat = model.predict(h=len(y))

        return y, y_hat

    # def predict(self):
    #     """
    #     Make predictions using the trained model.
    #     """
    #     return self.model.predict(h)

    def forecast(self, h=1):
        """
        Train using all data and forecast the next step.
        """
        return self.model.forecast(self.data.value.to_numpy(), h)
