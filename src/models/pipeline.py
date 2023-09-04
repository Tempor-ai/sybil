"""
Pipeline of transformers and models.
"""

import numpy as np
import pandas as pd
from .modelwrappers import AbstractModel
from .preprocessor import AbstractPreprocessor
from typing import Union, List


class Pipeline(AbstractModel):
    """
    Pipeline of transformers and models.
    """
    def __init__(self, transformers: List[AbstractPreprocessor], model: AbstractModel, *args, **kwargs):
        self.transformers = transformers
        self.model = model
        super().__init__(*args, **kwargs)

    def fit(self, y: Union[np.ndarray,pd.Series], X: Union[np.ndarray,pd.DataFrame]=None) -> float:
        for transformer in self.transformers:
            y = transformer.fit_transform(y)
            if X is not None:
                X = transformer.fit_transform(X)

        return self.model.fit(y, X)

    def predict(self, lookforward: int=1, X: Union[np.ndarray,pd.DataFrame]=None) -> np.ndarray:
        if X is not None:
            for transformer in self.transformers:
                X = transformer.transform(X)
        y = self.model.predict(lookforward, X)
        for transformer in reversed(self.transformers):
            y = transformer.inverse_transform(y)
        return y