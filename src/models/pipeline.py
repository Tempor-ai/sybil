"""
Pipeline of transformers and models.
"""

import numpy as np
import pandas as pd
from .modelwrappers import AbstractModel
from .preprocessor import AbstractPreprocessor
from typing import List


class Pipeline(AbstractModel):
    # TODO: Transform also y?
    """
    Pipeline of transformers and models.
    """
    def __init__(self, processors: List[AbstractPreprocessor], model: AbstractModel, *args, **kwargs):
        self.transformers = processors
        self.model = model
        super().__init__(*args, **kwargs)

    def _train(self, y: pd.Series, X: pd.DataFrame=None) -> float:
        for transformer in self.transformers:
            if X is not None:
                X = transformer.fit_transform(X)
        return self.model._train(y, X)

    def _predict(self, lookforward: int=1, X: pd.DataFrame=None) -> np.ndarray:
        if X is not None:
            for transformer in self.transformers:
                X = transformer.transform(X)
        y = self.model.predict(lookforward, X)
        return y