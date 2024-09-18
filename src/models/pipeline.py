"""
Pipeline of transformers and models.
"""

import numpy as np
import pandas as pd
from .modelwrappers import AbstractModel
from .preprocessor import AbstractPreprocessor,MinMaxScaler
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
            if isinstance(transformer, MinMaxScaler):
               y=transformer.fit_transform(y)
        return self.model._train(y, X)
    
    def _predict(self, lookforward: int=1, X: pd.DataFrame=None)-> np.ndarray:
        
        if X is not None:
            for transformer in self.transformers:
                X = transformer.transform(X)
        y = self.model.predict(lookforward, X)
        for transformer in self.transformers:
            if isinstance(transformer, MinMaxScaler):
                y = transformer.inverse_transform(y)
        return y
    
class ExternalPipeline(AbstractModel):
    # TODO: Transform also y?
    """
    Pipeline of transformers and models.
    """
    def __init__(self, processors: List[AbstractPreprocessor], model: AbstractModel, *args, **kwargs):
        self.transformers = processors
        self.model = model
        super().__init__(*args, **kwargs)

    def _train(self, data: pd.DataFrame, external_base_model_config) -> float:
        #TODO impelement preprocessors for NP
        return self.model._train(data, external_base_model_config)

    def _predict(self, lookforward: int=1, X: pd.DataFrame=None)-> np.ndarray:
        #TODO impelement preprocessors for NP
        y = self.model.predict(lookforward, X)
        return y