from darts import TimeSeries
from darts.metrics.metrics import mape
import pandas as pd

class ModelDatasetPair:
    def __init__(self, 
                 dataset, 
                 model,
                 test_raio=0.25,
                 eval_metric_funcs=[mape]):
        
        df = pd.read_csv(dataset)
        num_test = int(test_raio*len(df))

        all_data = TimeSeries.from_dataframe(df, "ds", "y")   
        train_series, test_series = all_data[:-num_test], all_data[-num_test:]

        self.data = {
            'dataset_address': dataset,
            'model_name': model,
            'train_series': train_series,
            'test_series': test_series,
            'test_series_pred': None,
            'eval_metric_funcs':eval_metric_funcs,
            'summary_errors':{an_eval_metric_func.__name__:None \
                              for an_eval_metric_func in eval_metric_funcs} # ex. MAPE:0.2, MSE:0.1
        }


class ForecastingManager:
    def __init__(self):
        self.entries_and_results = []

    def add_model_data_pair(self, 
                            dataset, 
                            model) -> None:
              
        result = ModelDatasetPair(dataset, model, eval_metric_funcs=[mape])
        self.entries_and_results.append(result.data)

    def get_entries_and_results(self):
        return self.entries_and_results
