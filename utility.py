import pandas as pd
from darts import TimeSeries
from darts.metrics.metrics import mape, mse

'''
class ModelDatasetPair:
    def __init__(self,
                 model, 
                 dataset,
                 time_col_name,
                 target_col_name, 
                 past_covariates_names,
                 future_covariates_names,
                 test_raio=0.25,
                 eval_metric_funcs=[mape]):
        
        df = pd.read_csv(dataset)
        num_test = int(test_raio*len(df))

        target_series = TimeSeries.from_dataframe(df, time_col_name, target_col_name)   
        target_series_train, target_series_test = target_series[:-num_test], target_series[-num_test:]

        past_covariates = TimeSeries.from_dataframe(df, time_col_name, past_covariates_names)   
        future_covariates = TimeSeries.from_dataframe(df, time_col_name, future_covariates_names)   

        self.data = {
            'dataset_address': dataset,
            'model_name':model,
            'target_series':{'train':target_series_train, 'test':target_series_test, 'test_preds':None},
            'covariates':{'past':past_covariates, 'future':future_covariates},
            'eval_metric_funcs':eval_metric_funcs,
            'summary_errors':{an_eval_metric_func.__name__:None \
                              for an_eval_metric_func in eval_metric_funcs} # ex. MAPE:0.2, MSE:0.1
        }
'''

class ModelDatasetPair:
    
    '''
    Given a model and dataset, organizes everything 
    needed for training/prediction/evaluation.
    ''' 

    def __init__(self, model, df_address, time_col_name, target_col_name,
                 past_covariates_names, future_covariates_names,
                 test_ratio=0.1, eval_metric_funcs=[mape, mse]):
        
        self.model = model
        self.df_address = df_address        
        self.time_col_name = time_col_name
        self.target_col_name = target_col_name
        self.past_covariates_names = past_covariates_names
        self.future_covariates_names = future_covariates_names
        self.test_ratio = test_ratio
        self.eval_metric_funcs = eval_metric_funcs

        self.df = pd.read_csv(self.df_address) #?
        self.num_test = int(self.test_ratio * len(self.df))

        target_series = self._create_time_series(self.target_col_name)
        self.target_series_train, self.target_series_test = self._split_time_series(target_series)
        
        self.past_covariates, self.future_covariates = None, None
        if self.past_covariates_names:
            self.past_covariates = self._create_time_series(self.past_covariates_names)
        
        if self.future_covariates_names:
            self.future_covariates = self._create_time_series(self.future_covariates_names)

        
        self.pair = self._get_pair()
        
    
    def _get_pair(self):               

        return {
            'model': self.model,
            'df_address': self.df_address,
            'df': self.df,
            'target_series': {'train': self.target_series_train, 
                              'test': self.target_series_test, \
                              'test_preds': None},                              
            'covariates': {'past': self.past_covariates, \
                           'future': self.future_covariates},
            'eval_metric_funcs': self.eval_metric_funcs,
            'summary_errors': {func.__name__: None for \
                               func in self.eval_metric_funcs}
        }
    
    
    def _split_time_series(self, time_series):       
        return time_series[:-self.num_test], time_series[-self.num_test:]
    
    
    def _create_time_series(self, col_to_be_converted):
        return TimeSeries.from_dataframe(self.df, self.time_col_name,\
                                          col_to_be_converted)


class ForecastingManager:
    def __init__(self):
        self.entries_and_results = []

    def add_model_data_pair(self, 
                            model,
                            dataset_address, 
                            time_col_name, 
                            target_col_name,
                            past_covariates_names=None,
                            future_covariates_names=None
                            ) -> None:
              
        result = ModelDatasetPair(model, 
                                  dataset_address, 
                                  time_col_name, 
                                  target_col_name, 
                                  past_covariates_names, 
                                  future_covariates_names)
        
        self.entries_and_results.append(result.pair)

    def get_entries_and_results(self):
        return self.entries_and_results
