"""
Prototype for the 2nd milestone of the project.
Using the Statsforecast wrapper to train a base model and predict on a dataset.
"""

import pickle
import argparse
import pandas as pd
from src.models.modelfactory import ModelFactory

DATASET_PASSENGER = "datasets/retail/air_passengers.csv"


def main():
    # Read the dataset path from the command line
    parser = argparse.ArgumentParser(description='Train a model using Statsforecast and predict on a dataset.')
    parser.add_argument('-d', '--dataset', type=str,
                        help='Path or url to the dataset csv file.',
                        default=DATASET_PASSENGER)
    args = vars(parser.parse_args())

    # Simulate API json payload received from the user
    api_json = {
        'data': pd.read_csv(args['dataset']).to_json(),
        #'model': {'type': 'darts_autoarima', 'score': ['smape', 'mape'], 'param': {}},
        'model': {'type': 'test', 'score': ['smape', 'mape'], 'param': {}}
        #'model': {'type': 'meta_lr',
        #          'score': ['smape', 'mape'],
        #          'param': {'base_models': [
        #              {'type': 'darts_autoarima'},
        #              {'type': 'darts_autotheta'},
        #              {'type': 'darts_autoets'},
        #              {'type': 'stats_autotheta'},
        #              {'type': 'stats_autoets'},
        #              {'type': 'stats_autoarima'}
        #          ]}}
    }

    # Prepare the dataset and create the model
    dataset = ModelFactory.prepare_dataset(pd.read_json(api_json['data']))
    model = ModelFactory.create_model(dataset,
                                      type=api_json['model']['type'],
                                      scorers=api_json['model']['score'],
                                      model_params=api_json['model']['param'])

    # Train the model
    training_info = model.train(dataset)
    print(f'Training Complete: {training_info}')

    # Pickling and unpickling the model just to test it, where should it happen?
    training_info['model'] = pickle.dumps(training_info['model'])
    training_info['model'] = pickle.loads(training_info['model'])

    # Predict next steps
    y_pred = training_info['model'].predict(lookforward=1)
    print(y_pred)


if __name__ == "__main__":
    main()
