"""
Prototype for the 2nd milestone of the project.
Using the Statsforecast wrapper to train a base model and predict on a dataset.
"""

import pickle
import argparse
import pandas as pd
from modelwrappers import AbstractModel, ModelFactory

DATASET_PASSENGER = "datasets/retail/air_passengers.csv"


def main():
    # Read the dataset path from the command line
    parser = argparse.ArgumentParser(description='Train a model using Statsforecast and predict on a dataset.')
    parser.add_argument('-d', '--dataset', type=str,
                        help='Path or url to the dataset csv file.',
                        default=DATASET_PASSENGER)
    args = vars(parser.parse_args())

    # Simulate API Json payload and create model
    api_json = {
        'data': pd.read_csv(args['dataset']).to_json(),
        'model': {'type': 'autotheta', 'score': ['smape', 'mape'], 'param': {}}
    }
    dataset = AbstractModel.prepare_dataset(pd.read_json(api_json['data']))
    model_info = api_json['model']
    model = ModelFactory.create_model(dataset, model_info=model_info)

    # Train model
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
