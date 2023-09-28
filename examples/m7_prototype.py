"""
Prototype for the 2nd milestone of the project.
Using the Statsforecast wrapper to train a base model and predict on a dataset.
"""

import pickle
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.modelfactory import ModelFactory

DEFAULT_DATASET = "datasets/retail/air_passengers.csv"


def main():
    # Read the dataset path from the command line
    parser = argparse.ArgumentParser(description='Train a model using Statsforecast and predict on a dataset.')
    parser.add_argument('-d', '--dataset', type=str,
                        help='Path or url to the dataset csv file.',
                        default=DEFAULT_DATASET)
    args = vars(parser.parse_args())

    # Simulate API json payload received from the user
    api_json = {
        'data': pd.read_csv(args['dataset']).to_json(),
        #'model': {'type': 'darts_lightgbm'}  # Used to test lightgbm only
        #'model': {'type': 'darts_rnn'}  # Used to test rnn only
        'model': {}   # Used to test the latest metamodel
    }

    # Prepare the dataset and create the model
    print(f'\nPreparing dataset {args["dataset"]} and creating model')
    dataset = ModelFactory.prepare_dataset(pd.read_json(api_json['data']))
    train_data, test_data = train_test_split(dataset, test_size=0.2, shuffle=False)
    model = ModelFactory.create_model(train_data, **api_json['model'])

    # Train the model
    training_info = model.train(train_data)
    print(f'Training Complete: {training_info}')

    # Pickling and unpickling the model just to test it, where should it happen?
    training_info['model'] = pickle.dumps(training_info['model'])
    model = pickle.loads(training_info['model'])

    # Predict and plot test set
    y_test = test_data.iloc[:, -1]
    x_test = test_data.iloc[:, :-1] if test_data.shape[1] > 1 else None
    y_pred = model.predict(lookforward=len(test_data), X=x_test)
    model.plot_prediction(y_test, X=x_test)
    print(f'\nModel scores OOS {model.score(y_test, X=x_test)}')
    print(f'\nModel predictions\n{y_pred}')


if __name__ == "__main__":
    main()
