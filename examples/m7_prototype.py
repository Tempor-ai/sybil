"""
Prototype for the 7th milestone of the project.
Using meta model and Darts LightGBM.
"""

import pickle
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.modelfactory import ModelFactory

DEFAULT_DATASET = "datasets/climate/temp_anom_w_forcing.csv"


def main():
    # Read the dataset path from the command line
    parser = argparse.ArgumentParser(description='Train a model using Statsforecast and predict on a dataset.')
    parser.add_argument('-d', '--dataset', type=str,
                        help='Path or url to the dataset csv file.',
                        default=DEFAULT_DATASET)
    args = vars(parser.parse_args())
    dataset = ModelFactory.prepare_dataset(pd.read_csv(args['dataset']))
    train_data, test_data = train_test_split(dataset, test_size=0.2, shuffle=False)

    # Simulate API json payload received from the user
    api_json = {
        'dataset': train_data,
        'type': 'darts_lightgbm',
        'scorers': ['smape', 'mape'],
        'params': {}
    }

    # Prepare the dataset and create the model
    model = ModelFactory.create_model(**api_json)

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
    print(model.score(y_test, X=x_test))
    print(y_pred)


if __name__ == "__main__":
    main()
