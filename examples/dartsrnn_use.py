"""
Example file on how to use Darts RNN model
"""

import pickle
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from darts.models.forecasting.rnn_model import RNNModel

from src.models.modelfactory import ModelFactory
from src.models.modelwrappers import DartsWrapper
from src.models.ts_utils import smape, mape

DEFAULT_DATASET = "../datasets/climate/temp_anom_w_forcing.csv"


def main():
    # Read the dataset path from the command line
    parser = argparse.ArgumentParser(description='Train a model using Statsforecast and predict on a dataset.')
    parser.add_argument('-d', '--dataset', type=str,
                        help='Path or url to the dataset csv file.',
                        default=DEFAULT_DATASET)
    args = vars(parser.parse_args())
    dataset = ModelFactory.prepare_dataset(pd.read_csv(args['dataset']))
    train_data, test_data = train_test_split(dataset, test_size=0.2, shuffle=False)

    # Prepare the dataset and create the model (not using the mode factory here)
    darts_model = RNNModel(model='LSTM', input_chunk_length=12, output_chunk_length=1, n_epochs=100)
    model = DartsWrapper(darts_model=darts_model, type='darts_rnn', scorers=[smape, mape])

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
