"""
Prototype for the 2nd milestone of the project.
Using the Statsforecast wrapper to train a base model and predict on a dataset.
"""
import argparse
import numpy as np
import pandas as pd
from statsforecast.models import AutoTheta, AutoETS
from modelwrappers import AbstractModel, StatsforecastWrapper
from sklearn.model_selection import train_test_split
from ts_utils import calculate_smape, get_seasonal_period

DATASET_PASSENGER = "datasets/retail/air_passengers.csv"
TIME_COL = 'ds'
VALUE_COL = 'y'


def main():
    # Read the dataset path from the command line
    parser = argparse.ArgumentParser(description='Train a model using Statsforecast and predict on a dataset.')
    parser.add_argument('-d', '--dataset', type=str,
                        help='Path or url to the dataset csv file.',
                        default=DATASET_PASSENGER)
    args = vars(parser.parse_args())

    # Load dataset
    dataset = pd.read_csv(args['dataset'])

    # Train model
    training_info = train(dataset)
    print(f'Training Complete: {training_info}')

    model = training_info['model']
    season_length = training_info['stats']['season_length']

    # Predict next steps
    y_pred = predict(model=model, lookforward=season_length)
    print(y_pred)


def train(data: pd.DataFrame) -> dict:
    """
    Train a model on the given data.

    @param data: The data to train on.
    Leftmost column is the time column and the rightmost column is the value column.
    Any columns in between are exogenous variables.
    @return: A dictionary containing the trained model and any other information.
    """
    # Prepare dataset
    columns = data.columns
    data = prepare_dataset(data, time_col=columns[0], value_col=columns[-1])
    print(data)

    # Create model
    season_length = get_seasonal_period(data)
    statsmodel = AutoTheta(season_length=season_length)
    model = StatsforecastWrapper(model=statsmodel, scorer=calculate_smape)

    # Prepare train and test data
    y = data.value
    y_train, y_test = train_test_split(y, test_size=0.1, shuffle=False)

    # Fit model on train data
    model.fit(y_train)

    # Plot test data and print test score
    test_score = model.score(y_train, y_test)
    print(f"Test score: {test_score:.2f}")
    model.plot_prediction(y_test, y_test)  # TODO: Move model cross validation to abstract class

    # Fit model on all data
    model.fit(y)

    # Return model and any other information
    return {'model': model,
            'evaluation': {'sMAPE': test_score},
            'stats': {'season_length': season_length}}


def predict(model: AbstractModel, lookforward: int) -> np.array:
    """
    Predict the next lookforward steps.

    :param model: Trained model as returned by train().
    :param lookforward: Number of forward steps to predict.
    :return: Array of predicted values of shape (lookforward,).
    """
    return model.predict(lookforward=lookforward)


# TODO: Integrate multivariate datasets
# TODO: How to add features?
def prepare_dataset(dataset: pd.DataFrame, time_col: str, value_col: str) -> pd.DataFrame:
    """
    Prepare the dataset for training or prediction.

    :param dataset: Dataframe containing the dataset.
    :param time_col: Column name identifying the time component.
    :param value_col: Column name identifying the value component.
    :return: Dataframe with the time and value columns renamed to 'datetime' and 'value' respectively.
    """
    clean_dataset = dataset.rename(columns={time_col: 'datetime',
                                            value_col: 'value'})
    clean_dataset['datetime'] = pd.to_datetime(clean_dataset['datetime'])
    return clean_dataset.set_index('datetime')


if __name__ == "__main__":
    main()
