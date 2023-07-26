import pandas as pd
from statsforecast.models import AutoTheta
from modelwrappers import StatsforecastWrapper
from sklearn.model_selection import train_test_split
from ts_utils import calculate_smape, get_seasonal_period

#DATASET_PATH = "datasets/SF_hospital_load.csv"
DATASET_PATH = "datasets/retail/air_passengers.csv"
TIME_COL = 'ds'
VALUE_COL = 'y'


def main():
    # Load dataset
    dataset = pd.read_csv(DATASET_PATH)

    # Prepare dataset
    dataset = prepare_dataset(dataset, time_col=TIME_COL, value_col=VALUE_COL)
    print(dataset)

    # Create model
    season_length = get_seasonal_period(dataset)
    statsmodel = AutoTheta(season_length=season_length)
    model = StatsforecastWrapper(model=statsmodel, scorer=calculate_smape)

    # Prepare train and test data
    X = dataset.value
    X_train, X_test = train_test_split(X, test_size=0.1, shuffle=False)

    # Fit model on train data
    model.fit(X_train)

    # Plot test data and print test score
    test_score = model.score(X_train, X_test)
    print(f"Test score: {test_score:.2f}")
    model.plot_prediction(X_test, X_test)

    # Fit model on all data
    model.fit(X)

    # Predict next steps
    y_pred = model.predict(X, lookforward=season_length)
    print(y_pred)


# TODO: Integrate multivariate datasets
# TODO: How to add features?
def prepare_dataset(dataset: pd.DataFrame, time_col: str, value_col: str) -> pd.DataFrame:
    clean_dataset = dataset.rename(columns={time_col: 'datetime',
                                            value_col: 'value'})
    clean_dataset['datetime'] = pd.to_datetime(clean_dataset['datetime'])
    return clean_dataset.set_index('datetime')


if __name__ == "__main__":
    main()
