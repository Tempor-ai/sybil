"""
Prototype for the integration of Deep SYBIL into SYBIL of this project.
"""
import yaml
import pickle
import argparse
import pandas as pd
from fastapi.encoders import jsonable_encoder
from sklearn.model_selection import train_test_split

from src.models.modelfactory import ModelFactory

# Through online GitHub
DEFAULT_DATASET = 'https://github.com/ourownstory/neuralprophet-data/raw/main/datasets/air_passengers.csv'
# DEFAULT_DATASET = "datasets/climate/temp_anom_w_forcing.csv"


def main():
    # Read the dataset path from the command line
    parser = argparse.ArgumentParser(description='Train a model using Statsforecast and predict on a dataset.')
    parser.add_argument('-d', '--dataset', type=str,
                        help='Path or url to the dataset csv file.',
                        default=DEFAULT_DATASET)
    args = vars(parser.parse_args())

    # 1) Extract Data ################################
    train_df = pd.read_csv(args['dataset'])
    # Define the required time and target columns
    time_col = train_df.columns[0]
    target_col = train_df.columns[-1]
    train_df[time_col] = train_df[time_col].astype(str)
    # Change target column to float
    train_df[target_col] = train_df[target_col].astype(float)
    
    # 2) Train API ################################ 
    # Train data: convert df to list-of-list
    train_data = []
    for value in train_df.values:
        train_data.append(list(value))
    # This is for YAML model_request
    file_path = 'sybil/examples/model_request.yaml'
    # For reading the model request from a yaml file
    with open(file_path, 'r') as file:
        model_request = yaml.safe_load(file)
    # Train API JSON Payload
    api_json = {
        'data': train_data,
        'model': model_request  # (optional) can be commented out
    }

    # 3) SYBIL service routes ################################
    train_request = api_json
    dataset = ModelFactory.prepare_dataset(pd.DataFrame(train_request['data']))
    # dataset = ModelFactory.prepare_dataset(pd.DataFrame(train_request.data))  # TO:DO

    # Get optional user specs
    model_info = train_request['model']
    # model_info = train_request.model  # TO:DO

    # If user did not pass in the model spec
    if model_info is None:
        # Create model objects with the ModelFactory defaults
        model = ModelFactory.create_model(dataset)
    else:
        model_info_json = jsonable_encoder(model_info)
        # Create model objects from the spec user passed in
        model = ModelFactory.create_model(dataset=dataset, **model_info_json)

    # Train model
    training_info = model.train(dataset)

    # Serialize, compress, and finally encode model in base64 ASCII, so it can be sent in JSON
    # output_model = base64.b64encode(blosc.compress(pickle.dumps(training_info['model'])))

    # output_model = ModelFactory.save(training_info['model'])
    
    # Build metrics JSON for response
    # metrics = []
    # for metric_type in training_info["metrics"]:
    #     metrics.append(Metric(type=metric_type,
    #                           value=training_info["metrics"][metric_type]
    #                           ))
    # if len(metrics) == 0:
    #     metrics = None

    # There is dynamacism in the metrics field
    # return TrainResponse(model=output_model,
    #                      type=training_info["type"],
    #                      metrics=metrics
    #                      )





























    dataset = ModelFactory.prepare_dataset(pd.read_csv(args['dataset']))
    train_data, test_data = train_test_split(dataset, test_size=0.2, shuffle=False)



    print(model_request)

    # Simulate API json payload received from the user
    api_json = {
        'data': train_data,
        'model': model_request  # (optional) can be commented out
}
    # api_json = {
    #     'dataset': train_data,
    #     'type': 'darts_lightgbm',
    #     'scorers': ['smape', 'mape'],
    #     'params': {}
    # }

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
