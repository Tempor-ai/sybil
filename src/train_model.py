import argparse
import csv
import requests
import json
import os
import logging

from typing import Union

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DATASET_PASSENGER = os.path.join(SCRIPT_DIR, "..", "datasets", "retail", "air_passengers.csv")
OUT_FILE = os.path.join(SCRIPT_DIR, "train_response.json")
DEFAULT_MODEL_REQUEST = json.dumps({'type': 'meta_wa',
                                    'score': ['smape', 'mape'],
                                    'param': {
                                        'base_models': [
                                            {'type': 'darts_autoarima'},
                                            {'type': 'darts_autotheta'},
                                            {'type': 'darts_autoets'},
                                            {'type': 'stats_autotheta'}
                                        ]
                                    }})


def main():

    # Read the dataset path from the command line
    parser = argparse.ArgumentParser(description='Send /train request to Sybil from file, receive response and place it in file')
    parser.add_argument('-d', '--dataset', type=str, help='Path or url to the dataset csv file.', default=DATASET_PASSENGER)
    parser.add_argument('-H', '--host', type=str, help='Domain or IP address of the Sybil gateway', default="localhost")
    parser.add_argument('-p', '--port', type=int, help='TCP/IP port of Sybil gateway', default=8000)
    parser.add_argument('-f', '--outfile', type=str, help='Output file for JSON response', default=OUT_FILE)
    parser.add_argument('-n', '--notsecure', action='store_true', help='If passed, protocol will be swapped to HTTP', default=False)
    parser.add_argument('-m', '--model_request', type=str, help='model field of the JSON request', default=DEFAULT_MODEL_REQUEST)

    args = vars(parser.parse_args())

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Convert data from CSV to array of arrays
    dataset = args["dataset"]
    logger.info("Input CSV data: %s", dataset)

    data = []
    with open(dataset) as csvfile:
        reader = csv.reader(csvfile)

        # Skip header row
        try:
            next(reader, None)
        except ValueError:
            pass

        for row in reader:
            row[1] = float(row[1])
            data.append(row)

    # Build /train endpoint JSON request
    model_request = json.loads(args["model_request"])
    logger.info("Model Request: %s", model_request)

    api_json = {
        'data': data,
        'model': model_request
    }

    # Build URL
    protocol = "https"
    if args["notsecure"] is True:
        logger.info("Using non-secure HTTP for connection")
        protocol = "http"
    else:
        logger.info("Using HTTPS for connection")

    url = "%s://%s:%s/train" % (protocol, args["host"], args["port"])
    logger.info("Calling RPC endpoint %s", url)

    # Call endpoint, receive response JSON, write to output file
    response = requests.post(url, json=api_json)

    json_out = json.dumps(response.json(), indent=4)
    logger.info("Response JSON, %s", json_out)
    with open(args["outfile"], "w") as file:
        file.write(json_out)


if __name__ == "__main__":
    main()
