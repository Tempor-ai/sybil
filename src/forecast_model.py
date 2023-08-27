import argparse
import csv
import requests
import json
import os
import logging

from typing import Union

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
OUT_FILE = os.path.join(SCRIPT_DIR, "forecast_response.json")
DEFAULT_MODEL_RESPONSE = os.path.join(SCRIPT_DIR, "train_response.json")
DATASET_TARGETS = os.path.join(SCRIPT_DIR, "..", "targets", "retail", "air_passengers.csv")


def main():

    # Read the dataset path from the command line
    parser = argparse.ArgumentParser(description='Send /train request to Sybil from file, receive response and place it in file')
    parser.add_argument('-d', '--dataset', type=str, help='Path or url to the target indexs for the forecasting.', default=DATASET_TARGETS)
    parser.add_argument('-H', '--host', type=str, help='Domain or IP address of the Sybil gateway', default="3.92.34.145")
    parser.add_argument('-p', '--port', type=int, help='TCP/IP port of Sybil gateway', default=443)
    parser.add_argument('-f', '--outfile', type=str, help='Output file for JSON response', default=OUT_FILE)
    parser.add_argument('-n', '--notsecure', action='store_true', help='If passed, protocol will be swapped to HTTP', default=False)
    parser.add_argument('-m', '--modelresponse', type=str, help='model ', default=DEFAULT_MODEL_RESPONSE)

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
            data.append(row[0])

    # Build /forecast endpoint JSON request

    # Extract model out of /train Response
    logger.info("Model Response: %s", args["modelresponse"])
    with open(args["modelresponse"], "r") as model_response:
        json_in = json.loads(model_response.read())
        model = json_in["model"]

    api_json = {
        'model': model,
        'predicts': data
    }

    # Build URL
    protocol = "https"
    if args["notsecure"] is True:
        logger.info("Using non-secure HTTP for connection")
        protocol = "http"
    else:
        logger.info("Using HTTPS for connection")

    url = "%s://%s:%s/forecast" % (protocol, args["host"], args["port"])
    logger.info("Calling RPC endpoint %s", url)

    # TODO gain a accredited cert and remove verify=False
    # Call endpoint, receive response JSON, write to output file
    response = requests.post(url, json=api_json, verify=False)

    json_out = json.dumps(response.json(), indent=4)
    logger.info("Response JSON, %s", json_out)
    with open(args["outfile"], "w") as file:
        file.write(json_out)


if __name__ == "__main__":
    main()
