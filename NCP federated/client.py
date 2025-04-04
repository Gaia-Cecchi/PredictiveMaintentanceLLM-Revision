import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import flwr as fl
import tensorflow as tf

from utils import batch_size, num_epochs, load_model, get_model_parameters, set_model_params, getClientData, getTrainAndTestData


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def get_parameters(self, config):
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        set_model_params(self.model, parameters)
        self.model.fit(self.x_train, self.y_train, epochs=num_epochs, verbose=0, batch_size=batch_size)
        return get_model_parameters(self.model), len(self.x_train), {}

    def evaluate(self, parameters, config):
        set_model_params(self.model, parameters)
        loss, mse = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"mse": mse}


# Define Client function
def client_fn(x_train, y_train, x_test, y_test):
    model, _ = load_model()
    return FlowerClient(model, x_train, y_train, x_test, y_test)


def main() -> None:
    # Parse command line argument `dataset`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--client-id",
        type=int,
        default=0,
        choices=range(0, 4),
        required=True,
        help="Specifies the dataset to be used. "
        "Picks 'Compressore 1.xlsx' by default",
    )
    parser.add_argument(
        "--toy",
        action="store_true",
        help="Set to true to quicky run the client using only 10 datasamples. "
        "Useful for testing purposes. Default: False",
    )
    args = parser.parse_args()

    # Get client data
    machines = ["Compressore 1", "Compressore 2", "Compressore 3", "Compressore 4"]  # adjust according to the machines you have
    client_df = getClientData(machines[args.client_id])
    x_train, y_train, x_test, y_test, _ = getTrainAndTestData(client_df)

    if args.toy:
        x_train, y_train = x_train[:10], y_train[:10]
        x_test, y_test = x_test[:10], y_test[:10]

    # Start client
    client_function = client_fn(x_train, y_train, x_test, y_test)
    fl.client.start_client(
        server_address = "localhost:8080",
        client = client_function.to_client(),
    )


if __name__ == "__main__":
    main()