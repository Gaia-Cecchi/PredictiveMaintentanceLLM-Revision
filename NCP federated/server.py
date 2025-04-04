import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import flwr as fl

from utils import num_clients, num_rounds, load_model
from typing import List, Tuple, Union, Optional, Dict
from flwr.common import Metrics, Parameters, Scalar, FitRes
from flwr.server.client_proxy import ClientProxy


# Define weighted average function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    examples = [num_examples for num_examples, _ in metrics]
    total_examples = sum(examples)
    weighted_metrics = {}

    for num_examples, m in metrics:
        for key, value in m.items():
            weighted_metrics[key] = weighted_metrics.get(key, 0) + value * (num_examples / total_examples)

    return weighted_metrics


# Custom strategy class to save model weights
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )
            # Save aggregated_ndarrays
            _, name = load_model()  # load model name
            np.savez(f"weights_{name}.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics


def main():
    # Define strategy
    strategy = SaveModelStrategy(
        fraction_fit = 1.0,
        fraction_evaluate = 1.0,
        min_available_clients = num_clients,
        min_fit_clients = num_clients,
        fit_metrics_aggregation_fn = weighted_average,
        evaluate_metrics_aggregation_fn = weighted_average,
    )

    # Start server
    fl.server.start_server(
        server_address = "0.0.0.0:8080",
        config = fl.server.ServerConfig(num_rounds=num_rounds),
        strategy = strategy,
    )
    

if __name__ == "__main__":
    main()