import flwr as fl
from flwr.server.server import ServerConfig
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from typing import List, Tuple
import numpy as np

# -----------------------------
# Ranking-based aggregation strategy
# -----------------------------
class RankingFed(fl.server.strategy.FedAvg):
    """
    RankingFed: Weighted average aggregation where client weights are proportional
    to a reported performance metric (e.g., accuracy).
    """

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ):
        if not results:
            print("⚠️ No client results received this round.")
            return None, {}

        # Extract parameter ndarrays and corresponding metrics
        param_list = []
        metrics_list = []
        num_examples = []  # fallback if metric missing

        for _, fit_res in results:
            # fit_res.parameters is fl.common.Parameters
            arrs = parameters_to_ndarrays(fit_res.parameters)
            param_list.append(arrs)

            # Try reading an explicit performance metric reported by client (e.g., accuracy)
            metric = None
            if fit_res.metrics:
                # prefer accuracy, fallback to 'f1' if provided
                metric = fit_res.metrics.get("accuracy") or fit_res.metrics.get("f1")
            metrics_list.append(metric)

            # fallback to example count if available
            num_ex = fit_res.num_examples if hasattr(fit_res, "num_examples") else None
            num_examples.append(num_ex if num_ex is not None else 1)

        # Convert None metrics to small value to avoid zero weight
        metrics_arr = np.array([m if (m is not None) else 0.0 for m in metrics_list], dtype=float)

        # If every metric is zero (or none), fallback to sample-size-weighted FedAvg
        if np.all(metrics_arr == 0.0):
            print("⚠️ No client metrics reported; falling back to FedAvg by sample size.")
            # Delegate to parent (FedAvg) which uses num_examples weighting
            return super().aggregate_fit(server_round, results, failures)

        # Normalize weights (we use metrics as positive weights)
        weights = metrics_arr / (metrics_arr.sum() + 1e-12)

        # Weighted aggregation: param_list is list of lists-of-arrays; shape: (n_clients, n_param_arrays)
        n_clients = len(param_list)
        n_param_arrays = len(param_list[0])
        aggregated = []

        for idx in range(n_param_arrays):
            # stack each client's idx-th param
            stacked = np.stack([param_list[c][idx] for c in range(n_clients)], axis=0)
            # weighted average across clients
            agg = np.tensordot(weights, stacked, axes=(0, 0))
            aggregated.append(agg.astype(stacked.dtype))

        aggregated_parameters = ndarrays_to_parameters(aggregated)

        # Optionally create aggregated metrics dict
        agg_metrics = {"avg_weighted_metric": float((metrics_arr * weights).sum())}

        print(f"📊 Round {server_round} - RankingFed weights: {weights.tolist()}")
        print(f"📊 Round {server_round} - aggregated metric (weighted avg): {agg_metrics['avg_weighted_metric']:.4f}")

        return aggregated_parameters, agg_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ):
        # reuse FedAvg evaluate aggregation for simplicity
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        print(f"✅ Evaluation aggregated for round {server_round}")
        return aggregated_loss, aggregated_metrics


# -----------------------------
# Entrypoint: choose strategy here (RankingFed)
# -----------------------------
def main():
    # Number of rounds
    num_rounds = 5

    # Choose strategy: RankingFed or FedAvg (baseline)
    strategy = RankingFed(
        fraction_fit=1.0,       # use all clients
        fraction_evaluate=1.0,  # evaluate on all clients
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        on_fit_config_fn=lambda rnd: {"local_epochs": 1},
    )

    print("🚀 Starting Flower server with RankingFed strategy...")
    fl.server.start_server(
        server_address="localhost:8080",
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
