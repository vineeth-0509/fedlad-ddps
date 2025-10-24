def start_federated_training(strategy_name: str, rounds: int = 5):
    """
    Simulate a federated training process.
    Replace this logic with your actual federated learning coordination later.
    """
    print(f"Starting federated training for strategy: {strategy_name} ({rounds} rounds)")

    # Simulated training history (replace with actual metrics)
    history = {
        "accuracy": 0.92 if strategy_name == "RankingFed" else 0.89 if strategy_name == "FedAVG" else 0.87,
        "f1_score": 0.90 if strategy_name == "RankingFed" else 0.88 if strategy_name == "FedAVG" else 0.85,
    }

    print(f"Completed federated training for {strategy_name}")
    return history
