import json
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime
from src.federated import start_federated_training  # assume you have this
from src.feature_utils import compute_metrics  # if available

# Path to save logs
LOG_FILE = os.path.join("logs", "evaluation_results.json")

def run_evaluation(strategy_name: str, num_rounds: int = 5):
    """
    Run one federated training session and return metrics.
    """
    print(f"\n🚀 Running Federated Evaluation for strategy: {strategy_name}\n")
    
    # Simulate a call to your federated training logic
    history = start_federated_training(strategy_name=strategy_name, rounds=num_rounds)
    
    # Extract metrics (simulate if not integrated yet)
    accuracy = history.get("accuracy", 0.0)
    f1 = history.get("f1_score", 0.0)
    
    result = {
        "strategy": strategy_name,
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4),
        "rounds": num_rounds,
        "timestamp": datetime.now().isoformat()
    }
    
    print(f"✅ Completed {strategy_name}: {result}\n")
    return result

def main():
    # Dictionary to store all evaluation results
    evaluation_results = {}

    # Run for all three strategies
    for strategy in ["RankingFed", "FedAVG", "Astraes"]:
        results = run_evaluation(strategy)
        evaluation_results[strategy] = results

    # Save all results
    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, "w") as f:
        json.dump(evaluation_results, f, indent=4)

    print("\n📊 Evaluation Summary Saved to logs/evaluation_results.json\n")

if __name__ == "__main__":
    main()
