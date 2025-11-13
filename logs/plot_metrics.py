import json
import matplotlib.pyplot as plt
from datetime import datetime

# Load the metrics logs
with open("metrics_logs.json", "r") as f:
    logs = json.load(f)

# Extract timestamps, accuracy, and F1 scores
timestamps = []
accuracy = []
f1_score = []

for entry in logs:
    metrics = entry.get("metrics", {})
    if "accuracy" in metrics and "f1_score" in metrics:
        timestamps.append(datetime.fromisoformat(entry.get("timestamp", "2025-01-01T00:00:00")))
        accuracy.append(metrics["accuracy"])
        f1_score.append(metrics["f1_score"])

# Plotting
plt.figure(figsize=(10,5))
plt.plot(timestamps, accuracy, marker='o', label='Accuracy')
plt.plot(timestamps, f1_score, marker='s', label='F1 Score')
plt.xlabel("Timestamp")
plt.ylabel("Score")
plt.title("Model Accuracy & F1 Score Over Time")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
