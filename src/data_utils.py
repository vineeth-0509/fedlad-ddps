import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from typing import Dict, List

# --- Configuration Constants ---
DATA_DIR = "data"
NUM_CLIENTS = 15

# Define the standardized list of known attacks
ALL_ATTACKS = [
    "BENIGN", "SYN", "UDP", "DDoS", "DrDNS", "Slowloris", "Slowread", "Slowheaders",
    "PortScan", "BruteForce", "Infiltration", "WebAttack"
]

# ---------------------------------------------------------------------
# STEP 1: LOAD AND CLEAN DATA
# ---------------------------------------------------------------------
def load_and_clean_data(file_name: str) -> pd.DataFrame:
    """Loads and cleans a dataset (CSV) for DDoS detection."""
    file_path = os.path.join(DATA_DIR, file_name)
    print(f"📂 Loading dataset: {file_path}")

    try:
        df = pd.read_csv(file_path, encoding="latin1", low_memory=False)
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return pd.DataFrame()

    # Standardize column names
    df.columns = df.columns.str.strip().str.replace(" ", "_")

    # Replace infinite values and fill NaNs
    df.replace([np.inf, -np.inf], np.finfo(np.float32).max, inplace=True)
    df.fillna(0, inplace=True)

    # Standardize and clean labels
    if "Label" in df.columns:
        df["Label"] = df["Label"].astype(str).str.strip().str.upper()

    return df


# ---------------------------------------------------------------------
# STEP 2: COMBINE AND MAP LABELS
# ---------------------------------------------------------------------
def preprocess_and_combine() -> pd.DataFrame:
    """Loads, cleans, and merges the 3 datasets into one combined DataFrame."""
    datasets = [
        load_and_clean_data("CICIDS2017.csv"),
        load_and_clean_data("CICDDOS2019.csv"),
        load_and_clean_data("InSDN.csv")
    ]

    valid_datasets = [d for d in datasets if not d.empty]
    if not valid_datasets:
        raise ValueError("❌ No datasets loaded successfully.")

    combined_df = pd.concat(valid_datasets, ignore_index=True)

    # Map labels to unified attack names
    def map_label(label):
        if "BENIGN" in label: return "BENIGN"
        if "SLOW" in label:
            if "LORIS" in label: return "Slowloris"
            if "READ" in label: return "Slowread"
            if "HEADER" in label: return "Slowheaders"
        if "DDOS" in label or "DOS" in label:
            if "SYN" in label: return "SYN"
            if "UDP" in label: return "UDP"
            if "DRDNS" in label: return "DrDNS"
            return "DDoS"
        if "PORTSCAN" in label: return "PortScan"
        if "BRUTE" in label: return "BruteForce"
        if "INFILTRATION" in label: return "Infiltration"
        if "WEBATTACK" in label: return "WebAttack"

        # Fallback
        return "DDoS" if label not in ALL_ATTACKS else label

    combined_df["Label"] = combined_df["Label"].apply(map_label)
    print(f"✅ Combined dataset shape: {combined_df.shape}")
    print(f"🔖 Unique labels: {combined_df['Label'].unique()}")

    return combined_df


# ---------------------------------------------------------------------
# STEP 3: NON-IID PARTITIONING
# ---------------------------------------------------------------------
def partition_data_non_iid(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """Partition data into 15 non-IID subsets (simulated clients)."""
    print(f"\n⚙️ Partitioning data into {NUM_CLIENTS} non-IID subsets...")

    benign_data = df[df["Label"] == "BENIGN"]
    attack_data = df[df["Label"] != "BENIGN"]

    client_partitions = {}

    for i in range(NUM_CLIENTS):
        # Define attack group types
        if i < 5:  # High-volume (SYN/UDP/DDoS)
            attack_subset = attack_data[attack_data["Label"].isin(["SYN", "UDP", "DDoS", "DrDNS"])]
        elif i < 10:  # Low-rate/probing
            attack_subset = attack_data[attack_data["Label"].isin(["Slowloris", "Slowread", "Slowheaders", "PortScan", "BruteForce"])]
        else:  # Mixed
            attack_subset = attack_data

        # Sampling logic
        total_size = int(len(df) * 0.005)
        n_attack = int(total_size * 0.8)
        n_benign = total_size - n_attack

        attack_sample = attack_subset.sample(n=min(n_attack, len(attack_subset)), random_state=i, replace=True)
        benign_sample = benign_data.sample(n=min(n_benign, len(benign_data)), random_state=i, replace=True)

        client_data = pd.concat([attack_sample, benign_sample], ignore_index=True)
        client_partitions[i] = client_data

        print(f"📡 Client {i}: {len(client_data)} samples ({n_attack} attack / {n_benign} benign)")

    print("✅ Non-IID partitioning complete.")
    return client_partitions


# ---------------------------------------------------------------------
# MAIN EXECUTION (for testing)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    combined = preprocess_and_combine()
    partitions = partition_data_non_iid(combined)

    total = sum(len(p) for p in partitions.values())
    print(f"\n📊 Total distributed samples: {total}")
