# federated_training.py
"""
Federated training simulation utilities for FedLAD.

High-level flow:
- load_client_csvs(): read client CSVs from data/clients/client_{i}.csv
- run_federated_learning(): for R rounds, train a local XGBoost for each client
  using `train_local_xgb()`, then evaluate the ensemble via `evaluate_ensemble()`.
- train_local_xgb(): uses safe_preprocess_and_split() from model_utils to avoid
  data leakage and class-imbalance issues; returns trained model + metadata.
- predict_proba_on(): given a trained model + feature list, prepare test features
  and return probability-of-attack vector.
- evaluate_ensemble(): aggregate client model probabilities (mean), convert true
  labels to binary safely, then compute binary metrics (accuracy, precision, recall, f1).

Important notes:
- This file expects `model_utils.safe_preprocess_and_split` and `model_utils.sanitize_dataframe`.
- This is a simulation: "federated" here means independent local training per client
  and server-side averaging of probabilities (ensemble). It's not a parameter-averaging FL algorithm.
"""

import os
import json
import time
import typing as t
from pathlib import Path

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# model_utils must expose:
# - safe_preprocess_and_split(df, label_col, balance, test_size, random_state)
# - sanitize_dataframe(df) to clean infinities, NaNs, overflow values etc.
from model_utils import safe_preprocess_and_split, sanitize_dataframe

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CLIENT_DIR = DATA_DIR / "clients"
LOGS_DIR = ROOT / "logs"
os.makedirs(CLIENT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


def load_client_csvs(client_count: int = 3) -> dict:
    """
    Load CSVs from data/clients/client_{i}.csv for 1..client_count.
    Returns dict mapping client_id -> DataFrame.
    """
    clients = {}
    for i in range(1, client_count + 1):
        p = CLIENT_DIR / f"client_{i}.csv"
        if p.exists():
            df = pd.read_csv(p)
            clients[f"client_{i}"] = df
        else:
            print(f"[federated] warning: missing {p}")
    return clients


def train_local_xgb(df: pd.DataFrame, label_col: str = "Label", seed: int = 42):
    """
    Train a local XGBoost model for a client's dataframe.
    - Uses safe_preprocess_and_split() (fits encoders & scaler on client's train split).
    - Returns (model, label_encoder_for_target, feature_columns)
    NOTE: We intentionally return only label encoder (for interpreting classes) and
    feature_cols. Per-client internal encoders/scalers are not returned here.
    """
    X_train, X_test, y_train, y_test, le = safe_preprocess_and_split(
        df,
        label_col=label_col,
        balance=True,
        test_size=0.2,
        random_state=seed
    )

    feature_cols = X_train.columns.tolist()

    model = xgb.XGBClassifier(
        n_estimators=60,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.7,
        colsample_bytree=0.7,
        tree_method="hist",   # faster on large data
        eval_metric="logloss",
        random_state=seed,
        verbosity=0
    )

    model.fit(X_train, y_train)
    return model, le, feature_cols


def predict_proba_on(model, le, feature_cols, df, label_col="Label"):
    """
    Prepare the given `df` to compute prediction probabilities with `model`.
    This function:
      - runs safe_preprocess_and_split(df, balance=False) to get a consistent X_test
        split out of df (we don't resample or balance here).
      - aligns X_test columns to model's feature_cols (adding missing columns=0).
      - returns a 1D numpy array of attack probabilities (for the positive class).
    Notes:
      - Because safe_preprocess_and_split does a deterministic stratified split
        with a fixed random_state default, repeated calls on the same df produce
        consistent X_test sizes (important for averaging).
      - If model.predict_proba returns shape (n,2) we return column 1,
        otherwise a fallback computed probability is returned.
    """
    # Use a deterministic split on the df to get a test portion - balance=False avoids resampling.
    _, X_test, _, y_test, _ = safe_preprocess_and_split(
        df, label_col=label_col, balance=False, test_size=0.2, random_state=42
    )

    # Ensure features align: add missing columns with zeros and keep the feature ordering
    missing_cols = [c for c in feature_cols if c not in X_test.columns]
    for c in missing_cols:
        X_test[c] = 0
    # Reorder columns - if extra columns exist in X_test they will be ignored by indexing
    X_test = X_test.reindex(columns=feature_cols, fill_value=0)

    proba = model.predict_proba(X_test)

    # binary vs multiclass shape handling
    if proba.shape[1] == 2:
        return proba[:, 1]
    else:
        # multiclass: treat "not class 0" as attack-probability proxy
        return (1 - proba[:, 0])


def _safe_convert_labels_to_binary(y_series: pd.Series) -> pd.Series:
    """
    Convert a label Series to binary 0/1 safely:
      - If numeric dtype:
          * If only two unique values -> treat them as 0/1 (astype int)
          * If more than two -> treat the mode (most frequent) as BENIGN (0),
            everything else becomes 1.
      - If string/object dtype:
          * Upper-case and check for known benign keywords ("BENIGN", "NORMAL", "GOOD")
            If contains any of those terms -> 0 else 1.
    Returns a Series of ints (0/1).
    """
    # Defensive: ensure y_series is a pandas Series
    if not isinstance(y_series, pd.Series):
        y_series = pd.Series(y_series)

    if y_series.isnull().any():
        # dropna guard — caller should ensure alignment with probabilities; raising helps debugging
        y_series = y_series.fillna(method="ffill").fillna(method="bfill").fillna(0)

    # Numeric labels
    if pd.api.types.is_numeric_dtype(y_series.dtype):
        uniques = y_series.unique()
        if len(uniques) <= 2:
            return y_series.astype(int)
        else:
            # Multi-class numeric: assume the most-common label is benign (mode)
            benign_class = y_series.mode().iloc[0]
            return (y_series != benign_class).astype(int)

    # Object / string labels
    y_up = y_series.astype(str).str.upper()
    benign_keywords = ["BENIGN", "NORMAL", "GOOD", "LEGIT"]

    def map_label(v: str) -> int:
        for k in benign_keywords:
            if k in v:
                return 0
        return 1

    return y_up.apply(map_label).astype(int)


def evaluate_ensemble(
    clients_models: t.List[t.Tuple[str, xgb.XGBClassifier, t.List[str], object]],
    test_df: pd.DataFrame,
    label_col: str = "Label"
) -> dict:
    """
    Given list of (client_id, model, feature_cols, label_encoder), compute ensembled
    probability predictions on test_df and return binary metrics.
    Steps:
      - For each client, compute per-sample attack probability vector (predict_proba_on)
      - Stack & average probabilities across clients (voting by mean-proba)
      - Convert ground-truth labels to binary safely (using _safe_convert_labels_to_binary)
      - Compute accuracy, precision, recall, f1-score (binary)
    Returns a dict of metrics, or {} if not computable.
    """
    if test_df is None or test_df.empty:
        return {}

    probas = []
    # For debugging, keep track of expected length
    expected_len = None

    for client_id, model, feature_cols, le in clients_models:
        try:
            p = predict_proba_on(model, le, feature_cols, test_df, label_col)
            if expected_len is None:
                expected_len = len(p)
            # basic length check — if mismatched lengths occur, we skip this client's predictions
            if expected_len is not None and len(p) != expected_len:
                print(f"[federated] warning: client {client_id} proba length {len(p)} != expected {expected_len}, skipping")
                continue
            probas.append(p)
        except Exception as e:
            print(f"[federated] predict error for {client_id}: {e}")

    if not probas:
        return {}

    # Average probabilities across stacked arrays: shape -> (n_samples,)
    avg_proba = np.mean(np.vstack(probas), axis=0)

    # Build y_true from the same test_df slice used by predict_proba_on.
    # We use safe_preprocess_and_split to produce the deterministic test split (same as predict_proba_on).
    # Note: caller must ensure predict_proba_on used same split params (test_size=0.2, random_state=42).
    _, X_test_dummy, _, y_test_dummy, _ = safe_preprocess_and_split(
        test_df, label_col=label_col, balance=False, test_size=0.2, random_state=42
    )

    # <-- IMPORTANT FIX: ensure y_test_dummy is a pandas Series here
    y_true_raw = pd.Series(y_test_dummy).reset_index(drop=True)

    # Convert to binary using safe method
    y_true = _safe_convert_labels_to_binary(y_true_raw)

    # Map probabilities -> binary predictions (threshold 0.5)
    y_pred = (avg_proba >= 0.5).astype(int)

    # Defensive length check
    if len(y_pred) != len(y_true):
        print(f"[federated] length mismatch: y_pred {len(y_pred)} vs y_true {len(y_true)}")
        # try to truncate/align by min length
        min_len = min(len(y_pred), len(y_true))
        y_pred = y_pred[:min_len]
        y_true = y_true.iloc[:min_len]

    # Compute metrics (binary)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "accuracy": float(round(acc, 6)),
        "precision": float(round(prec, 6)),
        "recall": float(round(rec, 6)),
        "f1_score": float(round(f1, 6)),
        "num_test": int(len(y_true))
    }


def run_federated_learning(client_count: int = 3, rounds: int = 3, label_col: str = "Label"):
    """
    Simulate federated training:
      - load client CSVs
      - optionally sanitize each client's DataFrame
      - each round: each client trains a local model
      - server evaluates an ensemble by averaging probabilities
      - logs per-round metrics to logs/federated_logs.json (ndjson)
    Returns list of per-round metrics (entries).
    """
    clients = load_client_csvs(client_count)
    if not clients:
        raise RuntimeError("No client CSVs found in data/clients/")

    # Build a global test set by sampling 20% from each client
    test_frames = []
    for cid, df in clients.items():
        if df.shape[0] > 10:
            test_frames.append(df.sample(frac=0.2, random_state=42))
    global_test = pd.concat(test_frames, ignore_index=True) if test_frames else None

    round_results = []
    for r in range(1, rounds + 1):
        round_start = time.time()
        print(f"[federated] Round {r} — training {len(clients)} clients")
        clients_models = []

        for idx, (cid, df) in enumerate(clients.items(), start=1):
            try:
                # sanitize: remove infs/huge values/nans etc
                df_clean = sanitize_dataframe(df.copy())
                model, le, feature_cols = train_local_xgb(df_clean, label_col=label_col, seed=42 + r + idx)
                clients_models.append((cid, model, feature_cols, le))
            except Exception as e:
                print(f"[federated] client {cid} train failed: {e}")

        # Evaluate ensemble on global_test (if available)
        metrics = {}
        try:
            metrics = evaluate_ensemble(clients_models, global_test, label_col=label_col) if global_test is not None else {}
        except Exception as e:
            print(f"[federated] ensemble evaluation failed: {e}")
            metrics = {}

        elapsed = round(time.time() - round_start, 2)
        entry = {
            "round": r,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": metrics,
            "num_clients": len(clients_models),
            "elapsed_sec": elapsed,
        }
        round_results.append(entry)

        # append to logs file (ndjson)
        with open(LOGS_DIR / "federated_logs.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        print(f"[federated] Round {r} metrics: {metrics}")

    return round_results


# quick test entrypoint
if __name__ == "__main__":
    res = run_federated_learning(client_count=3, rounds=3)
    print("Done. rounds:", len(res))
