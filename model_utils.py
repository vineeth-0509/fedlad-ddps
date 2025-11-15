# # model_utils.py
# import pandas as pd
# import numpy as np
# import xgboost as xgb
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
# # Import all required metrics
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# # ... (preprocess_local_dataset function remains the same) ...
# # # 🧩 Function: preprocess uploaded dataset
# def preprocess_local_dataset(df: pd.DataFrame):
#     """
#     Cleans, encodes categorical columns, and scales numeric columns.
#     Ensures the target (last column) is NOT scaled.
#     Returns the processed DataFrame, LabelEncoder, and Scaler.
#     """
#     df = df.dropna().reset_index(drop=True)

#     le = LabelEncoder()
#     scaler = StandardScaler()

#     # --- Identify target column (last column) ---
#     target_col = df.columns[-1]

#     # --- Encode categorical columns (excluding target if categorical) ---
#     for col in df.select_dtypes(include=["object"]).columns:
#         if col != target_col:
#             df[col] = le.fit_transform(df[col])

#     # --- Scale numeric columns except the target ---
#     numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
#     for col in numeric_cols:
#         if col != target_col:
#             df[col] = scaler.fit_transform(df[[col]])

#     # --- Ensure target is integer class labels (0/1 if binary) ---
#     if df[target_col].dtype != int and df[target_col].dtype != bool:
#         unique_vals = df[target_col].unique()
#         # If target values are text or continuous, encode to 0..n
#         if len(unique_vals) > 10 or np.any(df[target_col].apply(lambda x: isinstance(x, float))):
#             # Continuous target — convert to binary based on median
#             median_val = df[target_col].median()
#             df[target_col] = (df[target_col] > median_val).astype(int)
#         else:
#             df[target_col] = le.fit_transform(df[target_col])

#     return df, le, scaler


# # 🚀 Function: train XGBoost model and evaluate (REVISED)
# # def train_and_evaluate(df: pd.DataFrame):
# #     """
# #     Trains an XGBoost model, returns metrics, and the average attack probability.
# #     Assumes the last column in df is the target variable (0=Benign, 1=Attack).
# #     """
# #     X = df.iloc[:, :-1]
# #     y = df.iloc[:, -1]

# #     # Split dataset
# #     X_train, X_test, y_test = X.iloc[0:10], X.iloc[10:20], y.iloc[10:20] # Placeholder split to be safe
# #     X_train, X_test, y_train, y_test = train_test_split(
# #         X, y, test_size=0.2, random_state=42
# #     )

# #     # Train XGBoost classifier
# #     model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
# #     model.fit(X_train, y_train)

# #     # 1. Predictions for metrics
# #     y_pred = model.predict(X_test)

# #     # 2. Prediction Probabilities for Confidence (Probability of being class 1/Attack)
# #     # This is the actual confidence you want to display on the dashboard!
# #     y_proba = model.predict_proba(X_test)
# #     attack_proba = y_proba[:, 1] # Probability of being the positive class (Attack, often label 1)
    
# #     # Calculate the average attack confidence for the whole test set
# #     avg_attack_confidence = np.mean(attack_proba) 

# #     # Metrics
# #     accuracy = accuracy_score(y_test, y_pred)
# #     f1 = f1_score(y_test, y_pred, average="weighted")
# #     # Added Precision and Recall
# #     precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
# #     recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)

# #     metrics = {
# #         "strategy": "XGBoost",
# #         "accuracy": float(round(accuracy, 4)),
# #         "f1_score": float(round(f1, 4)),
# #         "precision": float(round(precision, 4)), # Now correctly calculated
# #         "recall": float(round(recall, 4)),       # Now correctly calculated
# #         # The key piece of data for the dashboard confidence
# #         "avg_attack_confidence": float(round(avg_attack_confidence, 4)),
# #     }

# #     return metrics


# # 🚀 train_and_evaluate (updated for multi-threat severity)
# def train_and_evaluate(df: pd.DataFrame):
#     """
#     Trains an XGBoost model, returns metrics and attack probabilities for multiple attack types.
#     Assumes the last column is target (0=Benign, 1=Attack).
#     """
#     X = df.iloc[:, :-1]
#     y = df.iloc[:, -1]

#     # Split dataset
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     # Train XGBoost classifier
#     model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
#     model.fit(X_train, y_train)

#     # Predictions
#     y_pred = model.predict(X_test)
#     y_proba = model.predict_proba(X_test)
    
#     # Probability of Attack (positive class)
#     attack_proba = y_proba[:, 1]
#     avg_attack_confidence = np.mean(attack_proba)

#     # Split attack confidence into multiple attack types
#     # Example: DDoS=50%, UDP Flood=30%, PortScan=15%, WebAttack=5%
#     # You can make this based on actual model probabilities if multiclass
#     severity_conf_split = np.array([0.5, 0.3, 0.15, 0.05])
#     attack_types_confidence = (avg_attack_confidence * severity_conf_split * 100).round(2)

#     # Metrics
#     accuracy = accuracy_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred, average="weighted")
#     precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
#     recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)

#     metrics = {
#         "strategy": "XGBoost",
#         "accuracy": float(round(accuracy, 4)),
#         "f1_score": float(round(f1, 4)),
#         "precision": float(round(precision, 4)),
#         "recall": float(round(recall, 4)),
#         "avg_attack_confidence": float(round(avg_attack_confidence, 4)),
#         "attack_types_confidence": attack_types_confidence.tolist(),  # NEW
#     }

#     return metrics


# model_utils.py  — FINAL PRODUCTION VERSION
# model_utils.py — FINAL FIXED VERSION
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from imblearn.over_sampling import RandomOverSampler


# -------------------------------------------------
# CLEAN EXTREME VALUES
# -------------------------------------------------
def sanitize_dataframe(df: pd.DataFrame):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    # remove extremely large values
    df = df.apply(lambda col: col.map(
        lambda x: np.nan if isinstance(x, (int, float)) and abs(x) > 1e12 else x
    ))

    df = df.dropna()
    return df


# -------------------------------------------------
# SAFE PREPROCESSING + STRATIFIED SPLITTING
# -------------------------------------------------
def safe_preprocess_and_split(
    df: pd.DataFrame,
    label_col: str,
    balance: bool = True,
    test_size=0.2,
    random_state=42
):
    df = sanitize_dataframe(df.copy())

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found!")

    # remove garbage columns
    drop_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    df = df.drop(columns=drop_cols, errors="ignore")

    # separate features & label
    y_raw = df[label_col].astype(str)
    X = df.drop(columns=[label_col])

    # identify types
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # stratified split (NO leakage)
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X, y_raw,
        test_size=test_size,
        stratify=y_raw,
        random_state=random_state
    )

    # encode target
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)

    # -------------------------
    # Encode categorical safely
    # -------------------------
    for col in cat_cols:
        # convert all to string
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

        le_col = LabelEncoder()
        le_col.fit(X_train[col])

        def safe_transform(val):
            val = str(val)
            if val in le_col.classes_:
                return int(le_col.transform([val])[0])
            else:
                return -1

        X_train[col] = X_train[col].map(safe_transform)
        X_test[col] = X_test[col].map(safe_transform)

    # scale numerical columns
    scaler = StandardScaler()
    if num_cols:
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])

    # handle imbalance
    if balance:
        ros = RandomOverSampler(random_state=random_state)
        X_train, y_train = ros.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test, le


# -------------------------------------------------
# TRAIN + EVALUATE MODEL SAFELY
# -------------------------------------------------
def train_and_evaluate_final(df: pd.DataFrame, label_col: str = "Label"):
    df = sanitize_dataframe(df)

    X_train, X_test, y_train, y_test, le = safe_preprocess_and_split(
        df,
        label_col=label_col,
        balance=True
    )

    num_classes = len(np.unique(y_train))

    # XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob" if num_classes > 2 else "binary:logistic",
        eval_metric="mlogloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    # predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # class confidence
    class_confidences = {
        le.inverse_transform([i])[0]: float(np.mean(y_proba[:, i]))
        for i in range(num_classes)
    }

    # metrics
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 6),
        "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 6),
        "recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 6),
        "f1_score": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 6),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "class_confidence": class_confidences,
        "classes": le.inverse_transform(np.arange(num_classes)).tolist(),
        "num_test_samples": len(y_test)
    }

    return metrics
