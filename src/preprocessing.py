# we call preprocessing logic here:
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA
from typing import Tuple, Dict

# ---------------------------------------------------------------------
# 🌍 GLOBAL FEATURE LIST (26 high-importance features)
# ---------------------------------------------------------------------
FEATURES = [
    "Flow_Duration", "Total_Fwd_Packets", "Total_Bwd_Packets", "Total_Length_of_Fwd_Packets",
    "Total_Length_of_Bwd_Packets", "Fwd_Packet_Length_Max", "Bwd_Packet_Length_Max",
    "Flow_Bytes_s", "Flow_Packets_s", "Flow_IAT_Mean", "Fwd_IAT_Mean", "Bwd_IAT_Mean",
    "Fwd_Packets_s", "Bwd_Packets_s", "Min_Packet_Length", "Max_Packet_Length",
    "Packet_Length_Mean", "Packet_Length_Std", "Avg_Fwd_Segment_Size", "Avg_Bwd_Segment_Size",
    "Subflow_Fwd_Bytes", "Subflow_Bwd_Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "Active_Mean", "Idle_Mean"
]


# ---------------------------------------------------------------------
# 1️⃣ FEATURE SELECTION / DIMENSION REDUCTION
# ---------------------------------------------------------------------
def select_and_scale_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """Detect numeric features, clean safely, and standardize."""
    df = df.copy()

    # Replace inf/-inf with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Detect numeric columns automatically (exclude label columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in ["Label", "Label_Encoded"]:
        if col in numeric_cols:
            numeric_cols.remove(col)

    if not numeric_cols:
        raise ValueError("❌ No numeric columns found in dataset after cleaning.")

    # Fill NaNs with mean values instead of dropping rows
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    print(f"🧩 Numeric columns detected: {len(numeric_cols)}")

    # Standardize numeric features
    X = df[numeric_cols].astype(np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_cols, index=df.index)
    return X_scaled_df, scaler


# ---------------------------------------------------------------------
# 2️⃣ LABEL ENCODING
# ---------------------------------------------------------------------
def encode_labels(df: pd.DataFrame) -> Tuple[pd.Series, LabelEncoder]:
    """Encode categorical attack labels into numeric values."""
    if "Label" not in df.columns:
        raise ValueError("❌ 'Label' column missing in dataset.")
    le = LabelEncoder()
    y = le.fit_transform(df["Label"].astype(str))
    return pd.Series(y, index=df.index, name="Label_Encoded"), le


# ---------------------------------------------------------------------
# 3️⃣ CLASS BALANCING
# ---------------------------------------------------------------------
def balance_classes(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Balance dataset using random oversampling."""
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y)
    return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name="Label_Encoded")


# ---------------------------------------------------------------------
# 4️⃣ PCA (OPTIONAL)
# ---------------------------------------------------------------------
def apply_pca(X: pd.DataFrame, n_components: int = 20) -> Tuple[pd.DataFrame, PCA]:
    """Reduce feature dimensionality using PCA."""
    if X.shape[1] < n_components:
        n_components = X.shape[1]  # prevent PCA from failing
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X)
    cols = [f"PC{i+1}" for i in range(n_components)]
    return pd.DataFrame(X_reduced, columns=cols), pca


# ---------------------------------------------------------------------
# 5️⃣ FULL PIPELINE
# ---------------------------------------------------------------------
def preprocess_local_dataset(df: pd.DataFrame, use_pca: bool = False) -> Tuple[pd.DataFrame, LabelEncoder, StandardScaler]:
    """
    Full preprocessing pipeline:
    - Clean and scale numeric features
    - Encode categorical labels
    - Balance classes
    - Optionally reduce dimensionality with PCA
    """
    print("⚙️  Preprocessing local dataset...")

    # Step 1: Feature scaling
    X_scaled, scaler = select_and_scale_features(df)

    # Step 2: Label encoding
    y_encoded, le = encode_labels(df)

    # Step 3: Class balancing
    X_bal, y_bal = balance_classes(X_scaled, y_encoded)

    # Step 4: Optional PCA for efficiency
    if use_pca:
        X_bal, _ = apply_pca(X_bal)

    # Combine final dataset
    processed_df = pd.concat([X_bal, y_bal], axis=1)
    print(f"✅ Preprocessed dataset shape: {processed_df.shape}")
    return processed_df, le, scaler


# ---------------------------------------------------------------------
# 🔍 TESTING ENTRY POINT
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from data_utils import preprocess_and_combine, partition_data_non_iid

    # Combine datasets
    combined = preprocess_and_combine()

    # Partition into Non-IID subsets
    partitions = partition_data_non_iid(combined)

    # Select one client dataset
    client_df = partitions[0]  # Client 0
    if isinstance(client_df, tuple):
        # if partition_data_non_iid returns (train, test)
        client_df = pd.concat(client_df)

    # Run preprocessing
    processed_df, le, scaler = preprocess_local_dataset(client_df, use_pca=True)
    print(processed_df.head())
