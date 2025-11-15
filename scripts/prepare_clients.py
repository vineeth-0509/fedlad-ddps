import os
import pandas as pd
import numpy as np

CLIENT_DIR = "data/clients"

def clean_and_align_clients():
    files = [
        f"{CLIENT_DIR}/client_1.csv",
        f"{CLIENT_DIR}/client_2.csv",
        f"{CLIENT_DIR}/client_3.csv",
    ]

    dfs = []

    # Read all
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        dfs.append(df)

    # Find common columns
    common_cols = set(dfs[0].columns)
    for df in dfs[1:]:
        common_cols &= set(df.columns)

    common_cols = list(common_cols)

    print("COMMON COLUMNS:", len(common_cols))

    # Keep only common columns
    for i, f in enumerate(files):
        df = dfs[i][common_cols].copy()

        # Fill NaN
        df = df.fillna(0)

        # Save back
        df.to_csv(f, index=False)
        print("Saved cleaned:", f)

if __name__ == "__main__":
    clean_and_align_clients()
