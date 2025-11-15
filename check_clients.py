import pandas as pd

for i in range(1, 4):
    df = pd.read_csv(f"data/clients/client_{i}.csv")
    print(f"client_{i}.csv → {df.shape}")
