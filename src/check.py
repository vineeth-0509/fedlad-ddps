import os

for fname in ["CICIDS2017.csv", "CICDDOS2019.csv", "InSDN.csv"]:
    path = os.path.join("data", fname)
    print(f"{path}: {'Found' if os.path.isfile(path) else 'Not Found'}")
