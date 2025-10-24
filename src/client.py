# import flwr as fl
# import numpy as np
# from sklearn.model_selection import train_test_split
# from preprocessing import preprocess_local_dataset
# from data_utils import preprocess_and_combine, partition_data_non_iid

# # Load and preprocess client data with train/test split
# def load_client_data(client_id: int):
#     # Load combined and partitioned datasets
#     combined = preprocess_and_combine()
#     partitions = partition_data_non_iid(combined)

#     client_df = partitions[client_id]

#     # Split client data into train/test sets (80/20 split, stratified by label)
#     train_df, test_df = train_test_split(
#         client_df,
#         test_size=0.2,
#         random_state=42,
#         stratify=client_df["Label"]
#     )

#     # Preprocess train and test data
#     train_proc, le, scaler = preprocess_local_dataset(train_df, use_pca=True)
#     test_proc, _, _ = preprocess_local_dataset(test_df, use_pca=True)

#     # Separate features and labels
#     X_train = train_proc.drop("Label_Encoded", axis=1).values
#     y_train = train_proc["Label_Encoded"].values

#     X_test = test_proc.drop("Label_Encoded", axis=1).values
#     y_test = test_proc["Label_Encoded"].values

#     return X_train, y_train, X_test, y_test, le, scaler


# class FedLADClient(fl.client.NumPyClient):
#     def __init__(self, client_id: int):
#         print(f"🚀 Initializing Local Controller {client_id}")
#         self.client_id = client_id

#         # Load client data
#         self.X_train, self.y_train, self.X_test, self.y_test, self.le, self.scaler = load_client_data(client_id)

#         # Initialize your model here (example: XGBoost, sklearn model, etc.)
#         import xgboost as xgb
#         self.model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
#         self.model.fit(self.X_train, self.y_train)

#     def get_parameters(self, config):
#         # Return model parameters as a list of numpy arrays
#         return [self.model.get_booster().get_dump()]

#     def set_parameters(self, parameters):
#         # Set model parameters from the server
#         # You need to implement this if your model supports setting parameters directly
#         pass

#     def fit(self, parameters, config):
#         # Update model with new parameters and train on local data
#         # For example, you might want to update your model here with parameters

#         # Fit model on local data
#         self.model.fit(self.X_train, self.y_train)
#         return self.get_parameters(), len(self.X_train), {}

#     def evaluate(self, parameters, config):
#         # Evaluate model on local test data
#         preds = self.model.predict(self.X_test)
#         accuracy = np.mean(preds == self.y_test)
#         return float(accuracy), len(self.X_test), {"accuracy": float(accuracy)}


# if __name__ == "__main__":
#     client = FedLADClient(client_id=0)
#     fl.client.start_numpy_client(server_address="localhost:8080", client=client)




# import flwr as fl
# import numpy as np
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from preprocessing import preprocess_local_dataset
# from data_utils import preprocess_and_combine, partition_data_non_iid
# from sklearn.metrics import accuracy_score, log_loss, f1_score

# def load_client_data(client_id: int):
#     combined = preprocess_and_combine()
#     partitions = partition_data_non_iid(combined)

#     client_df = partitions[client_id]

#     train_df, test_df = train_test_split(
#         client_df,
#         test_size=0.2,
#         random_state=42,
#         stratify=client_df["Label"]
#     )

#     train_proc, le, scaler = preprocess_local_dataset(train_df, use_pca=True)
#     test_proc, _, _ = preprocess_local_dataset(test_df, use_pca=True)

#     X_train = train_proc.drop("Label_Encoded", axis=1).values
#     y_train = train_proc["Label_Encoded"].values

#     X_test = test_proc.drop("Label_Encoded", axis=1).values
#     y_test = test_proc["Label_Encoded"].values

#     return X_train, y_train, X_test, y_test, le, scaler


# class FedLADClient(fl.client.NumPyClient):
#     def __init__(self, client_id: int):
#         print(f"🚀 Initializing Local Controller {client_id}")
#         self.client_id = client_id

#         # Load client data
#         self.X_train, self.y_train, self.X_test, self.y_test, self.le, self.scaler = load_client_data(client_id)

#         # Use LogisticRegression for simplicity and compatibility
#         self.model = LogisticRegression(max_iter=1000)
#         self.model.fit(self.X_train, self.y_train)

#     def get_parameters(self, config):
#         # Return model coefficients and intercept as list of numpy arrays
#         return [self.model.coef_, self.model.intercept_]

#     def set_parameters(self, parameters, config):
#         # Set model coefficients and intercept from parameters list
#         self.model.coef_ = parameters[0]
#         self.model.intercept_ = parameters[1]

#     def fit(self, parameters, config):
#         # parameters, config signature supported
#         # Set global parameters (if provided)
#         if parameters is not None:
#             self.set_parameters(parameters, config)

#         # Fit locally
#         self.model.fit(self.X_train, self.y_train)

#         # Compute metrics on local training set (or a small val split)
#         preds_proba = self.model.predict_proba(self.X_train)
#         preds = np.argmax(preds_proba, axis=1) if preds_proba.ndim > 1 else (preds_proba > 0.5).astype(int)
#         acc = float(accuracy_score(self.y_train, preds))
#         try:
#             ll = float(log_loss(self.y_train, preds_proba, labels=np.unique(self.y_train)))
#         except Exception:
#             ll = 0.0

#         # return parameters, number of examples, metrics dict
#         return self.get_parameters(config), len(self.X_train), {"accuracy": acc, "loss": ll}

#     def evaluate(self, parameters, config):
#         # Set params then evaluate on local test set
#         if parameters is not None:
#             self.set_parameters(parameters, config)

#         preds_proba = self.model.predict_proba(self.X_test)
#         preds = np.argmax(preds_proba, axis=1) if preds_proba.ndim > 1 else (preds_proba > 0.5).astype(int)
#         acc = float(accuracy_score(self.y_test, preds))
#         f1 = float(f1_score(self.y_test, preds, average="weighted"))

#         # compute test loss safely
#         try:
#             ll = float(log_loss(self.y_test, preds_proba, labels=np.unique(self.y_test)))
#         except Exception:
#             ll = 0.0

#         return float(ll), len(self.X_test), {"accuracy": acc, "f1": f1}


# # if __name__ == "__main__":
# #     client = FedLADClient(client_id=0)
# #     fl.client.start_numpy_client(server_address="localhost:8080", client=client)

# if __name__ == "__main__":
#     import sys
#     client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
#     client = FedLADClient(client_id=client_id)
#     fl.client.start_client(server_address="localhost:8080", client=client.to_client())


# import flwr as fl
# import numpy as np
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from preprocessing import preprocess_local_dataset
# from data_utils import preprocess_and_combine, partition_data_non_iid
# from sklearn.metrics import accuracy_score, log_loss, f1_score


# # Load and preprocess client data
# def load_client_data(client_id: int):
#     combined = preprocess_and_combine()
#     partitions = partition_data_non_iid(combined)

#     client_df = partitions[client_id]

#     train_df, test_df = train_test_split(
#         client_df,
#         test_size=0.2,
#         random_state=42,
#         stratify=client_df["Label"]
#     )

#     train_proc, le, scaler = preprocess_local_dataset(train_df, use_pca=True)
#     test_proc, _, _ = preprocess_local_dataset(test_df, use_pca=True)

#     X_train = train_proc.drop("Label_Encoded", axis=1).values
#     y_train = train_proc["Label_Encoded"].values

#     X_test = test_proc.drop("Label_Encoded", axis=1).values
#     y_test = test_proc["Label_Encoded"].values

#     return X_train, y_train, X_test, y_test, le, scaler


# class FedLADClient(fl.client.NumPyClient):
#     def __init__(self, client_id: int):
#         print(f"🚀 Initializing Local Controller {client_id}")
#         self.client_id = client_id

#         # Load client data
#         self.X_train, self.y_train, self.X_test, self.y_test, self.le, self.scaler = load_client_data(client_id)

#         # Initialize a LogisticRegression model
#         self.model = LogisticRegression(max_iter=1000)
#         self.model.fit(self.X_train, self.y_train)

#     def get_parameters(self, config):
#         """Return model coefficients and intercept as list of numpy arrays."""
#         return [self.model.coef_, self.model.intercept_]

#     def set_parameters(self, parameters, config):
#         """Set model coefficients and intercept from parameters list."""
#         self.model.coef_ = parameters[0]
#         self.model.intercept_ = parameters[1]

#     def fit(self, parameters, config):
#         """Update model with new parameters and train on local data."""
#         if parameters is not None:
#             self.set_parameters(parameters, config)

#         # Fit locally
#         self.model.fit(self.X_train, self.y_train)

#         # Compute metrics on local training set (or a small validation split)
#         preds_proba = self.model.predict_proba(self.X_train)
#         preds = np.argmax(preds_proba, axis=1) if preds_proba.ndim > 1 else (preds_proba > 0.5).astype(int)
#         acc = float(accuracy_score(self.y_train, preds))
#         try:
#             ll = float(log_loss(self.y_train, preds_proba, labels=np.unique(self.y_train)))
#         except Exception:
#             ll = 0.0

#         # Return updated parameters, number of training examples, and metrics
#         return self.get_parameters(config), len(self.X_train), {"accuracy": acc, "loss": ll}

#     def evaluate(self, parameters, config):
#         """Set params then evaluate on local test set."""
#         if parameters is not None:
#             self.set_parameters(parameters, config)

#         preds_proba = self.model.predict_proba(self.X_test)
#         preds = np.argmax(preds_proba, axis=1) if preds_proba.ndim > 1 else (preds_proba > 0.5).astype(int)
#         acc = float(accuracy_score(self.y_test, preds))
#         f1 = float(f1_score(self.y_test, preds, average="weighted"))

#         # Compute test loss safely
#         try:
#             ll = float(log_loss(self.y_test, preds_proba, labels=np.unique(self.y_test)))
#         except Exception:
#             ll = 0.0

#         # Return loss, number of test examples, and additional metrics
#         return float(ll), len(self.X_test), {"accuracy": acc, "f1": f1}


# if __name__ == "__main__":
#     import sys
#     client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
#     client = FedLADClient(client_id=client_id)
#     fl.client.start_client(server_address="localhost:8080", client=client.to_client())


import flwr as fl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from preprocessing import preprocess_local_dataset
from data_utils import preprocess_and_combine, partition_data_non_iid
from sklearn.metrics import accuracy_score, log_loss, f1_score


# Load and preprocess client data
def load_client_data(client_id: int):
    combined = preprocess_and_combine()
    partitions = partition_data_non_iid(combined)

    client_df = partitions[client_id]

    train_df, test_df = train_test_split(
        client_df,
        test_size=0.2,
        random_state=42,
        stratify=client_df["Label"]
    )

    train_proc, le, scaler = preprocess_local_dataset(train_df, use_pca=True)
    test_proc, _, _ = preprocess_local_dataset(test_df, use_pca=True)

    X_train = train_proc.drop("Label_Encoded", axis=1).values
    y_train = train_proc["Label_Encoded"].values

    X_test = test_proc.drop("Label_Encoded", axis=1).values
    y_test = test_proc["Label_Encoded"].values

    return X_train, y_train, X_test, y_test, le, scaler


class FedLADClient(fl.client.NumPyClient):
    def __init__(self, client_id: int):
        print(f"🚀 Initializing Local Controller {client_id}")
        self.client_id = client_id

        # Load client data
        self.X_train, self.y_train, self.X_test, self.y_test, self.le, self.scaler = load_client_data(client_id)

        # Initialize a LogisticRegression model
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(self.X_train, self.y_train)

    def get_parameters(self, config):
        """Return model coefficients and intercept as list of numpy arrays."""
        return [self.model.coef_, self.model.intercept_]

    def set_parameters(self, parameters, config):
        """Set model coefficients and intercept from parameters list."""
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]

    def fit(self, parameters, config):
        """Update model with new parameters and train on local data."""
        if parameters is not None:
            self.set_parameters(parameters, config)

        # Fit locally
        self.model.fit(self.X_train, self.y_train)

        # Compute metrics on local training set (or a small validation split)
        preds_proba = self.model.predict_proba(self.X_train)
        preds = np.argmax(preds_proba, axis=1) if preds_proba.ndim > 1 else (preds_proba > 0.5).astype(int)
        acc = float(accuracy_score(self.y_train, preds))
        try:
            ll = float(log_loss(self.y_train, preds_proba, labels=np.unique(self.y_train)))
        except Exception:
            ll = 0.0

        # Return updated parameters, number of training examples, and metrics
        return self.get_parameters(config), len(self.X_train), {"accuracy": acc, "loss": ll}

    def evaluate(self, parameters, config):
        """Set params then evaluate on local test set."""
        if parameters is not None:
            self.set_parameters(parameters, config)

        preds_proba = self.model.predict_proba(self.X_test)
        preds = np.argmax(preds_proba, axis=1) if preds_proba.ndim > 1 else (preds_proba > 0.5).astype(int)
        acc = float(accuracy_score(self.y_test, preds))
        f1 = float(f1_score(self.y_test, preds, average="weighted"))

        # Compute test loss safely
        try:
            ll = float(log_loss(self.y_test, preds_proba, labels=np.unique(self.y_test)))
        except Exception:
            ll = 0.0

        # Return loss, number of test examples, and additional metrics
        return float(ll), len(self.X_test), {"accuracy": acc, "f1": f1}


if __name__ == "__main__":
    import sys
    client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    client = FedLADClient(client_id=client_id)
    fl.client.start_client(server_address="localhost:8080", client=client.to_client())
