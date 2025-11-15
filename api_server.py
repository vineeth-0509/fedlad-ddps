import os
import json
import shutil
import traceback
from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from model_utils import train_and_evaluate_final
from federated_training import run_federated_learning

app = FastAPI(title="FedLAD API Server")

# ---------------------------------------
#               CORS
# ---------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------
#          DIRECTORIES
# ---------------------------------------
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "data/uploads"
CLIENT_DIR = BASE_DIR / "data/clients"
LOGS_DIR = BASE_DIR / "logs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CLIENT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ===============================================================
#   Utility: Convert NumPy → Python types
# ===============================================================
def to_python(obj):
    """Convert numpy types into native python types."""
    if isinstance(obj, (np.integer, )):
        return int(obj)
    if isinstance(obj, (np.floating, )):
        return float(obj)
    if isinstance(obj, (np.ndarray, )):
        return obj.tolist()
    return obj

def clean_dict(d: dict):
    """Recursively convert all values inside dict."""
    clean = {}
    for k, v in d.items():
        if isinstance(v, dict):
            clean[k] = clean_dict(v)
        elif isinstance(v, list):
            clean[k] = [clean_dict(i) if isinstance(i, dict) else to_python(i) for i in v]
        else:
            clean[k] = to_python(v)
    return clean

# ===============================================================
#   Utility: Write NDJSON safely
# ===============================================================
def append_jsonl(path: Path, data: dict):
    """Write JSON object as a single NDJSON line."""
    cleaned = clean_dict(data)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(cleaned) + "\n")

# ===============================================================
#  POST /upload — dataset upload + split + centralized baseline
# ===============================================================
@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        file_path = UPLOAD_DIR / file.filename

        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if not file.filename.lower().endswith(".csv"):
            file_path.unlink(missing_ok=True)
            raise HTTPException(400, "Only CSV files are supported.")

        df = pd.read_csv(file_path)
        if df.empty:
            raise HTTPException(400, "Uploaded CSV is empty.")

        # ---- Create 3 clients ----
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        splits = np.array_split(df_shuffled, 3)

        for i, split_df in enumerate(splits, start=1):
            split_df.to_csv(CLIENT_DIR / f"client_{i}.csv", index=False)

        # ---- Centralized XGBoost baseline ----
        try:
            raw_metrics = train_and_evaluate_final(df, label_col="Label")
            metrics = clean_dict(raw_metrics)
        except Exception as e:
            traceback.print_exc()
            metrics = {"error": f"Training failed: {str(e)}"}

        # ---- Build severity summary (UI friendly) ----
        severity_summary = []
        total_rows = df.shape[0]

        for label in df["Label"].unique():
            count = int(df[df["Label"] == label].shape[0])
            confidence = round((count / total_rows) * 100, 2)
            severity_summary.append({
                "label": str(label),
                "confidence": float(confidence),
                "severity": float(round(confidence / 10, 2)),
            })

        # ---- Save centralized metrics ----
        append_jsonl(LOGS_DIR / "metrics_logs.json", {
            "strategy": "Centralized XGBoost",
            "metrics": metrics
        })

        # ---- Save evaluation summary ----
        append_jsonl(LOGS_DIR / "evaluation.json", {
            "dataset": file.filename,
            "metrics": metrics,
            "severity_data": severity_summary
        })

        return {
            "message": "Upload complete. Dataset split into 3 clients.",
            "summary": {
                "clients_created": 3,
                "metrics": metrics,
                "severity_data": severity_summary,
                "preview": df.head(5).to_dict(orient="records"),
            },
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Upload failed: {str(e)}")

# ===============================================================
#  POST /federated-train
# ===============================================================
@app.post("/federated-train")
def federated_train(rounds: int = 3, client_count: int = 3):
    try:
        # Ensure client files exist
        for i in range(1, client_count + 1):
            if not (CLIENT_DIR / f"client_{i}.csv").exists():
                raise HTTPException(400, f"Missing client_{i}.csv")

        results = run_federated_learning(client_count, rounds)

        # Append each round to NDJSON
        for r in results:
            append_jsonl(LOGS_DIR / "federated_logs.json", clean_dict(r))

        return {
            "message": "Federated training completed.",
            "rounds": len(results),
            "results": results
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Federated training failed: {e}")

# ===============================================================
# GET /logs — evaluation.json (NDJSON)
# ===============================================================
@app.get("/logs")
def get_logs():
    path = LOGS_DIR / "evaluation.json"
    if not path.exists():
        return []

    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except:
                    continue
    return items

# ===============================================================
# GET /metrics — latest centralized metric
# ===============================================================
@app.get("/metrics")
def get_metrics():
    path = LOGS_DIR / "metrics_logs.json"
    if not path.exists():
        return {}

    items = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    items.append(json.loads(line))
                except:
                    continue

    return items[-1] if items else {}

# ===============================================================
# GET /federated-results — NDJSON reader
# ===============================================================
@app.get("/federated-results")
def get_federated_results():
    path = LOGS_DIR / "federated_logs.json"
    if not path.exists():
        return []

    items = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    items.append(json.loads(line))
                except:
                    continue

    return items

# ===============================================================
# SSE Status Stream
# ===============================================================
@app.get("/training-status")
def training_status():
    def stream():
        steps = [
            "Preparing dataset...",
            "Preprocessing...",
            "Training...",
            "Aggregating...",
            "Final evaluation...",
            "Training complete 🎉",
        ]
        for msg in steps:
            yield f"data: {msg}\n\n"
            sleep(1)

    return StreamingResponse(stream(), media_type="text/event-stream")

# ===============================================================
# Root
# ===============================================================
@app.get("/")
def root():
    return {"message": "FedLAD API running!"}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting FedLAD FastAPI on http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
