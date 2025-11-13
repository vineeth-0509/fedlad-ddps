# import os, sys, json, shutil, traceback
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# import pandas as pd
# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import StreamingResponse
# from datetime import datetime
# from time import sleep
# from typing import Generator

# # Import your ML utilities
# # NOTE: The imported 'train_and_evaluate' must return a dict with 'accuracy', 'f1_score', 'precision', and 'recall' keys.
# from model_utils import preprocess_local_dataset, train_and_evaluate

# app = FastAPI(title="FedLAD API Server")

# # --- Enable CORS for frontend ---
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],    # Update to your Next.js origin if needed
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Ensure necessary directories exist
# os.makedirs("data/uploads", exist_ok=True)
# os.makedirs("logs", exist_ok=True)

# # --- Upload Endpoint ---
# @app.post("/upload")
# async def upload_dataset(file: UploadFile = File(...)):
#     """
#     Accepts a dataset (CSV), saves it, runs preprocessing + training, logs results,
#     and includes dynamic severity metrics in the response.
#     """
#     upload_dir = os.path.join("data", "uploads")
#     logs_dir = "logs"

#     file_path = os.path.join(upload_dir, file.filename)
#     evaluation_log = os.path.join(logs_dir, "evaluation.json")
#     metrics_log = os.path.join(logs_dir, "metrics_logs.json")

#     try:
#         # Save uploaded file
#         with open(file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)

#         if not file.filename.endswith(".csv"):
#             # Clean up the partially saved file if it's the wrong type
#             os.remove(file_path)
#             raise HTTPException(status_code=400, detail="Only CSV files are supported.")

#         print("⚙️  Preprocessing local dataset...")
#         df = pd.read_csv(file_path)

#         if df.empty:
#             raise HTTPException(status_code=400, detail="Uploaded CSV is empty or invalid.")

#         original_shape = df.shape
#         processed_df, le, scaler = preprocess_local_dataset(df)
#         print(f"✅ Preprocessed dataset shape: {processed_df.shape}")

#         # Train model
#         print("🚀 Training XGBoost model...")
#         metrics = train_and_evaluate(processed_df)
#         print("✅ Training completed successfully!")

#         # --- Compute severity dynamically (NEW LOGIC) ---
#         print("📊 Computing severity data...")
#         # Get metrics with a default of 0 to prevent KeyError if missing
#         # acc = metrics.get("accuracy", 0)
#         # f1 = metrics.get("f1_score", 0)
#         # prec = metrics.get("precision", 0)
#         # rec = metrics.get("recall", 0)
#         avg_attack_conf = metrics.get("avg_attack_confidence", 0) * 100
#         attack_types_conf = metrics.get("attack_types_confidence", [50, 30, 15, 5])

#         severity_data = [
#             {"label": "DDoS", "confidence": attack_types_conf[0], "severity": 9.5},
#             {"label": "UDP Flood", "confidence": attack_types_conf[1], "severity": 7.5},
#             {"label": "PortScan", "confidence": attack_types_conf[2], "severity": 6.5},
#             {"label": "WebAttack", "confidence": attack_types_conf[3], "severity": 5.5},
#             {"label": "Benign", "confidence": round(100 - avg_attack_conf, 2), "severity": 1.0},
#          ]


#         # --- Prepare log entries ---
#         timestamp = datetime.utcnow().isoformat()
#         evaluation_entry = {
#             "timestamp": timestamp,
#             "dataset": file.filename,
#             "model": "XGBoost",
#             "severity_data": severity_data,  # <-- ADDED
#             **metrics
#         }

#         # --- Append to evaluation.json ---
#         if os.path.exists(evaluation_log):
#             try:
#                 with open(evaluation_log, "r", encoding="utf-8") as f:
#                     data = json.load(f)
#                     if not isinstance(data, list):
#                         data = []
#             except json.JSONDecodeError:
#                 data = []
#         else:
#             data = []

#         data.append(evaluation_entry)

#         # Safe write (atomic replace)
#         tmp_path = evaluation_log + ".tmp"
#         with open(tmp_path, "w", encoding="utf-8") as f:
#             json.dump(data, f, indent=2)
#         os.replace(tmp_path, evaluation_log)

#         # --- Append to metrics_logs.json (NDJSON) ---
#         with open(metrics_log, "a", encoding="utf-8") as f:
#             # Added severity_data to the metrics log entry
#             json.dump({"timestamp": timestamp, "metrics": metrics, "severity_data": severity_data}, f)
#             f.write("\n")

#         # --- Return summary ---
#         summary = {
#             "original_shape": original_shape,
#             "processed_shape": processed_df.shape,
#             "metrics": metrics,
#             "severity_data": severity_data,  # <-- ADDED to the return summary
#             "sample_preview": processed_df.head(5).to_dict(orient="records"),
#         }

#         return {
#             "message": f"✅ File '{file.filename}' uploaded, preprocessed, trained, and logged successfully!",
#             "summary": summary,
#         }

#     except Exception as e:
#         print("❌ ERROR during upload:", e)
#         traceback.print_exc()
#         # Clean up the partially saved file on general error
#         if os.path.exists(file_path):
#             try:
#                 os.remove(file_path)
#             except OSError:
#                 pass # Ignore if removal fails
#         raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")




# # --- Metrics Endpoint ---
# @app.get("/metrics")
# def get_metrics():
#     """
#     Reads the latest metrics from logs/metrics_logs.json (if exists).
#     Returns an average of recent runs or most recent metrics.
#     """
#     metrics_file = "logs/metrics_logs.json"
#     if not os.path.exists(metrics_file):
#         raise HTTPException(status_code=404, detail="Metrics log not found")

#     metrics_list = []
#     with open(metrics_file, "r", encoding="utf-8") as f:
#         for line in f:
#             try:
#                 entry = json.loads(line)
#                 # Append the entire metrics section, which now includes severity_data
#                 metrics_list.append(entry) 
#             except Exception:
#                 continue

#     if not metrics_list:
#         raise HTTPException(status_code=400, detail="No metrics found")

#     # Use the latest metrics for display
#     latest_entry = metrics_list[-1]
    
#     # Return the metrics and severity data from the latest run
#     return {"strategy": "XGBoost (Local)", **latest_entry}


# # --- Training Status (SSE Streaming Endpoint) ---
# @app.get("/training-status")
# def training_status() -> StreamingResponse:
#     """
#     Streams training progress updates to the frontend in real-time.
#     Example use: progress bars or console logs in Next.js UI.
#     """

#     def event_stream() -> Generator[str, None, None]:
#         steps = [
#             "Preparing dataset...",
#             "Preprocessing data...",
#             "Training model...",
#             "Evaluating model...",
#             "Saving metrics...",
#             "Training completed successfully 🎉",
#         ]
#         for step in steps:
#             yield f"data: {step}\n\n"
#             sleep(1.5)

#     return StreamingResponse(event_stream(), media_type="text/event-stream")


# # --- Root endpoint ---
# @app.get("/")
# def root():
#     return {"message": "🚀 FedLAD FastAPI server is running!"}


# # --- Server startup ---
# if __name__ == "__main__":
#     import uvicorn
#     # Directories are checked before uvicorn.run for convenience
#     os.makedirs("data/uploads", exist_ok=True)
#     os.makedirs("logs", exist_ok=True)
#     print("🚀 Starting FedLAD FastAPI Server on http://127.0.0.1:8000")
#     uvicorn.run(app, host="0.0.0.0", port=8000)




import os, sys, json, shutil, traceback
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from datetime import datetime
from time import sleep
from typing import Generator

# ML utilities
from model_utils import preprocess_local_dataset, train_and_evaluate

app = FastAPI(title="FedLAD API Server")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create dirs
os.makedirs("data/uploads", exist_ok=True)
os.makedirs("logs", exist_ok=True)



# ---------------------------------------------------------
#  POST /upload
# ---------------------------------------------------------
@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    upload_dir = "data/uploads"
    logs_dir = "logs"

    file_path = os.path.join(upload_dir, file.filename)
    evaluation_log = os.path.join(logs_dir, "evaluation.json")
    metrics_log = os.path.join(logs_dir, "metrics_logs.json")

    try:
        # save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if not file.filename.endswith(".csv"):
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

        print("⚙️  Preprocessing dataset...")
        df = pd.read_csv(file_path)

        if df.empty:
            raise HTTPException(status_code=400, detail="CSV is empty or invalid.")

        original_shape = df.shape
        processed_df, le, scaler = preprocess_local_dataset(df)

        print(f"✅ Preprocessed shape: {processed_df.shape}")

        print("🚀 Training model...")
        metrics = train_and_evaluate(processed_df)
        print("✅ Training complete!")

        # --------------------------
        # Severity calculation
        # --------------------------
        print("📊 Computing severity metrics...")

        avg_attack_conf = metrics.get("avg_attack_confidence", 0) * 100
        attack_types_conf = metrics.get("attack_types_confidence", [50, 30, 15, 5])

        severity_data = [
            {"label": "DDoS", "confidence": attack_types_conf[0], "severity": 9.5},
            {"label": "UDP Flood", "confidence": attack_types_conf[1], "severity": 7.5},
            {"label": "PortScan", "confidence": attack_types_conf[2], "severity": 6.5},
            {"label": "WebAttack", "confidence": attack_types_conf[3], "severity": 5.5},
            {"label": "Benign", "confidence": round(100 - avg_attack_conf, 2), "severity": 1.0},
        ]

        timestamp = datetime.utcnow().isoformat()

        evaluation_entry = {
            "timestamp": timestamp,
            "dataset": file.filename,
            "model": "XGBoost",
            "severity_data": severity_data,
            "metrics": metrics
        }

        # -----------------------------------
        # Append to evaluation.json (JSON list)
        # -----------------------------------
        if os.path.exists(evaluation_log):
            try:
                with open(evaluation_log, "r") as f:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = []
            except:
                data = []
        else:
            data = []

        data.append(evaluation_entry)

        tmp = evaluation_log + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, evaluation_log)

        # -----------------------------------
        # Append to metrics_logs.json (NDJSON)
        # -----------------------------------
        with open(metrics_log, "a", encoding="utf-8") as f:
            json.dump({
                "timestamp": timestamp,
                "metrics": metrics,
                "severity_data": severity_data
            }, f)
            f.write("\n")

        return {
            "message": f"✅ File '{file.filename}' uploaded and trained successfully!",
            "summary": {
                "original_shape": original_shape,
                "processed_shape": processed_df.shape,
                "metrics": metrics,
                "severity_data": severity_data,
                "sample_preview": processed_df.head(5).to_dict(orient="records"),
            }
        }

    except Exception as e:
        traceback.print_exc()
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))



# ---------------------------------------------------------
#  GET /logs   -------------- (Needed for UI)
# ---------------------------------------------------------
@app.get("/logs")
def get_logs():
    log_file = "logs/evaluation.json"
    if not os.path.exists(log_file):
        return []  # return empty list for first run

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except:
        return []  # corrupted file fallback



# ---------------------------------------------------------
#  GET /metrics
# ---------------------------------------------------------
@app.get("/metrics")
def get_metrics():
    metrics_file = "logs/metrics_logs.json"
    if not os.path.exists(metrics_file):
        return {}

    metrics_list = []
    with open(metrics_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                metrics_list.append(json.loads(line))
            except:
                continue

    if not metrics_list:
        return {}

    return metrics_list[-1]  # latest



# ---------------------------------------------------------
#  GET /training-status (SSE)
# ---------------------------------------------------------
@app.get("/training-status")
def training_status():
    def event_stream():
        steps = [
            "Preparing dataset...",
            "Preprocessing data...",
            "Training model...",
            "Evaluating...",
            "Saving logs...",
            "Training complete 🎉"
        ]
        for s in steps:
            yield f"data: {s}\n\n"
            sleep(1)
    return StreamingResponse(event_stream(), media_type="text/event-stream")



# ---------------------------------------------------------
# Root
# ---------------------------------------------------------
@app.get("/")
def root():
    return {"message": "🚀 FedLAD FastAPI running!"}



if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting FedLAD FastAPI Server on http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
