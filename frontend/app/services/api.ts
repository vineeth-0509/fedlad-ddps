import axios from "axios";

const API_BASE = "http://127.0.0.1:8000";

// ---------------------------------------------------------
// Helper: Parse NDJSON safely (used by federated-results)
// ---------------------------------------------------------
const parseNDJSON = (text: string) => {
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .map((line) => {
      try {
        return JSON.parse(line);
      } catch {
        return null;
      }
    })
    .filter(Boolean);
};

// ---------------------------------------------------------
// GET Evaluation Logs (/logs)
// ---------------------------------------------------------
export const getEvaluationData = async () => {
  try {
    const res = await axios.get(`${API_BASE}/logs`);
    return Array.isArray(res.data) ? res.data : [];
  } catch (err) {
    console.error("❌ Failed to fetch evaluation logs:", err);
    return [];
  }
};

// ---------------------------------------------------------
// GET Centralized Metrics (/metrics)
// This endpoint returns the latest NDJSON line as an object
// ---------------------------------------------------------
export const getMetrics = async () => {
  try {
    const res = await axios.get(`${API_BASE}/metrics`);

    const data = res.data as { strategy?: string; metrics?: { accuracy: number; f1_score: number; precision: number; recall: number } };
    if (!data || !data.metrics) return [];

    return [
      {
        strategy: data.strategy || "Centralized XGBoost",
        accuracy: Number(data.metrics.accuracy) || 0,
        f1_score: Number(data.metrics.f1_score) || 0,
        precision: Number(data.metrics.precision) || 0,
        recall: Number(data.metrics.recall) || 0,
      },
    ];
  } catch (err) {
    console.error("❌ Failed to fetch centralized metrics:", err);
    return [];
  }
};

// ---------------------------------------------------------
// UPLOAD Dataset (/upload)
// ---------------------------------------------------------
export const uploadDataset = async (file: File) => {
  try {
    const formData = new FormData();
    formData.append("file", file);

    const res = await axios.post(`${API_BASE}/upload`, formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });

    return res.data;
  } catch (err) {
    console.error("❌ Dataset upload failed:", err);
    throw err;
  }
};

// ---------------------------------------------------------
// START Federated Training (/federated-train)
// ---------------------------------------------------------
export const startFederatedTraining = async (rounds: number) => {
  try {
    const res = await axios.post(`${API_BASE}/federated-train?rounds=${rounds}`);
    return res.data;
  } catch (err) {
    console.error("❌ Federated training error:", err);
    throw err;
  }
};

// ---------------------------------------------------------
// GET Federated Results (/federated-results)
// The backend returns NDJSON (line-by-line JSON)
// ---------------------------------------------------------
export const getFederatedResults = async () => {
  try {
    const res = await fetch(`${API_BASE}/federated-results`);
    const data = await res.json();

    if (!Array.isArray(data)) return [];

    return data; // backend already returns correct JSON array
  } catch (err) {
    console.error("❌ Failed to fetch federated results:", err);
    return [];
  }
};
