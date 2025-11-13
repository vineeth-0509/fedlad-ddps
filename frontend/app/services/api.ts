

// // app/services/api.ts
// import axios from "axios";

// export const API_BASE = "http://127.0.0.1:8000";

// // ----------------------
// // GET Logs (evaluation.json)
// // ----------------------
// export const getEvaluationData = async () => {
//   try {
//     const res = await axios.get(`${API_BASE}/logs`);
//     return Array.isArray(res.data) ? res.data : [];
//   } catch (err) {
//     console.error("❌ Failed to fetch evaluation logs:", err);
//     return [];
//   }
// };

// // ----------------------
// // GET Metrics (metrics_logs.json latest entry)
// // ----------------------
// export const getMetrics = async () => {
//   try {
//     const res = await axios.get(`${API_BASE}/metrics`);
//     return res.data || null;
//   } catch (err) {
//     console.error("❌ Failed to fetch metrics:", err);
//     return null;
//   }
// };

// // ----------------------
// // UPLOAD Dataset & Train
// // ----------------------
// export const uploadDataset = async (file: File) => {
//   const formData = new FormData();
//   formData.append("file", file);

//   try {
//     const res = await axios.post(`${API_BASE}/upload`, formData, {
//       headers: { "Content-Type": "multipart/form-data" },
//     });
//     return res.data;
//   } catch (err) {
//     console.error("❌ Dataset upload failed:", err);
//     throw err;
//   }
// };

// // ----------------------
// // Training Status Stream (SSE)
// // ----------------------
// export const listenTrainingStatus = (onMessage: (msg: string) => void) => {
//   const eventSource = new EventSource(`${API_BASE}/training-status`);

//   eventSource.onmessage = (event) => {
//     onMessage(event.data);
//   };

//   eventSource.onerror = () => {
//     console.error("❌ SSE connection failed");
//     eventSource.close();
//   };

//   return eventSource;
// };

import axios from "axios";

const API_BASE = "http://127.0.0.1:8000";

// -----------------------
// GET Evaluation Logs
// -----------------------
export const getEvaluationData = async () => {
  try {
    const res = await axios.get(`${API_BASE}/logs`);
    return Array.isArray(res.data) ? res.data : [];
  } catch (err) {
    console.error("❌ Failed to fetch evaluation data:", err);
    return [];
  }
};

// -----------------------
// GET Metrics
// -----------------------
interface MetricsPayload {
  strategy?: string;
  metrics?: {
    accuracy?: number;
    f1_score?: number;
    precision?: number;
    recall?: number;
  };
}

export const getMetrics = async () => {
  try {
    const res = await axios.get<MetricsPayload>(`${API_BASE}/metrics`);
    const data = res.data;

    if (!data || !data.metrics) return [];

    return [
      {
        strategy: data.strategy,
        accuracy: data.metrics.accuracy,
        f1_score: data.metrics.f1_score,
        precision: data.metrics.precision,
        recall: data.metrics.recall,
      },
    ];
  } catch (err) {
    console.error("❌ Failed to fetch metrics:", err);
    return [];
  }
};

// -----------------------
// UPLOAD Dataset
// -----------------------
export const uploadDataset = async (file: File) => {
  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await axios.post(`${API_BASE}/upload`, formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });

    return res.data;
  } catch (err) {
    console.error("❌ Dataset upload failed:", err);
    throw err;
  }
};
