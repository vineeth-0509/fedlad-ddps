import axios from "axios";

const API_BASE = "http://127.0.0.1:8000";


export const getEvaluationData = async () => {
  try {
    const res = await axios.get(`${API_BASE}/logs`);
    return res.data; // Assuming backend returns JSON logs
  } catch (err) {
    console.error("❌ Failed to fetch evaluation data:", err);
    return {};
  }
};

export const getMetrics = async () => {
  try {
    const res = await axios.get(`${API_BASE}/metrics`);
    return res.data; // Should return accuracy, precision, recall, f1_score
  } catch (err) {
    console.error("❌ Failed to fetch metrics:", err);
    return [];
  }
};

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
