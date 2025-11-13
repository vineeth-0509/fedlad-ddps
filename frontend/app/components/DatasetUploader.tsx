/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import React, { useState, useEffect } from "react";

interface Props {
  onUploadComplete?: (summary: any) => void;
}

const API_BASE = "http://127.0.0.1:8000"; // ✅ always use the same as backend

const DatasetUploader: React.FC<Props> = ({ onUploadComplete }) => {
  const [file, setFile] = useState<File | null>(null);
  const [progress, setProgress] = useState<number>(0);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState("");

  // Simulate progress bar animation (UI feedback)
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (uploading) {
      setProgress(0);
      interval = setInterval(() => {
        setProgress((prev) => {
          if (prev < 90) return prev + 5;
          clearInterval(interval);
          return prev;
        });
      }, 300);
    }
    return () => clearInterval(interval);
  }, [uploading]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      setFile(e.target.files[0]);
      setMessage("");
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setMessage("⚠️ Please select a CSV file before uploading.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      setUploading(true);
      setMessage("⏳ Uploading and training model...");

      const response = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        setProgress(100);
        const result = await response.json();
        console.log("✅ Upload success:", result);
        setMessage("✅ Dataset uploaded and model training completed!");

        // ✅ Automatically trigger dashboard refresh
        setTimeout(() => {
          if (onUploadComplete) onUploadComplete({summary: result.summary});
        }, 1500);
      } else {
        const errText = await response.text();
        console.error("❌ Upload failed:", errText);
        setMessage(`❌ Upload failed: ${errText}`);
      }
    } catch (error) {
      console.error("❌ Upload error:", error);
      setMessage("❌ Something went wrong during upload.");
    } finally {
      setUploading(false);
      setTimeout(() => setProgress(0), 2000);
    }
  };

  return (
    <div className="p-6 border border-slate-700 rounded-2xl bg-slate-800/60 shadow-lg space-y-4">
      <h2 className="text-xl font-semibold text-cyan-400">📤 Upload New Dataset</h2>
      <p className="text-sm text-gray-400">
        Upload your CSV dataset to retrain the federated DDoS detection model.
      </p>

      <input
        type="file"
        accept=".csv"
        onChange={handleFileChange}
        className="block w-full text-sm text-gray-300 border border-slate-600 rounded-lg cursor-pointer bg-slate-700 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-cyan-600 file:text-white hover:file:bg-cyan-700"
      />

      {/* Upload Button */}
      <button
        onClick={handleUpload}
        disabled={uploading}
        className={`w-full px-6 py-2 rounded-lg font-medium transition-colors ${
          uploading
            ? "bg-gray-600 cursor-not-allowed"
            : "bg-cyan-600 hover:bg-cyan-700"
        }`}
      >
        {uploading ? "Uploading..." : "Start Upload"}
      </button>

      {/* Progress Bar */}
      {uploading && (
        <div className="w-full bg-slate-700 rounded-full h-2 overflow-hidden">
          <div
            className="bg-cyan-500 h-2 rounded-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
      )}

      {/* Status Message */}
      {message && (
        <p className="text-sm text-gray-300 mt-2 whitespace-pre-wrap">{message}</p>
      )}
    </div>
  );
};

export default DatasetUploader;
