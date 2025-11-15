/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import React, { useState } from "react";
import { Play, Loader2, CheckCircle, Activity } from "lucide-react";
import { startFederatedTraining } from "../services/api";
import confetti from "canvas-confetti";

interface Props {
  onTrainingComplete?: () => void;
  setTrainingStatus: (status: "idle" | "running" | "completed") => void;  // <-- FIX
}

const FederatedTraining: React.FC<Props> = ({ onTrainingComplete, setTrainingStatus }) => {
  const [rounds, setRounds] = useState<number>(3);
  const [loading, setLoading] = useState(false);

  const [status, setStatus] = useState<"idle" | "running" | "completed">("idle");
  const [message, setMessage] = useState("");
  const [currentRound, setCurrentRound] = useState(0);

  const handleStartTraining = async () => {
    setLoading(true);
    setStatus("running");
    setTrainingStatus("running");   // <-- IMPORTANT
    setMessage("🚀 Federated Learning Started...");
    setCurrentRound(0);

    try {
      const res: any = await startFederatedTraining(rounds);

      if (res?.results && Array.isArray(res.results)) {
        res.results.forEach((round: any, i: number) => {
          setTimeout(() => {
            setCurrentRound(round.round);
          }, i * 700);
        });
      }

      setTimeout(() => {
        setStatus("completed");
        setTrainingStatus("completed"); // <-- IMPORTANT
        setMessage("🎉 Federated Training Completed Successfully!");

        confetti({
          particleCount: 120,
          spread: 100,
          origin: { y: 0.7 },
        });

        if (onTrainingComplete) {
          setTimeout(() => onTrainingComplete(), 1000);
        }
      }, rounds * 750);
    } catch (err) {
      console.error(err);
      setStatus("idle");
      setTrainingStatus("idle"); // <-- IMPORTANT
      setMessage("❌ Error running Federated Learning");
    }

    setLoading(false);
  };

  return (
    <div className="p-6 bg-slate-800/60 rounded-xl border border-slate-700 shadow-lg space-y-4">
      <h2 className="text-2xl font-semibold text-cyan-400 flex items-center gap-2">
        🛰️ Federated Learning Control Panel
      </h2>

      <div className="flex items-center gap-4">
        <label className="text-gray-300">Rounds:</label>
        <input
          type="number"
          min={1}
          max={20}
          value={rounds}
          disabled={loading}
          onChange={(e) => setRounds(parseInt(e.target.value))}
          className="w-24 bg-slate-700 border border-slate-600 text-white rounded-lg px-2 py-1"
        />
      </div>

      <button
        onClick={handleStartTraining}
        disabled={loading}
        className={`flex items-center justify-center gap-2 w-full py-2 rounded-lg text-white font-semibold transition-all ${
          loading
            ? "bg-gray-600 cursor-not-allowed"
            : "bg-cyan-600 hover:bg-cyan-700"
        }`}
      >
        {loading ? <Loader2 className="animate-spin" /> : <Play />}
        {loading ? "Running..." : "Start Federated Training"}
      </button>

      {status !== "idle" && (
        <div
          className={`p-4 rounded-lg flex items-center gap-3 border transition-all ${
            status === "running"
              ? "bg-blue-900/40 border-blue-700 text-blue-300"
              : "bg-green-900/40 border-green-700 text-green-300"
          }`}
        >
          {status === "running" && <Activity className="animate-pulse" />}
          {status === "completed" && <CheckCircle className="text-green-400" />}

          <div>
            {status === "running" && (
              <p>
                ⏳ Training Round <b>{currentRound}</b> of <b>{rounds}</b>
              </p>
            )}
            {status === "completed" && <p>🎉 Training Completed!</p>}
          </div>
        </div>
      )}

      {message && (
        <p className="text-sm text-gray-300 mt-2 whitespace-pre-wrap">
          {message}
        </p>
      )}
    </div>
  );
};

export default FederatedTraining;
