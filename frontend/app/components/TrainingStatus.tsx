"use client";

import React from "react";
import { Loader2, CheckCircle, Activity } from "lucide-react";

interface Props {
  status: "idle" | "running" | "completed";
  currentRound: number;
  totalRounds: number;
}

const TrainingStatus: React.FC<Props> = ({ status, currentRound, totalRounds }) => {
  if (status === "idle") return null;

  return (
    <div className="p-4 mt-4 bg-slate-800/60 rounded-lg border border-slate-700 shadow-md flex items-center gap-4 text-white">
      {status === "running" && <Loader2 className="animate-spin text-cyan-400" />}
      {status === "completed" && <CheckCircle className="text-green-400" />}
      
      <div>
        {status === "running" && (
          <p className="font-semibold text-cyan-300">
            Federated Training Running — Round {currentRound}/{totalRounds}
          </p>
        )}
        {status === "completed" && (
          <p className="font-semibold text-green-400">
            Federated Training Completed Successfully 🎉
          </p>
        )}
      </div>
    </div>
  );
};

export default TrainingStatus;
