/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import React from "react";
import { Clock, CheckCircle } from "lucide-react";

interface Props {
  results: any[];
}

const TrainingTimeline: React.FC<Props> = ({ results }) => {
  if (!results || results.length === 0) return null;

  return (
    <div className="mt-6 p-6 bg-slate-800/60 rounded-xl border border-slate-700 shadow-lg">
      <h2 className="text-xl font-semibold text-cyan-400 flex items-center gap-2 mb-4">
        <Clock /> Training Timeline
      </h2>

      <div className="space-y-4">
        {results.map((round, idx) => (
          <div
            key={idx}
            className="p-4 bg-slate-900/50 rounded-lg border border-slate-700 hover:border-cyan-500 transition-all"
          >
            <div className="flex items-center justify-between">
              <h3 className="font-bold text-white">
                Round {round.round} — {round.timestamp}
              </h3>

              <CheckCircle className="text-green-400" />
            </div>

            <p className="text-gray-300 text-sm mt-1">
              Clients Trained: {round.num_clients}
            </p>

            {round.metrics && Object.keys(round.metrics).length > 0 ? (
              <p className="text-cyan-300 text-sm mt-1">
                ✔ Accuracy: {round.metrics.accuracy} — F1: {round.metrics.f1_score}
              </p>
            ) : (
              <p className="text-red-400 text-sm mt-1">⚠ No Metrics for this round</p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default TrainingTimeline;
