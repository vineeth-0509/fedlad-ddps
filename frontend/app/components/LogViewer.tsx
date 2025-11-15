/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import React from "react";

interface LogViewerProps {
  logs: any[]; // <-- FIX: logs is an ARRAY
}

const LogViewer: React.FC<LogViewerProps> = ({ logs }) => {
  if (!logs || logs.length === 0)
    return (
      <div className="text-gray-400 text-center py-4">
        No logs available.
      </div>
    );

  return (
    <div className="bg-background/50 backdrop-blur-sm border border-border/50 rounded-lg p-4 overflow-auto text-sm font-mono max-h-[400px] shadow-inner space-y-4">
      {logs.map((entry, index) => (
        <div
          key={index}
          className="bg-card/50 border border-border/30 rounded-lg p-3 hover:border-primary/50 transition-colors"
        >
          {Object.entries(entry).map(([key, value]) => (
            <div key={key} className="py-1">
              <span className="text-primary font-semibold">{key}: </span>
              <span className="text-muted-foreground">
                {typeof value === "object"
                  ? JSON.stringify(value, null, 2)
                  : String(value)}
              </span>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
};

export default LogViewer;
