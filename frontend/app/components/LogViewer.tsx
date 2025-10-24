"use client";
import type React from "react";
interface LogViewerProps {
  logs: Record<string, any>;
}
const LogViewer: React.FC<LogViewerProps> = ({ logs }) => {
  return (
    <div className="bg-background/50 backdrop-blur-sm border border-border/50 rounded-lg p-4 overflow-auto text-sm font-mono max-h-[400px] shadow-inner">
      <div className="space-y-3">
        {Object.entries(logs).map(([key, value]) => (
          <div
            key={key}
            className="py-2 px-3 rounded-lg bg-card/50 border border-border/30 hover:border-primary/50 transition-colors"
          >
            {" "}
            <span className="text-primary font-semibold">{key}: </span>{" "}
            <span className="text-muted-foreground">
              {JSON.stringify(value)}
            </span>{" "}
          </div>
        ))}{" "}
      </div>{" "}
    </div>
  );
};
export default LogViewer;
