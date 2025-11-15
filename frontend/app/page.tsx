/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import React, { useEffect, useRef, useState } from "react";
import {
  Shield,
  Activity,
  Radar,
  Zap,
  CheckCircle,
  Flame,
  BarChart,
  FileText,
} from "lucide-react";

import DatasetUploader from "./components/DatasetUploader";
import SeverityCard from "./components/SeverityCard";
import PerformanceChart from "./components/PerformanceChart";
import LogViewer from "./components/LogViewer";
import {
  getEvaluationData,
  getFederatedResults,
  getMetrics,
} from "./services/api";
import FederatedTraining from "./components/FederatedTraining";

export interface SeverityData {
  label: string;
  confidence: number;
  severity: number;
  color: string;
  icon: React.ReactNode;
}

export interface ChartData {
  strategy: string;
  accuracy: number;
  f1_score: number;
  precision: number;
  recall: number;
}

const defaultSeverityData: SeverityData[] = [
  {
    label: "DDoS",
    confidence: 95,
    severity: 9.5,
    color: "from-red-600 to-red-800",
    icon: <Shield />,
  },
  {
    label: "UDP Flood",
    confidence: 82,
    severity: 7.5,
    color: "from-orange-500 to-orange-700",
    icon: <Zap />,
  },
  {
    label: "PortScan",
    confidence: 73,
    severity: 6.5,
    color: "from-yellow-500 to-amber-700",
    icon: <Radar />,
  },
  {
    label: "WebAttack",
    confidence: 60,
    severity: 5.5,
    color: "from-blue-600 to-blue-800",
    icon: <Activity />,
  },
  {
    label: "Benign",
    confidence: 99,
    severity: 1.0,
    color: "from-emerald-500 to-emerald-700",
    icon: <CheckCircle />,
  },
];

const Dashboard: React.FC = () => {
  const [logs, setLogs] = useState<any[]>([]);
  const [metrics, setMetrics] = useState<ChartData[]>([]);
  const [loading, setLoading] = useState(true);
  const [severityData, setSeverityData] =
    useState<SeverityData[]>(defaultSeverityData);
  const [trainingStatus, setTrainingStatus] = useState<
    "idle" | "running" | "completed"
  >("idle");

  // keep a ref to avoid multiple overlapping fetches
  const fetchingRef = useRef(false);

  /**
   * map severity entries from server to UI-friendly shape (add icon/color)
   */
  const mapSeverityItems = (items: any[]): SeverityData[] => {
    return items.map((item: any) => {
      const label = item.label ?? String(item.name ?? "Unknown");
      const confidence = typeof item.confidence === "number" ? item.confidence : Number(item.confidence) || 0;
      const severity = typeof item.severity === "number" ? item.severity : Number(item.severity) || 0;

      let icon: React.ReactNode = <Shield />;
      let color = "from-slate-600 to-slate-800";

      switch (label) {
        case "DDoS":
          icon = <Shield />;
          color = "from-red-600 to-red-800";
          break;
        case "UDP Flood":
          icon = <Zap />;
          color = "from-orange-500 to-orange-700";
          break;
        case "PortScan":
          icon = <Radar />;
          color = "from-yellow-500 to-amber-700";
          break;
        case "WebAttack":
          icon = <Activity />;
          color = "from-blue-600 to-blue-800";
          break;
        case "Benign":
          icon = <CheckCircle />;
          color = "from-emerald-500 to-emerald-700";
          break;
        default:
          icon = <Shield />;
          color = "from-slate-600 to-slate-800";
      }

      return {
        label,
        confidence,
        severity,
        color,
        icon,
      };
    });
  };

  /**
   * Fetch centralized metrics, federated results and logs.
   * Keeps the UI robust to shape differences.
   */
  const fetchData = async () => {
    if (fetchingRef.current) return;
    fetchingRef.current = true;

    try {
      // Build final metrics array (centralized first, federated second)
      const finalMetrics: ChartData[] = [];

      // 1) Centralized metrics (GET /metrics)
      try {
        const centralized = await getMetrics(); // api returns [] or [{...}]
        if (Array.isArray(centralized) && centralized.length > 0) {
          const m = centralized[0]; // expected object
          finalMetrics.push({
            strategy: m.strategy ?? "Centralized XGBoost",
            accuracy: typeof m.accuracy === "number" ? m.accuracy : Number(m.accuracy) || 0,
            precision: typeof m.precision === "number" ? m.precision : Number(m.precision) || 0,
            recall: typeof m.recall === "number" ? m.recall : Number(m.recall) || 0,
            f1_score: typeof m.f1_score === "number" ? m.f1_score : Number(m.f1_score) || 0,
          });
        }
      } catch (err) {
        console.warn("Failed to load centralized metrics", err);
      }

      // 2) Federated results (GET /federated-results)
      try {
        const federatedResults = await getFederatedResults();
        if (Array.isArray(federatedResults) && federatedResults.length > 0) {
          // use last round as "Federated Avg" (your backend writes NDJSON lines)
          const lastRound = federatedResults[federatedResults.length - 1];
          const fm = lastRound?.metrics ?? lastRound;
          finalMetrics.push({
            strategy: "Federated Avg",
            accuracy: typeof fm.accuracy === "number" ? fm.accuracy : Number(fm.accuracy) || 0,
            precision: typeof fm.precision === "number" ? fm.precision : Number(fm.precision) || 0,
            recall: typeof fm.recall === "number" ? fm.recall : Number(fm.recall) || 0,
            f1_score: typeof fm.f1_score === "number" ? fm.f1_score : Number(fm.f1_score) || 0,
          });
        }
      } catch (err) {
        // backend sometimes returns 500 during training; handle gracefully
        console.warn("Failed to load federated results (may be training)", err);
      }

      setMetrics(finalMetrics);

      // 3) Logs (GET /logs)
      try {
        const logsResponse = await getEvaluationData(); // array expected
        const logsArray = Array.isArray(logsResponse) ? logsResponse : [];
        setLogs(logsArray);

        // If the latest log contains severity_data, update the severity cards
        if (logsArray.length > 0) {
          const latest = logsArray[logsArray.length - 1];
          const severity_data = latest?.summary?.severity_data ?? latest?.severity_data ?? latest?.metrics?.attack_types_confidence ?? null;

          // The API may provide an array or an object with arrays; normalize
          if (Array.isArray(severity_data) && severity_data.length > 0) {
            setSeverityData(mapSeverityItems(severity_data));
          } else if (Array.isArray(latest?.severity_data)) {
            setSeverityData(mapSeverityItems(latest.severity_data));
          } else {
            // fallback, don't override existing severityData
          }
        }
      } catch (err) {
        console.warn("Failed to fetch logs", err);
        setLogs([]); // safe fallback
      }
    } catch (err) {
      console.error("❌ Error in fetchData", err);
    } finally {
      fetchingRef.current = false;
      setLoading(false);
    }
  };

  // Called when dataset uploader finishes: it passes summary
  const handleUploadComplete = (summary: any) => {
    // summary may contain severity_data
    const severity_data =
      summary?.summary?.severity_data || summary?.severity_data;

    if (Array.isArray(severity_data) && severity_data.length > 0) {
      setSeverityData(mapSeverityItems(severity_data));
    }

    // refresh dashboard to pick up new metrics/logs
    fetchData();
  };

  // Initial load
  useEffect(() => {
    fetchData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Poll while training is running (poll every 2.5s)
  useEffect(() => {
    if (trainingStatus !== "running") return;
    const id = setInterval(() => {
      fetchData();
    }, 2500);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trainingStatus]);

  // Loading UI
  if (loading)
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-900 text-cyan-400 text-xl">
        <span className="animate-pulse">⏳ Loading FedLAD Dashboard...</span>
      </div>
    );

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-950 to-slate-900 text-white p-6 sm:p-10 space-y-10">
      <header className="rounded-2xl bg-gradient-to-r from-cyan-600 to-indigo-600 p-8 text-center shadow-xl">
        <h1 className="text-4xl font-extrabold tracking-wide">
          FedLAD DDoS Detection Dashboard
        </h1>
        <p className="text-sm text-white/80 mt-2">
          Centralized vs Federated Learning — Real-Time Attack Detection
        </p>
      </header>

      <section>
        <DatasetUploader onUploadComplete={handleUploadComplete} />
      </section>

      <section className="mt-10">
        <FederatedTraining
          onTrainingComplete={fetchData}
          setTrainingStatus={(s) => setTrainingStatus(s)}
        />
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-4 text-cyan-400 flex items-center gap-2">
          <Flame className="w-6 h-6" /> Real-Time Threat Overview
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-6">
          {severityData.map((item) => (
            <SeverityCard key={item.label} {...item} />
          ))}
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-4 text-cyan-400 flex items-center gap-2">
          <BarChart className="w-6 h-6" /> Strategy Performance
        </h2>
        <div className="rounded-2xl bg-slate-800/50 p-6 border border-slate-700 shadow-lg">
          {metrics.length > 0 ? (
            <PerformanceChart data={metrics} />
          ) : (
            <p className="text-gray-400 text-center">No metrics available yet.</p>
          )}
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-4 text-cyan-400 flex items-center gap-2">
          <FileText className="w-6 h-6" /> Evaluation Logs
        </h2>
        <div className="rounded-2xl bg-slate-800/50 border border-slate-700 p-6 shadow-lg">
          <LogViewer logs={logs} />
        </div>
      </section>
    </div>
  );
};

export default Dashboard;
