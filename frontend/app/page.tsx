/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import React, { useEffect, useState } from "react";
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
import { getEvaluationData, getMetrics } from "./services/api";

// --- Types ---
export interface SeverityData {
  label: string;
  confidence: number;
  severity: number;
  color: string; // Tailwind gradient classes
  icon: React.ReactNode;
}

export interface ChartData {
  strategy: string;
  accuracy: number;
  f1_score: number;
}

// --- Static Data for Threat Overview (Default) ---
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


// --- Dashboard Component ---
const Dashboard: React.FC = () => {
  // --- State Management ---
  const [logs, setLogs] = useState<any[]>([]);
  const [metrics, setMetrics] = useState<ChartData[]>([]);
  const [loading, setLoading] = useState(true);
  
  // 1. Declare state for dynamic severity data
  const [severityData, setSeverityData] = useState<SeverityData[]>(defaultSeverityData);


  // --- Data Fetching Function ---
  const fetchData = async () => {
    try {
      const [logsResponse, metricsResponse] = await Promise.all([
        getEvaluationData(),
        getMetrics(),
      ]);

      // ✅ Safely handle logs
      const logsData = Array.isArray(logsResponse) ? logsResponse : [];
      setLogs(logsData);

      // ✅ Safely handle metrics (convert single object → array)
      let metricsData: ChartData[] = [];

      if (Array.isArray(metricsResponse)) {
        metricsData = metricsResponse as ChartData[];
      } else if (
        metricsResponse &&
        typeof metricsResponse === "object" &&
        "strategy" in metricsResponse
      ) {
        metricsData = [metricsResponse as ChartData];
      }

      setMetrics(metricsData);
      
      // OPTIONAL: Also update severity data from the latest metrics fetch on page load
      if (logsData.length > 0 && logsData[0]?.severity_data) {
          handleUploadComplete({ summary: logsData[0] });
      }

    } catch (error) {
      console.error("❌ Error loading dashboard data:", error);
      setLogs([]);
      setMetrics([]);
    } finally {
      setLoading(false);
    }
  };
  
  // 2. Implement handleUploadComplete function
  const handleUploadComplete = (summary: any) => {
      // The summary object from the API response is wrapped in a 'summary' key,
      // so we use summary.summary.severity_data if calling directly from the uploader
      // or just summary.severity_data if using the logs object
      const severity_data = summary?.summary?.severity_data || summary?.severity_data;

      if (severity_data) {
        // map backend data to include the same icons/colors your UI expects
        const mappedData: SeverityData[] = severity_data.map((item: any) => {
          let icon, color;
          switch (item.label) {
            case "DDoS": icon = <Shield />; color = "from-red-600 to-red-800"; break;
            case "UDP Flood": icon = <Zap />; color = "from-orange-500 to-orange-700"; break;
            case "PortScan": icon = <Radar />; color = "from-yellow-500 to-amber-700"; break;
            case "WebAttack": icon = <Activity />; color = "from-blue-600 to-blue-800"; break;
            case "Benign": icon = <CheckCircle />; color = "from-emerald-500 to-emerald-700"; break;
            default: icon = <Shield />; color = "from-slate-600 to-slate-800";
          }
          return { ...item, color, icon };
        });
        setSeverityData(mappedData);
      }

      fetchData(); // refresh logs and metrics
  };
  // ---------------------------------------------

  // --- Data Fetching Effect ---
  useEffect(() => {
    fetchData();
  }, []); // Runs once when page loads

  // --- Loading State UI ---
  if (loading)
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-900 text-cyan-400 text-xl">
        <span className="animate-pulse">
          ⏳ Loading FedLAD Dashboard...
        </span>
      </div>
    );

  // --- Main Dashboard UI ---
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-950 to-slate-900 text-white p-6 sm:p-10 space-y-10">
      {/* HEADER */}
      <header className="rounded-2xl bg-gradient-to-r from-cyan-600 to-indigo-600 p-8 text-center shadow-xl">
        <h1 className="text-4xl font-extrabold tracking-wide text-white drop-shadow-md">
          FedLAD DDoS Detection Dashboard
        </h1>
        <p className="text-sm sm:text-base text-white/80 mt-2">
          Federated Learning Strategy Comparison & Real-Time Threat Analysis
        </p>
      </header>

      {/* DATASET UPLOADER */}
      <section>
        {/* 3. Use the new handleUploadComplete function */}
        <DatasetUploader onUploadComplete={handleUploadComplete} /> 
      </section>

      {/* THREAT OVERVIEW */}
      <section>
        <h2 className="text-2xl font-semibold mb-4 text-cyan-400 flex items-center gap-2">
          <Flame className="w-6 h-6" /> Real-Time Threat Overview
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-6">
          {/* Use the state variable severityData */}
          {severityData.map((item) => (
            <SeverityCard key={item.label} {...item} />
          ))}
        </div>
      </section>

      {/* STRATEGY PERFORMANCE */}
      <section>
        <h2 className="text-2xl font-semibold mb-4 text-cyan-400 flex items-center gap-2">
          <BarChart className="w-6 h-6" /> Strategy Performance
        </h2>
        <div className="rounded-2xl bg-slate-800/50 p-6 border border-slate-700 shadow-lg">
          <PerformanceChart data={metrics} />
        </div>
      </section>

      {/* EVALUATION LOGS */}
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