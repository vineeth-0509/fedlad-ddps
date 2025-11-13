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
  { label: "DDoS", confidence: 95, severity: 9.5, color: "from-red-600 to-red-800", icon: <Shield /> },
  { label: "UDP Flood", confidence: 82, severity: 7.5, color: "from-orange-500 to-orange-700", icon: <Zap /> },
  { label: "PortScan", confidence: 73, severity: 6.5, color: "from-yellow-500 to-amber-700", icon: <Radar /> },
  { label: "WebAttack", confidence: 60, severity: 5.5, color: "from-blue-600 to-blue-800", icon: <Activity /> },
  { label: "Benign", confidence: 99, severity: 1.0, color: "from-emerald-500 to-emerald-700", icon: <CheckCircle /> },
];

const Dashboard: React.FC = () => {
  const [logs, setLogs] = useState<any[]>([]);
  const [metrics, setMetrics] = useState<ChartData[]>([]);
  const [loading, setLoading] = useState(true);
  const [severityData, setSeverityData] = useState<SeverityData[]>(defaultSeverityData);

  const fetchData = async () => {
    try {
      const [logsResponse, metricsResponse]: [any, any] = await Promise.all([
        getEvaluationData(),
        getMetrics(),
      ]);

      const logsData = Array.isArray(logsResponse) ? logsResponse : [];
      setLogs(logsData);


      const normalizeMetricItem = (raw: any): ChartData => {
        const strategy = typeof raw?.strategy === 'string' ? raw.strategy : typeof raw?.metrics?.strategy === 'string' ? raw.metrics.strategy : "";
        const accuracy = typeof raw?.accuracy === "number" ?  raw.accuracy : typeof raw?.metrics?.accuracy === "number" ? raw.metrics.accuracy : 0;
        const f1_score = typeof raw?.f1_score === "number" ? raw.f1_score : typeof raw?.metrics?.f1_score === 'number' ? raw.metrics.f1_score : 0;

        const precision = typeof raw?.precision === "number" ? raw.precision : typeof raw?.metrics?.precision === "number" ? raw.metrics.precision : 0;
        const recall =  typeof raw?.recall === "number"
          ? raw.recall
          : typeof raw?.metrics?.recall === "number"
          ? raw.metrics.recall
          : 0;

      return {
        strategy,
        accuracy,
        f1_score,
        precision,
        recall,
      };

      }

      // let metricsData: ChartData[] = [];
      // if (Array.isArray(metricsResponse)) metricsData = metricsResponse;
      // else if (metricsResponse && typeof metricsResponse === "object" && "strategy" in metricsResponse)
      //   metricsData = [metricsResponse as ChartData];
        
      let metricsData : ChartData[] = [];
      if(Array.isArray(metricsResponse)){
        metricsData = (metricsResponse as any[]).map((item) => normalizeMetricItem(item)); 
      } else if(metricsResponse && typeof metricsResponse === "object"){
          if(metricsResponse?.metrics && typeof metricsResponse.metrics === "object"){
              metricsData = [normalizeMetricItem({...metricsResponse.metrics, strategy:metricsResponse.strategy})];
          }else {
            metricsData = [normalizeMetricItem(metricsResponse)];
          }
      }
      setMetrics(metricsData);

      // if (logsData.length > 0 && logsData[0]?.severity_data)
      //   handleUploadComplete({ summary: logsData[0] });
    } catch (err) {
      console.error("❌ Error loading data:", err);
      setLogs([]); setMetrics([]);
    } finally {
      setLoading(false);
    }
  };

  const handleUploadComplete = (summary: any) => {
    const severity_data = summary?.summary?.severity_data || summary?.severity_data;
    if (severity_data) {
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
    fetchData();
  };

  useEffect(() => { fetchData(); }, []);

  if (loading)
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-900 text-cyan-400 text-xl">
        <span className="animate-pulse">⏳ Loading FedLAD Dashboard...</span>
      </div>
    );

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-950 to-slate-900 text-white p-6 sm:p-10 space-y-10">
      <header className="rounded-2xl bg-gradient-to-r from-cyan-600 to-indigo-600 p-8 text-center shadow-xl">
        <h1 className="text-4xl font-extrabold tracking-wide">FedLAD DDoS Detection Dashboard</h1>
        <p className="text-sm text-white/80 mt-2">Federated Learning Strategy Comparison & Real-Time Threat Analysis</p>
      </header>

      <section>
        <DatasetUploader onUploadComplete={handleUploadComplete} />
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-4 text-cyan-400 flex items-center gap-2">
          <Flame className="w-6 h-6" /> Real-Time Threat Overview
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-6">
          {severityData.map((item) => <SeverityCard key={item.label} {...item} />)}
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-4 text-cyan-400 flex items-center gap-2">
          <BarChart className="w-6 h-6" /> Strategy Performance
        </h2>
        <div className="rounded-2xl bg-slate-800/50 p-6 border border-slate-700 shadow-lg">
          {metrics.length > 0
            ? <PerformanceChart data={metrics} />
            : <p className="text-gray-400 text-center">No metrics available yet.</p>}
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
