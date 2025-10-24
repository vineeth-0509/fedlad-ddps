"use client";
import type React from "react";
import {
  ResponsiveContainer,
  BarChart,
  XAxis,
  YAxis,
  Tooltip,
  Bar,
  CartesianGrid,
  Legend,
} from "recharts";
import type { ChartData } from "@/app/page";
interface Props {
  data: ChartData[];
}
const PerformanceChart: React.FC<Props> = ({ data }) => {
  const colors = ["#8b5cf6", "#06b6d4"];
  return (
    <div className="w-full">
      {" "}
      <div className="h-80 w-full">
        {" "}
        <ResponsiveContainer width="100%" height="100%">
          {" "}
          <BarChart
            data={data}
            margin={{ top: 20, right: 30, left: 0, bottom: 20 }}
          >
            {" "}
            <defs>
              {" "}
              <linearGradient id="colorAccuracy" x1="0" y1="0" x2="0" y2="1">
                {" "}
                <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.8} />{" "}
                <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0.2} />{" "}
              </linearGradient>{" "}
              <linearGradient id="colorF1" x1="0" y1="0" x2="0" y2="1">
                {" "}
                <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.8} />{" "}
                <stop offset="95%" stopColor="#06b6d4" stopOpacity={0.2} />{" "}
              </linearGradient>{" "}
            </defs>{" "}
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="rgba(255,255,255,0.1)"
            />{" "}
            <XAxis dataKey="strategy" stroke="rgba(255,255,255,0.5)" />{" "}
            <YAxis stroke="rgba(255,255,255,0.5)" />{" "}
            <Tooltip
              contentStyle={{
                backgroundColor: "rgba(15, 23, 42, 0.95)",
                border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: "8px",
                color: "#fff",
              }}
              cursor={{ fill: "rgba(139, 92, 246, 0.1)" }}
            />{" "}
            <Legend wrapperStyle={{ paddingTop: "20px" }} />{" "}
            <Bar
              dataKey="accuracy"
              fill="url(#colorAccuracy)"
              radius={[8, 8, 0, 0]}
              name="Accuracy"
            />{" "}
            <Bar
              dataKey="f1_score"
              fill="url(#colorF1)"
              radius={[8, 8, 0, 0]}
              name="F1 Score"
            />{" "}
          </BarChart>{" "}
        </ResponsiveContainer>{" "}
      </div>{" "}
    </div>
  );
};
export default PerformanceChart;
