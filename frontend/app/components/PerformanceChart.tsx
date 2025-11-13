// "use client";
// import React from "react";
// import {
//   ResponsiveContainer,
//   BarChart,
//   XAxis,
//   YAxis,
//   Tooltip,
//   Bar,
//   CartesianGrid,
//   Legend,
// } from "recharts";

// export interface ChartData {
//   strategy: string;
//   accuracy: number;
//   f1_score: number;
// }

// interface Props {
//   data: ChartData[] | ChartData; // support both object and array input
// }

// const PerformanceChart: React.FC<Props> = ({ data }) => {
//   // Ensure the chart always receives an array
//   const normalizedData = Array.isArray(data) ? data : [data];

//   const colors = ["#8b5cf6", "#06b6d4"];

//   return (
//     <div className="w-full">
//       <div className="h-80 w-full">
//         <ResponsiveContainer width="100%" height="100%">
//           <BarChart
//             data={normalizedData}
//             margin={{ top: 20, right: 30, left: 0, bottom: 20 }}
//           >
//             <defs>
//               <linearGradient id="colorAccuracy" x1="0" y1="0" x2="0" y2="1">
//                 <stop offset="5%" stopColor={colors[0]} stopOpacity={0.8} />
//                 <stop offset="95%" stopColor={colors[0]} stopOpacity={0.2} />
//               </linearGradient>
//               <linearGradient id="colorF1" x1="0" y1="0" x2="0" y2="1">
//                 <stop offset="5%" stopColor={colors[1]} stopOpacity={0.8} />
//                 <stop offset="95%" stopColor={colors[1]} stopOpacity={0.2} />
//               </linearGradient>
//             </defs>

//             <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
//             <XAxis dataKey="strategy" stroke="rgba(255,255,255,0.5)" />
//             <YAxis stroke="rgba(255,255,255,0.5)" />

//             <Tooltip
//               contentStyle={{
//                 backgroundColor: "rgba(15, 23, 42, 0.95)",
//                 border: "1px solid rgba(255,255,255,0.1)",
//                 borderRadius: "8px",
//                 color: "#fff",
//               }}
//               cursor={{ fill: "rgba(139, 92, 246, 0.1)" }}
//             />

//             <Legend wrapperStyle={{ paddingTop: "20px" }} />

//             <Bar
//               dataKey="accuracy"
//               fill="url(#colorAccuracy)"
//               radius={[8, 8, 0, 0]}
//               name="Accuracy"
//             />
//             <Bar
//               dataKey="f1_score"
//               fill="url(#colorF1)"
//               radius={[8, 8, 0, 0]}
//               name="F1 Score"
//             />
//           </BarChart>
//         </ResponsiveContainer>
//       </div>
//     </div>
//   );
// };

// export default PerformanceChart;

"use client";
import React from "react";
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

export interface ChartData {
  strategy: string;
  accuracy: number;
  f1_score: number;
  precision: number;
  recall: number;
}

interface Props {
  data: ChartData[] | ChartData;
}

const PerformanceChart: React.FC<Props> = ({ data }) => {
  // Always normalize to array
  const normalizedData = Array.isArray(data) ? data : [data];

  return (
    <div className="w-full">
      <div className="h-96 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={normalizedData}
            margin={{ top: 20, right: 30, left: 0, bottom: 20 }}
          >
            <defs>
              <linearGradient id="colorAccuracy" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#34d399" stopOpacity={0.8} />
                <stop offset="95%" stopColor="#34d399" stopOpacity={0.2} />
              </linearGradient>

              <linearGradient id="colorF1" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#60a5fa" stopOpacity={0.8} />
                <stop offset="95%" stopColor="#60a5fa" stopOpacity={0.2} />
              </linearGradient>

              <linearGradient id="colorPrecision" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#fbbf24" stopOpacity={0.8} />
                <stop offset="95%" stopColor="#fbbf24" stopOpacity={0.2} />
              </linearGradient>

              <linearGradient id="colorRecall" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#f472b6" stopOpacity={0.8} />
                <stop offset="95%" stopColor="#f472b6" stopOpacity={0.2} />
              </linearGradient>
            </defs>

            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis dataKey="strategy" stroke="rgba(255,255,255,0.5)" />
            <YAxis stroke="rgba(255,255,255,0.5)" />

            <Tooltip
              contentStyle={{
                backgroundColor: "rgba(15, 23, 42, 0.95)",
                border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: "8px",
                color: "#fff",
              }}
            />

            <Legend wrapperStyle={{ paddingTop: "20px" }} />

            <Bar
              dataKey="accuracy"
              fill="url(#colorAccuracy)"
              radius={[8, 8, 0, 0]}
              name="Accuracy"
            />
            <Bar
              dataKey="f1_score"
              fill="url(#colorF1)"
              radius={[8, 8, 0, 0]}
              name="F1 Score"
            />
            <Bar
              dataKey="precision"
              fill="url(#colorPrecision)"
              radius={[8, 8, 0, 0]}
              name="Precision"
            />
            <Bar
              dataKey="recall"
              fill="url(#colorRecall)"
              radius={[8, 8, 0, 0]}
              name="Recall"
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default PerformanceChart;
