/* eslint-disable @typescript-eslint/ban-ts-comment */
import type React from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

interface Props {
  label: string
  confidence: number
  severity: number
  color: string
  icon: React.ReactNode
}

const SeverityCard: React.FC<Props> = ({
  label,
  confidence,
  severity,
  color,
  icon,
}) => {
  // Determine severity level and badge color based on the severity score
  const severityLevel =
    severity > 8
      ? "Critical"
      : severity > 5
      ? "High"
      : severity > 2
      ? "Medium"
      : "Low"

  // Map severity score to a badge variant, assuming Shadcn/ui variants like 'destructive', 'secondary', 'default', 'outline'
  const severityColor: "destructive" | "secondary" | "default" | "outline" =
    severity > 8
      ? "destructive"
      : severity > 5
      ? "secondary"
      : severity > 2
      ? "default"
      : "outline"

  return (
    // Card component with dynamic background color (via `color` prop) and hover effects
    <Card
      className={`relative overflow-hidden bg-gradient-to-br ${color} text-white rounded-xl shadow-lg transition-all transform hover:scale-105 hover:shadow-2xl group border-0`}
    >
      <CardContent className="p-5 flex flex-col items-center text-center space-y-3">
        {/* Icon */}
        <div className="text-4xl group-hover:scale-110 transition-transform">
          {icon}
        </div>
        
        {/* Label */}
        <h3 className="text-lg font-bold">{label}</h3>

        {/* Confidence Progress Bar */}
        <div className="w-full space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="opacity-90">Confidence</span>
            <span className="font-semibold">{confidence}%</span>
          </div>
          <div className="w-full h-1.5 bg-white/20 rounded-full overflow-hidden">
            {/* Progress bar fill with dynamic width */}
            <div
              className="h-full bg-white/80 rounded-full transition-all"
              style={{ width: `${confidence}%` }}
            />
          </div>
        </div>

        {/* Severity Badge */}
        <div className="w-full pt-2 border-t border-white/20">
          <Badge
            // @ts-ignore: The variant prop should accept the mapped string
            variant={severityColor}
            className="w-full justify-center text-xs font-medium"
          >
            Severity: {severity.toFixed(1)} - {severityLevel}
          </Badge>
        </div>
      </CardContent>

      {/* Hover overlay for a subtle effect */}
      <div className="absolute inset-0 bg-white/10 opacity-0 group-hover:opacity-20 rounded-xl blur-2xl transition-opacity" />
    </Card>
  )
}

export default SeverityCard