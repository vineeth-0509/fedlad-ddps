import type React from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

// --- Component Props Interface ---
interface StatCardProps {
  label: string // The title/label for the statistic (e.g., "Total Users")
  value: string // The main numerical value (e.g., "12,345")
  icon: React.ReactNode // An icon component to display
  trend: string // A string indicating the trend (e.g., "+5.2%" or "-1.1%")
}

// --- StatCard Component ---
const StatCard: React.FC<StatCardProps> = ({ label, value, icon, trend }) => {
  // Logic to determine if the trend is positive (starts with a '+')
  const isPositive = trend.startsWith("+")

  return (
    <Card
      // Card styling: relative positioning for hover effects, subtle gradient, and hover border
      className="relative overflow-hidden bg-gradient-to-br from-card to-card/50 border-border/50 hover:border-primary/50 transition-all group"
    >
      <CardContent className="p-6">
        <div className="flex items-start justify-between mb-4">
          {/* Icon Container */}
          <div className="p-2 rounded-lg bg-primary/10 text-primary group-hover:bg-primary/20 transition-colors">
            {icon}
          </div>
          
          {/* Trend Badge: changes variant based on positivity */}
          <Badge 
            variant={isPositive ? "default" : "secondary"} // Assuming 'default' is the primary/positive color
            className="text-xs"
          >
            {trend}
          </Badge>
        </div>
        
        {/* Label and Value */}
        <p className="text-muted-foreground text-sm font-medium mb-1">{label}</p>
        <p className="text-3xl font-bold text-foreground">{value}</p>
      </CardContent>
      
      {/* Absolute overlay for a subtle light-sweep hover effect */}
      <div className="absolute inset-0 bg-gradient-to-r from-primary/0 via-primary/5 to-primary/0 opacity-0 group-hover:opacity-100 transition-opacity" />
    </Card>
  )
}

export default StatCard