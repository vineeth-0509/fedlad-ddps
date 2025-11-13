import type React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

// Assuming ChartData structure based on usage in the component
// (You'll need to define this type in your actual application file, e.g., '@/app/page')
/*
type ChartData = {
  strategy: string;
  accuracy: number; // expected value between 0 and 1
  f1_score: number; // expected value between 0 and 1
};
*/
import type { ChartData } from "@/app/page";

interface Props {
  data: ChartData[];
}

const StrategyComparison: React.FC<Props> = ({ data }) => {
  // Sort the data in descending order based on accuracy
  const sortedData = [...data].sort((a, b) => b.accuracy - a.accuracy);

  return (
    <Card className="bg-gradient-to-br from-card to-card/50 border-border/50 h-full">
      <CardHeader className="pb-4">
        <CardTitle className="text-lg">Strategy Rankings</CardTitle>
      </CardHeader>

      <CardContent className="space-y-4">
        {sortedData.map((strategy, idx) => (
          <div key={strategy.strategy} className="space-y-2">
            {/* Header: Rank, Strategy Name, and Accuracy Percentage */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {/* Rank Badge */}
                <Badge
                  variant="outline"
                  className="w-6 h-6 flex items-center justify-center p-0 font-bold">
                  {idx + 1}
                </Badge>
                <span className="font-medium text-sm">{strategy.strategy}</span>
              </div>
              <span className="text-sm font-bold text-primary">
                {(strategy.accuracy * 100).toFixed(1)}%
              </span>
            </div>

            {/* Accuracy Progress Bar */}
            <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-primary to-secondary rounded-full transition-all"
                // FIX: Correctly setting the width style using curly braces for the object
                // and a template literal (backticks) for the string value
                style={{ width: `${strategy.accuracy * 100}%` }}
              />
            </div>

            {/* Footer: F1 Score */}
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>F1 Score: {(strategy.f1_score * 100).toFixed(1)}%</span>
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
};

export default StrategyComparison;
