import { trpc } from "@/lib/trpc";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Loader2, Grid3X3, AlertTriangle, TrendingDown } from "lucide-react";
import { FOUNDATIONS, SUB_FOUNDATIONS } from "../../../shared/tks";
import { useState } from "react";

export default function Heatmap() {
  const heatmapQuery = trpc.analytics.heatmap.useQuery();
  const [selectedFoundation, setSelectedFoundation] = useState<number | null>(null);

  const heatmapData = heatmapQuery.data || [];

  // Aggregate by foundation
  const foundationAgg: Record<number, { total: number; completed: number; active: number }> = {};
  for (const f of FOUNDATIONS) {
    foundationAgg[f.id] = { total: 0, completed: 0, active: 0 };
  }
  for (const row of heatmapData) {
    const fId = row.foundationId ?? 0;
    if (foundationAgg[fId]) {
      foundationAgg[fId].total += Number(row.total) || 0;
      foundationAgg[fId].completed += Number(row.completed) || 0;
      foundationAgg[fId].active += Number(row.active) || 0;
    }
  }

  const maxTotal = Math.max(1, ...Object.values(foundationAgg).map(a => a.total));

  // Detect overload and neglect
  const totalTasks = Object.values(foundationAgg).reduce((s, a) => s + a.total, 0);
  const avgTasks = totalTasks / 7;
  const overloaded = FOUNDATIONS.filter(f => foundationAgg[f.id].total > avgTasks * 1.8);
  const neglected = FOUNDATIONS.filter(f => foundationAgg[f.id].total < avgTasks * 0.3 && totalTasks > 3);

  // Sub-foundation data for selected foundation
  const selectedSubs = selectedFoundation
    ? SUB_FOUNDATIONS.filter(sf => sf.foundationId === selectedFoundation)
    : [];
  const subAgg: Record<string, { total: number; completed: number }> = {};
  for (const sf of selectedSubs) {
    subAgg[sf.id] = { total: 0, completed: 0 };
  }
  for (const row of heatmapData) {
    if (row.subFoundationId && subAgg[row.subFoundationId]) {
      subAgg[row.subFoundationId].total += Number(row.total) || 0;
      subAgg[row.subFoundationId].completed += Number(row.completed) || 0;
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
          <Grid3X3 className="h-6 w-6 text-tks-gold" />
          Foundation Heatmap
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          Visualize your life balance across the 7 Foundations
        </p>
      </div>

      {heatmapQuery.isLoading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-6 w-6 animate-spin text-tks-gold" />
        </div>
      ) : (
        <>
          {/* Alerts */}
          {(overloaded.length > 0 || neglected.length > 0) && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {overloaded.length > 0 && (
                <Card className="border-tks-flame/30 bg-tks-flame/5">
                  <CardContent className="py-3 px-4">
                    <div className="flex items-center gap-2 mb-1">
                      <AlertTriangle className="h-4 w-4 text-tks-flame" />
                      <span className="text-xs font-semibold text-tks-flame">Overloaded</span>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {overloaded.map(f => f.name).join(", ")} — disproportionately high task concentration
                    </p>
                  </CardContent>
                </Card>
              )}
              {neglected.length > 0 && (
                <Card className="border-blue-500/30 bg-blue-500/5">
                  <CardContent className="py-3 px-4">
                    <div className="flex items-center gap-2 mb-1">
                      <TrendingDown className="h-4 w-4 text-blue-400" />
                      <span className="text-xs font-semibold text-blue-400">Neglected</span>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {neglected.map(f => f.name).join(", ")} — needs more attention
                    </p>
                  </CardContent>
                </Card>
              )}
            </div>
          )}

          {/* 7-Cell Heatmap Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
            {FOUNDATIONS.map(f => {
              const agg = foundationAgg[f.id];
              const intensity = maxTotal > 0 ? agg.total / maxTotal : 0;
              const completionRate = agg.total > 0 ? agg.completed / agg.total : 0;
              const isSelected = selectedFoundation === f.id;

              return (
                <Card
                  key={f.id}
                  className={`cursor-pointer transition-all hover:scale-[1.02] ${isSelected ? "ring-2 ring-tks-gold" : ""}`}
                  style={{
                    borderColor: f.color + "40",
                    backgroundColor: f.color + Math.round(intensity * 20 + 5).toString(16).padStart(2, "0"),
                  }}
                  onClick={() => setSelectedFoundation(isSelected ? null : f.id)}
                >
                  <CardContent className="py-4 px-4 text-center">
                    <div className="text-2xl mb-1">{f.symbol}</div>
                    <h3 className="text-sm font-semibold" style={{ color: f.color }}>{f.name}</h3>
                    <p className="text-xs text-muted-foreground mt-1">{f.world} World</p>
                    <div className="mt-3 space-y-1">
                      <div className="text-xl font-bold" style={{ color: f.color }}>{agg.total}</div>
                      <p className="text-[10px] text-muted-foreground">tasks</p>
                      {/* Completion bar */}
                      <div className="h-1.5 rounded-full bg-secondary/50 overflow-hidden mt-2">
                        <div
                          className="h-full rounded-full transition-all"
                          style={{
                            width: `${completionRate * 100}%`,
                            backgroundColor: f.color,
                          }}
                        />
                      </div>
                      <p className="text-[10px] text-muted-foreground">
                        {Math.round(completionRate * 100)}% complete
                      </p>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>

          {/* Sub-Foundation Drill-Down */}
          {selectedFoundation && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium flex items-center gap-2">
                  {FOUNDATIONS.find(f => f.id === selectedFoundation)?.symbol}
                  {FOUNDATIONS.find(f => f.id === selectedFoundation)?.name} — Sub-Foundations
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-3">
                  {selectedSubs.map(sf => {
                    const data = subAgg[sf.id] || { total: 0, completed: 0 };
                    const fColor = FOUNDATIONS.find(f => f.id === sf.foundationId)?.color || "#888";
                    return (
                      <div key={sf.id} className="p-3 rounded-lg border" style={{ borderColor: fColor + "20" }}>
                        <div className="flex items-center justify-between">
                          <span className="text-sm font-medium" style={{ color: fColor }}>{sf.name}</span>
                          <span className="text-xs text-muted-foreground">{data.total} tasks</span>
                        </div>
                        <p className="text-xs text-muted-foreground mt-1">{sf.description}</p>
                        <div className="h-1 rounded-full bg-secondary/50 overflow-hidden mt-2">
                          <div
                            className="h-full rounded-full"
                            style={{
                              width: `${data.total > 0 ? (data.completed / data.total) * 100 : 0}%`,
                              backgroundColor: fColor,
                            }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Balance Analytics */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">Balance Analytics</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {FOUNDATIONS.map(f => {
                  const agg = foundationAgg[f.id];
                  const pct = totalTasks > 0 ? (agg.total / totalTasks) * 100 : 0;
                  const ideal = 100 / 7;
                  const deviation = pct - ideal;
                  return (
                    <div key={f.id} className="flex items-center gap-3">
                      <span className="w-24 text-xs font-medium truncate" style={{ color: f.color }}>
                        {f.name}
                      </span>
                      <div className="flex-1 h-2 rounded-full bg-secondary/30 overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all"
                          style={{ width: `${Math.min(pct, 100)}%`, backgroundColor: f.color }}
                        />
                      </div>
                      <span className="w-12 text-xs text-right text-muted-foreground">
                        {pct.toFixed(0)}%
                      </span>
                      <span className={`w-16 text-xs text-right ${deviation > 5 ? "text-tks-flame" : deviation < -5 ? "text-blue-400" : "text-green-400"}`}>
                        {deviation > 0 ? "+" : ""}{deviation.toFixed(0)}%
                      </span>
                    </div>
                  );
                })}
              </div>
              <p className="text-[10px] text-muted-foreground mt-3">
                Ideal distribution: ~{(100 / 7).toFixed(0)}% per foundation. Deviation shows over/under-allocation.
              </p>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
