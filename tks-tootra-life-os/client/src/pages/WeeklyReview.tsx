import { trpc } from "@/lib/trpc";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Loader2, BarChart3, Download, CheckCircle2, XCircle, Clock } from "lucide-react";
import { FOUNDATIONS } from "../../../shared/tks";
import { useState, useMemo, useCallback } from "react";
import { toast } from "sonner";

function getWeekBounds() {
  const now = new Date();
  const day = now.getDay();
  const diff = now.getDate() - day + (day === 0 ? -6 : 1);
  const weekStart = new Date(now.setDate(diff));
  weekStart.setHours(0, 0, 0, 0);
  const weekEnd = new Date(weekStart);
  weekEnd.setDate(weekEnd.getDate() + 7);
  return { weekStart, weekEnd };
}

export default function WeeklyReview() {
  const { weekStart, weekEnd } = useMemo(() => getWeekBounds(), []);

  const statsQuery = trpc.analytics.weeklyStats.useQuery({
    weekStart: weekStart.toISOString(),
    weekEnd: weekEnd.toISOString(),
  });

  const historyQuery = trpc.analytics.reviewHistory.useQuery();
  const [exportFormat, setExportFormat] = useState<"csv" | "json">("json");
  const [exportEnabled, setExportEnabled] = useState(false);
  const exportQuery = trpc.analytics.export.useQuery(
    { format: exportFormat },
    {
      enabled: exportEnabled,
      retry: false,
      staleTime: 0,
    }
  );
  const saveReview = trpc.analytics.saveReview.useMutation({
    onSuccess: () => { historyQuery.refetch(); toast.success("Review snapshot saved"); },
  });

  const stats = statsQuery.data;
  const history = historyQuery.data || [];

  // When export data arrives, trigger download
  const lastExportRef = useMemo(() => ({ triggered: false }), []);
  if (exportQuery.data && exportEnabled && !lastExportRef.triggered) {
    lastExportRef.triggered = true;
    const { data: content, contentType } = exportQuery.data;
    if (content) {
      const blob = new Blob([content], { type: contentType });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `tks-export-${new Date().toISOString().slice(0, 10)}.${exportFormat}`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success(`Exported as ${exportFormat.toUpperCase()}`);
    }
    setExportEnabled(false);
  }

  const handleExport = useCallback((format: "csv" | "json") => {
    lastExportRef.triggered = false;
    setExportFormat(format);
    setExportEnabled(true);
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
            <BarChart3 className="h-6 w-6 text-tks-gold" />
            Weekly Review
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            Week of {weekStart.toLocaleDateString()} — {weekEnd.toLocaleDateString()}
          </p>
        </div>
        <div className="flex gap-2">
          <Button size="sm" variant="outline" className="text-xs" onClick={() => handleExport("csv")}>
            <Download className="h-3 w-3 mr-1" /> CSV
          </Button>
          <Button size="sm" variant="outline" className="text-xs" onClick={() => handleExport("json")}>
            <Download className="h-3 w-3 mr-1" /> JSON
          </Button>
        </div>
      </div>

      {statsQuery.isLoading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-6 w-6 animate-spin text-tks-gold" />
        </div>
      ) : (
        <>
          {/* Summary Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Card>
              <CardContent className="py-4 px-4 text-center">
                <Clock className="h-5 w-5 text-tks-blue mx-auto mb-2" />
                <div className="text-2xl font-bold text-foreground">{stats?.total || 0}</div>
                <p className="text-xs text-muted-foreground">Total Tasks</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="py-4 px-4 text-center">
                <CheckCircle2 className="h-5 w-5 text-green-400 mx-auto mb-2" />
                <div className="text-2xl font-bold text-green-400">{stats?.completed || 0}</div>
                <p className="text-xs text-muted-foreground">Completed</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="py-4 px-4 text-center">
                <XCircle className="h-5 w-5 text-muted-foreground mx-auto mb-2" />
                <div className="text-2xl font-bold text-muted-foreground">{stats?.dropped || 0}</div>
                <p className="text-xs text-muted-foreground">Dropped</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="py-4 px-4 text-center">
                <BarChart3 className="h-5 w-5 text-tks-gold mx-auto mb-2" />
                <div className="text-2xl font-bold text-tks-gold">
                  {stats && stats.total > 0 ? Math.round((stats.completed / stats.total) * 100) : 0}%
                </div>
                <p className="text-xs text-muted-foreground">Completion Rate</p>
              </CardContent>
            </Card>
          </div>

          {/* Foundation Balance This Week */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">Foundation Balance This Week</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {FOUNDATIONS.map(f => {
                  const fData = (stats?.byFoundation as any)?.[f.id] || { total: 0, completed: 0 };
                  const pct = stats && stats.total > 0 ? (fData.total / stats.total) * 100 : 0;
                  return (
                    <div key={f.id} className="flex items-center gap-3">
                      <span className="w-6 text-center">{f.symbol}</span>
                      <span className="w-24 text-xs font-medium truncate" style={{ color: f.color }}>
                        {f.name}
                      </span>
                      <div className="flex-1 h-3 rounded-full bg-secondary/30 overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all"
                          style={{ width: `${Math.min(pct, 100)}%`, backgroundColor: f.color }}
                        />
                      </div>
                      <span className="w-8 text-xs text-right text-muted-foreground">{fData.total}</span>
                      <span className="w-16 text-xs text-right text-muted-foreground">
                        {fData.completed}/{fData.total}
                      </span>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>

          {/* Drift Analysis */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">Drift Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              {stats && stats.total > 0 ? (
                <div className="space-y-2">
                  {FOUNDATIONS.map(f => {
                    const fData = (stats.byFoundation as any)?.[f.id] || { total: 0, completed: 0 };
                    const actual = stats.total > 0 ? (fData.total / stats.total) * 100 : 0;
                    const ideal = 100 / 7;
                    const drift = actual - ideal;
                    if (Math.abs(drift) < 3) return null;
                    return (
                      <div key={f.id} className="flex items-center gap-2 text-xs">
                        <span style={{ color: f.color }}>{f.symbol} {f.name}</span>
                        <span className={drift > 0 ? "text-tks-flame" : "text-blue-400"}>
                          {drift > 0 ? `+${drift.toFixed(0)}% over-indexed` : `${drift.toFixed(0)}% under-indexed`}
                        </span>
                      </div>
                    );
                  }).filter(Boolean)}
                  {FOUNDATIONS.every(f => {
                    const fData = (stats.byFoundation as any)?.[f.id] || { total: 0 };
                    const actual = stats.total > 0 ? (fData.total / stats.total) * 100 : 0;
                    return Math.abs(actual - 100 / 7) < 3;
                  }) && (
                    <p className="text-xs text-green-400">Well balanced this week — no significant drift detected.</p>
                  )}
                </div>
              ) : (
                <p className="text-xs text-muted-foreground">No data for drift analysis this week.</p>
              )}
            </CardContent>
          </Card>

          {/* Save Snapshot */}
          <Button
            className="bg-tks-gold hover:bg-tks-gold-light text-background"
            onClick={() => saveReview.mutate({ weekStart: weekStart.toISOString() })}
            disabled={saveReview.isPending}
          >
            {saveReview.isPending ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
            Save Weekly Snapshot
          </Button>

          {/* History */}
          {history.length > 0 && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium">Review History</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {history.map((snap: any) => (
                    <div key={snap.id} className="flex items-center justify-between py-2 border-b border-border last:border-0">
                      <span className="text-xs text-muted-foreground">
                        Week of {new Date(snap.weekStart).toLocaleDateString()}
                      </span>
                      <div className="flex items-center gap-3 text-xs">
                        <span>{snap.totalTasks} tasks</span>
                        <span className="text-green-400">{snap.completedTasks} done</span>
                        <span className="text-tks-gold">
                          {snap.completionRate != null ? Math.round(snap.completionRate * 100) : 0}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
