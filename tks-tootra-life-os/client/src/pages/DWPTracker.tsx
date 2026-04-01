import { trpc } from "@/lib/trpc";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { DWPFull, DWPMismatchAlert } from "@/components/DWPDisplay";
import { FoundationBadge, ArchetypeBadge } from "@/components/FoundationBadge";
import { Loader2, ClipboardList } from "lucide-react";
import { DWP_STAGES } from "../../../shared/tks";
import { toast } from "sonner";
import { useLocation } from "wouter";
import ReadAloud from "@/components/ReadAloud";

export default function DWPTracker() {
  const [, setLocation] = useLocation();
  const tasksQuery = trpc.tasks.list.useQuery({});
  const updateTask = trpc.tasks.update.useMutation({
    onSuccess: () => { tasksQuery.refetch(); toast.success("D/W/P stage updated"); },
  });

  const activeTasks = (tasksQuery.data || []).filter((t: any) =>
    ["inbox", "scheduled", "active"].includes(t.status)
  );

  // Aggregate D/W/P stats
  const dStats = { 0: 0, 1: 0, 2: 0, 3: 0 };
  const wStats = { 0: 0, 1: 0, 2: 0, 3: 0 };
  const pStats = { 0: 0, 1: 0, 2: 0, 3: 0 };
  for (const t of activeTasks as any[]) {
    const dIdx = DWP_STAGES.desire.findIndex(s => s.id === t.desireStage);
    const wIdx = DWP_STAGES.wisdom.findIndex(s => s.id === t.wisdomStage);
    const pIdx = DWP_STAGES.power.findIndex(s => s.id === t.powerStage);
    if (dIdx >= 0) (dStats as any)[dIdx]++;
    if (wIdx >= 0) (wStats as any)[wIdx]++;
    if (pIdx >= 0) (pStats as any)[pIdx]++;
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
          <ClipboardList className="h-6 w-6 text-tks-gold" />
          D/W/P Lifecycle Tracker
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          Monitor Desire, Wisdom, and Power across all active tasks
        </p>
      </div>

      {tasksQuery.isLoading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-6 w-6 animate-spin text-tks-gold" />
        </div>
      ) : (
        <>
          {/* Aggregate Overview */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {(["desire", "wisdom", "power"] as const).map(phase => {
              const stages = DWP_STAGES[phase];
              const stats = phase === "desire" ? dStats : phase === "wisdom" ? wStats : pStats;
              const colors = { desire: "#E03C1F", wisdom: "#2B4E9B", power: "#C8962E" };
              return (
                <Card key={phase}>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium capitalize" style={{ color: colors[phase] }}>
                      {phase} Distribution
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      {stages.map((stage, i) => {
                        const count = (stats as any)[i] || 0;
                        const pct = activeTasks.length > 0 ? (count / activeTasks.length) * 100 : 0;
                        return (
                          <div key={stage.id} className="flex items-center gap-2">
                            <span className="w-24 text-xs truncate">{stage.name}</span>
                            <div className="flex-1 h-2 rounded-full bg-secondary/30 overflow-hidden">
                              <div
                                className="h-full rounded-full"
                                style={{ width: `${pct}%`, backgroundColor: colors[phase] }}
                              />
                            </div>
                            <span className="w-6 text-xs text-right text-muted-foreground">{count}</span>
                          </div>
                        );
                      })}
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>

          {/* Per-Task D/W/P */}
          <div className="space-y-3">
            <h2 className="text-lg font-semibold">Per-Task Lifecycle</h2>
            {activeTasks.length === 0 ? (
              <Card className="border-dashed">
                <CardContent className="flex flex-col items-center justify-center py-12 text-center">
                  <p className="text-sm text-muted-foreground">No active tasks to track</p>
                </CardContent>
              </Card>
            ) : (
              activeTasks.map((task: any) => (
                <Card
                  key={task.id}
                  className="hover:border-tks-gold/30 transition-all cursor-pointer"
                  onClick={() => setLocation(`/tasks/${task.id}`)}
                >
                  <CardContent className="py-4 px-4 space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <h3 className="text-sm font-medium">{task.title}</h3>
                        <FoundationBadge foundationId={task.foundationId} />
                        <ArchetypeBadge archetype={task.archetype} />
                      </div>
                      <ReadAloud
                        compact
                        text={`${task.title}. Desire: ${task.desireStage?.replace(/_/g, " ") || "not set"}. Wisdom: ${task.wisdomStage?.replace(/_/g, " ") || "not set"}. Power: ${task.powerStage?.replace(/_/g, " ") || "not set"}. ${task.dwpDiagnosis || ""}`}
                      />
                    </div>

                    <DWPFull
                      desireStage={task.desireStage}
                      wisdomStage={task.wisdomStage}
                      powerStage={task.powerStage}
                      diagnosis={task.dwpDiagnosis}
                    />

                    <DWPMismatchAlert
                      desireStage={task.desireStage}
                      wisdomStage={task.wisdomStage}
                      powerStage={task.powerStage}
                    />

                    {/* Inline Stage Editors */}
                    <div className="flex gap-2 flex-wrap" onClick={(e) => e.stopPropagation()}>
                      <Select
                        value={task.desireStage || ""}
                        onValueChange={(v) => updateTask.mutate({ id: task.id, desireStage: v })}
                      >
                        <SelectTrigger className="h-7 w-36 text-xs">
                          <SelectValue placeholder="Desire" />
                        </SelectTrigger>
                        <SelectContent>
                          {DWP_STAGES.desire.map(s => (
                            <SelectItem key={s.id} value={s.id}>{s.name}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <Select
                        value={task.wisdomStage || ""}
                        onValueChange={(v) => updateTask.mutate({ id: task.id, wisdomStage: v })}
                      >
                        <SelectTrigger className="h-7 w-36 text-xs">
                          <SelectValue placeholder="Wisdom" />
                        </SelectTrigger>
                        <SelectContent>
                          {DWP_STAGES.wisdom.map(s => (
                            <SelectItem key={s.id} value={s.id}>{s.name}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <Select
                        value={task.powerStage || ""}
                        onValueChange={(v) => updateTask.mutate({ id: task.id, powerStage: v })}
                      >
                        <SelectTrigger className="h-7 w-36 text-xs">
                          <SelectValue placeholder="Power" />
                        </SelectTrigger>
                        <SelectContent>
                          {DWP_STAGES.power.map(s => (
                            <SelectItem key={s.id} value={s.id}>{s.name}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </CardContent>
                </Card>
              ))
            )}
          </div>
        </>
      )}
    </div>
  );
}
