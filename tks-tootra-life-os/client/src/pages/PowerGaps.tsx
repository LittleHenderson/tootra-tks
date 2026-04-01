import { trpc } from "@/lib/trpc";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Loader2, Shield, CheckCircle2, XCircle, AlertCircle, Sparkles } from "lucide-react";
import { POWER_CATEGORIES } from "../../../shared/tks";
import { useState } from "react";
import { toast } from "sonner";

export default function PowerGaps() {
  const tasksQuery = trpc.tasks.list.useQuery({});
  const [selectedTaskId, setSelectedTaskId] = useState<number | null>(null);
  const [generatedSubtasks, setGeneratedSubtasks] = useState<Record<string, string[]>>({});

  const powerQuery = trpc.power.getForTask.useQuery(
    { taskId: selectedTaskId! },
    { enabled: !!selectedTaskId }
  );

  const updatePower = trpc.power.update.useMutation({
    onSuccess: () => { powerQuery.refetch(); toast.success("Power inventory updated"); },
  });

  const generateSubs = trpc.power.generateSubtasks.useMutation({
    onSuccess: (data, variables) => {
      setGeneratedSubtasks(prev => ({ ...prev, [variables.category]: data.subtasks }));
    },
  });

  const activeTasks = (tasksQuery.data || []).filter((t: any) => ["inbox", "scheduled", "active"].includes(t.status));
  const powerItems = powerQuery.data || [];

  const STATUS_ICONS: Record<string, { icon: any; color: string; label: string }> = {
    have: { icon: CheckCircle2, color: "#4CAF50", label: "Have" },
    lack: { icon: XCircle, color: "#E03C1F", label: "Lack" },
    partial: { icon: AlertCircle, color: "#C8962E", label: "Partial" },
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
          <Shield className="h-6 w-6 text-tks-gold" />
          Power Gap Analysis
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          Assess the 8 resource categories for each task and generate subtasks to close gaps
        </p>
      </div>

      {/* Task Selector */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium">Select a Task to Analyze</CardTitle>
        </CardHeader>
        <CardContent>
          {tasksQuery.isLoading ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : activeTasks.length === 0 ? (
            <p className="text-sm text-muted-foreground">No active tasks to analyze</p>
          ) : (
            <div className="flex flex-wrap gap-2">
              {activeTasks.map((task: any) => (
                <Button
                  key={task.id}
                  variant={selectedTaskId === task.id ? "default" : "outline"}
                  size="sm"
                  className={selectedTaskId === task.id ? "bg-tks-gold text-background" : ""}
                  onClick={() => setSelectedTaskId(task.id)}
                >
                  {task.title}
                </Button>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Power Inventory Grid */}
      {selectedTaskId && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {POWER_CATEGORIES.map(cat => {
            const item = powerItems.find((p: any) => p.category === cat.id);
            const status = item?.status || "lack";
            const statusCfg = STATUS_ICONS[status] || STATUS_ICONS.lack;
            const StatusIcon = statusCfg.icon;
            const subs = generatedSubtasks[cat.id] || [];
            const selectedTask = activeTasks.find((t: any) => t.id === selectedTaskId);

            return (
              <Card key={cat.id} className="border" style={{ borderColor: statusCfg.color + "30" }}>
                <CardContent className="py-4 px-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="text-lg">{cat.icon}</span>
                      <div>
                        <h3 className="text-sm font-medium">{cat.name}</h3>
                        <p className="text-[10px] text-muted-foreground">{cat.description}</p>
                      </div>
                    </div>
                    <StatusIcon className="h-5 w-5" style={{ color: statusCfg.color }} />
                  </div>

                  {/* Status Toggle */}
                  <div className="flex gap-1">
                    {(["have", "partial", "lack"] as const).map(s => {
                      const cfg = STATUS_ICONS[s];
                      return (
                        <Button
                          key={s}
                          size="sm"
                          variant={status === s ? "default" : "outline"}
                          className={`h-7 text-xs flex-1 ${status === s ? "" : ""}`}
                          style={status === s ? { backgroundColor: cfg.color, color: "#fff" } : {}}
                          onClick={() => updatePower.mutate({
                            taskId: selectedTaskId,
                            category: cat.id as any,
                            status: s,
                          })}
                        >
                          {cfg.label}
                        </Button>
                      );
                    })}
                  </div>

                  {/* Notes */}
                  {item?.notes && (
                    <p className="text-xs text-muted-foreground italic">{item.notes}</p>
                  )}

                  {/* Generate Subtasks */}
                  {status !== "have" && (
                    <div>
                      <Button
                        size="sm"
                        variant="outline"
                        className="h-7 text-xs w-full border-tks-gold/30 text-tks-gold hover:bg-tks-gold/10"
                        onClick={() => generateSubs.mutate({
                          taskTitle: selectedTask?.title || "",
                          category: cat.id,
                        })}
                        disabled={generateSubs.isPending}
                      >
                        {generateSubs.isPending ? (
                          <Loader2 className="h-3 w-3 animate-spin mr-1" />
                        ) : (
                          <Sparkles className="h-3 w-3 mr-1" />
                        )}
                        Generate Subtasks to Close Gap
                      </Button>
                      {subs.length > 0 && (
                        <ul className="mt-2 space-y-1">
                          {subs.map((sub, i) => (
                            <li key={i} className="text-xs text-muted-foreground flex items-start gap-1.5">
                              <span className="text-tks-gold mt-0.5">-</span>
                              {sub}
                            </li>
                          ))}
                        </ul>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
