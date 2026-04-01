import { trpc } from "@/lib/trpc";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { EquationDisplay, EquationCanonical } from "@/components/EquationChip";
import { FoundationBadge, SubFoundationBadge, ArchetypeBadge, ConfidenceBadge } from "@/components/FoundationBadge";
import { DWPFull, DWPMismatchAlert } from "@/components/DWPDisplay";
import { useLocation, useParams } from "wouter";
import { Loader2, ArrowLeft, CheckCircle2, Play, XCircle, CalendarCheck, Trash2 } from "lucide-react";
import { toast } from "sonner";
import ReadAloud from "@/components/ReadAloud";
import Scenarioize from "@/components/Scenarioize";

function buildReadAloudText(task: any, powerItems: any[]): string {
  const parts: string[] = [];
  parts.push(`Task: ${task.title}.`);
  if (task.description) parts.push(task.description);
  if (task.archetype) parts.push(`This is a ${task.archetype} task.`);
  if (task.urgency) parts.push(`Urgency is ${task.urgency}.`);
  if (task.desireStage) parts.push(`Desire stage: ${task.desireStage.replace(/_/g, " ")}.`);
  if (task.wisdomStage) parts.push(`Wisdom stage: ${task.wisdomStage.replace(/_/g, " ")}.`);
  if (task.powerStage) parts.push(`Power stage: ${task.powerStage.replace(/_/g, " ")}.`);
  if (task.dwpDiagnosis) parts.push(`Diagnosis: ${task.dwpDiagnosis}`);
  if (powerItems.length > 0) {
    const haves = powerItems.filter((p: any) => p.status === "have").map((p: any) => p.category);
    const lacks = powerItems.filter((p: any) => p.status === "lack").map((p: any) => p.category);
    if (haves.length) parts.push(`You have: ${haves.join(", ")}.`);
    if (lacks.length) parts.push(`You lack: ${lacks.join(", ")}.`);
  }
  if (task.equationCanonical) parts.push(`TKS Equation: ${task.equationCanonical}`);
  return parts.join(" ");
}

function buildDWPReadAloud(task: any): string {
  const parts: string[] = [];
  if (task.desireStage) parts.push(`Desire stage is ${task.desireStage.replace(/_/g, " ")}`);
  if (task.wisdomStage) parts.push(`Wisdom stage is ${task.wisdomStage.replace(/_/g, " ")}`);
  if (task.powerStage) parts.push(`Power stage is ${task.powerStage.replace(/_/g, " ")}`);
  if (task.dwpDiagnosis) parts.push(`Diagnosis: ${task.dwpDiagnosis}`);
  return parts.join(". ") || "No lifecycle data available.";
}

export default function TaskDetail() {
  const params = useParams<{ id: string }>();
  const [, setLocation] = useLocation();
  const taskId = Number(params.id);

  const taskQuery = trpc.tasks.get.useQuery({ id: taskId }, { enabled: !!taskId });
  const powerQuery = trpc.power.getForTask.useQuery({ taskId }, { enabled: !!taskId });
  const updateTask = trpc.tasks.update.useMutation({
    onSuccess: () => { taskQuery.refetch(); toast.success("Task updated"); },
  });
  const deleteTask = trpc.tasks.delete.useMutation({
    onSuccess: () => { setLocation("/tasks"); toast.success("Task deleted"); },
  });

  const task = taskQuery.data as any;
  const powerItems = powerQuery.data || [];

  if (taskQuery.isLoading) {
    return (
      <div className="flex items-center justify-center py-24">
        <Loader2 className="h-6 w-6 animate-spin text-tks-gold" />
      </div>
    );
  }

  if (!task) {
    return (
      <div className="space-y-4">
        <Button variant="outline" size="sm" onClick={() => setLocation("/tasks")}>
          <ArrowLeft className="h-4 w-4 mr-1" /> Back
        </Button>
        <p className="text-sm text-muted-foreground">Task not found</p>
      </div>
    );
  }

  const STATUS_ACTIONS: Record<string, Array<{ label: string; status: string; icon: any; color: string }>> = {
    inbox: [
      { label: "Schedule", status: "scheduled", icon: CalendarCheck, color: "text-blue-400" },
      { label: "Drop", status: "dropped", icon: XCircle, color: "text-muted-foreground" },
    ],
    scheduled: [
      { label: "Start", status: "active", icon: Play, color: "text-green-400" },
      { label: "Back to Inbox", status: "inbox", icon: ArrowLeft, color: "text-muted-foreground" },
    ],
    active: [
      { label: "Complete", status: "done", icon: CheckCircle2, color: "text-green-400" },
      { label: "Drop", status: "dropped", icon: XCircle, color: "text-muted-foreground" },
    ],
    done: [],
    dropped: [
      { label: "Reopen", status: "inbox", icon: ArrowLeft, color: "text-tks-gold" },
    ],
  };

  const actions = STATUS_ACTIONS[task.status] || [];

  const POWER_STATUS_COLORS: Record<string, string> = {
    have: "#4CAF50",
    lack: "#E03C1F",
    partial: "#C8962E",
  };

  return (
    <div className="space-y-6 max-w-3xl">
      {/* Back */}
      <Button variant="outline" size="sm" onClick={() => setLocation("/tasks")}>
        <ArrowLeft className="h-4 w-4 mr-1" /> Back to Tasks
      </Button>

      {/* Header */}
      <div className="space-y-3">
        <div className="flex items-start justify-between gap-4">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <h1 className="text-2xl font-bold tracking-tight">{task.title}</h1>
              <ReadAloud
                compact
                text={buildReadAloudText(task, powerItems)}
              />
            </div>
            {task.description && (
              <p className="text-sm text-muted-foreground">{task.description}</p>
            )}
          </div>
          <div className="flex gap-1 shrink-0">
            {actions.map((action) => (
              <Button
                key={action.status}
                size="sm"
                variant="outline"
                className={`text-xs ${action.color}`}
                onClick={() => updateTask.mutate({ id: task.id, status: action.status as any })}
              >
                <action.icon className="h-3 w-3 mr-1" />
                {action.label}
              </Button>
            ))}
            <Button
              size="sm"
              variant="outline"
              className="text-xs text-destructive"
              onClick={() => { if (confirm("Delete this task?")) deleteTask.mutate({ id: task.id }); }}
            >
              <Trash2 className="h-3 w-3" />
            </Button>
          </div>
        </div>

        {/* Badges */}
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs px-2 py-0.5 rounded uppercase font-medium bg-secondary text-secondary-foreground">
            {task.status}
          </span>
          <FoundationBadge foundationId={task.foundationId} size="md" />
          <SubFoundationBadge subFoundationId={task.subFoundationId} />
          <ArchetypeBadge archetype={task.archetype} />
          <ConfidenceBadge confidence={task.mappingConfidence} />
          {task.priorityScore != null && (
            <span className="text-xs bg-tks-gold/20 text-tks-gold px-2 py-0.5 rounded">
              Priority: {Math.round(task.priorityScore)}
            </span>
          )}
          {task.urgency && (
            <span className={`text-xs px-2 py-0.5 rounded capitalize ${
              task.urgency === "critical" ? "bg-destructive/20 text-destructive" :
              task.urgency === "high" ? "bg-tks-flame/20 text-tks-flame" :
              task.urgency === "medium" ? "bg-tks-gold/20 text-tks-gold" :
              "bg-secondary text-muted-foreground"
            }`}>
              {task.urgency}
            </span>
          )}
          {task.durationMinutes && (
            <span className="text-xs text-muted-foreground">{task.durationMinutes} min</span>
          )}
        </div>
      </div>

      {/* Raw Input */}
      {task.rawInput && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs font-medium text-muted-foreground">Original Input</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm italic text-muted-foreground">"{task.rawInput}"</p>
          </CardContent>
        </Card>
      )}

      {/* TKS Equation */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium">TKS Equation</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {task.equationAst && (
            <EquationDisplay tokens={task.equationAst as any[]} />
          )}
          {task.equationCanonical && (
            <EquationCanonical equation={task.equationCanonical} />
          )}
        </CardContent>
      </Card>

      {/* D/W/P Lifecycle */}
      <Card>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm font-medium">D/W/P Lifecycle</CardTitle>
            <ReadAloud
              compact
              text={buildDWPReadAloud(task)}
            />
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
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
        </CardContent>
      </Card>

      {/* Power Inventory */}
      {(powerItems as any[]).length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Power Inventory</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-2">
              {(powerItems as any[]).map((item: any) => (
                <div
                  key={item.id}
                  className="flex items-center gap-2 p-2 rounded-lg border text-xs"
                  style={{ borderColor: (POWER_STATUS_COLORS[item.status] || "#888") + "30" }}
                >
                  <span
                    className="w-2 h-2 rounded-full shrink-0"
                    style={{ backgroundColor: POWER_STATUS_COLORS[item.status] || "#888" }}
                  />
                  <span className="font-medium capitalize">{item.category}</span>
                  <span className="ml-auto capitalize text-muted-foreground">{item.status}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Scenarioize — Vision Engine */}
      <Scenarioize taskId={task.id} taskTitle={task.title} autoOpen={typeof window !== 'undefined' && new URLSearchParams(window.location.search).get('scenarioize') === 'true'} />

      {/* Full Read Aloud */}
      <Card>
        <CardContent className="py-3">
          <ReadAloud
            text={buildReadAloudText(task, powerItems)}
            label="Read Full Task Summary"
          />
        </CardContent>
      </Card>

      {/* Metadata */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-xs font-medium text-muted-foreground">Metadata</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-2 text-xs text-muted-foreground">
            <div>Created: {new Date(task.createdAt).toLocaleString()}</div>
            <div>Updated: {new Date(task.updatedAt).toLocaleString()}</div>
            {task.scheduledAt && <div>Scheduled: {new Date(task.scheduledAt).toLocaleString()}</div>}
            {task.completedAt && <div>Completed: {new Date(task.completedAt).toLocaleString()}</div>}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
