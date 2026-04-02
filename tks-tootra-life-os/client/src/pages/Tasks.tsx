import { trpc } from "@/lib/trpc";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useState } from "react";
import { FoundationBadge, ArchetypeBadge } from "@/components/FoundationBadge";
import { DWPCompact } from "@/components/DWPDisplay";
import { useLocation } from "wouter";
import { Loader2, CheckCircle2, Play, XCircle, RotateCcw, CalendarCheck, Send, Flame, ImagePlay } from "lucide-react";
import { TASK_STATUSES } from "../../../shared/tks";
import { toast } from "sonner";
import VoiceRecorder from "@/components/VoiceRecorder";

const STATUS_CONFIG: Record<string, { label: string; color: string; icon: any }> = {
  inbox: { label: "Inbox", color: "#C8962E", icon: CalendarCheck },
  scheduled: { label: "Scheduled", color: "#2B4E9B", icon: CalendarCheck },
  active: { label: "Active", color: "#4CAF50", icon: Play },
  done: { label: "Done", color: "#388E3C", icon: CheckCircle2 },
  dropped: { label: "Dropped", color: "#9E9E9E", icon: XCircle },
};

export default function Tasks() {
  const [, setLocation] = useLocation();
  const [activeTab, setActiveTab] = useState("all");
  const [quickInput, setQuickInput] = useState("");
  const [isQuickCapturing, setIsQuickCapturing] = useState(false);

  const tasksQuery = trpc.tasks.list.useQuery(activeTab === "all" ? {} : { status: activeTab });
  const modeQuery = trpc.settings.getMode.useQuery();
  const isWorkSprint = modeQuery.data?.mode === "work_sprint";

  const createTask = trpc.tasks.create.useMutation({
    onSuccess: () => {
      setQuickInput("");
      setIsQuickCapturing(false);
      tasksQuery.refetch();
      toast.success("Task captured");
    },
    onError: (err) => {
      setIsQuickCapturing(false);
      toast.error("Failed: " + err.message);
    },
  });

  const updateTask = trpc.tasks.update.useMutation({
    onSuccess: () => { tasksQuery.refetch(); toast.success("Task updated"); },
  });
  const deleteTask = trpc.tasks.delete.useMutation({
    onSuccess: () => { tasksQuery.refetch(); toast.success("Task deleted"); },
  });

  const handleQuickCapture = () => {
    if (!quickInput.trim()) return;
    setIsQuickCapturing(true);
    createTask.mutate({ rawInput: quickInput.trim() });
  };

  const tasks = tasksQuery.data || [];

  const getNextActions = (status: string) => {
    switch (status) {
      case "inbox": return [
        { label: "Schedule", status: "scheduled", icon: CalendarCheck, color: "text-blue-400" },
        { label: "Drop", status: "dropped", icon: XCircle, color: "text-muted-foreground" },
      ];
      case "scheduled": return [
        { label: "Start", status: "active", icon: Play, color: "text-green-400" },
        { label: "Back to Inbox", status: "inbox", icon: RotateCcw, color: "text-muted-foreground" },
      ];
      case "active": return [
        { label: "Done", status: "done", icon: CheckCircle2, color: "text-green-400" },
        { label: "Drop", status: "dropped", icon: XCircle, color: "text-muted-foreground" },
      ];
      default: return [];
    }
  };

  // Sprint mode: one-click quick actions
  const getSprintActions = (status: string) => {
    switch (status) {
      case "inbox": return [
        { label: "Start", status: "active", icon: Play, color: "text-green-400" },
        { label: "Drop", status: "dropped", icon: XCircle, color: "text-muted-foreground" },
      ];
      case "scheduled": return [
        { label: "Start", status: "active", icon: Play, color: "text-green-400" },
      ];
      case "active": return [
        { label: "Done", status: "done", icon: CheckCircle2, color: "text-green-400" },
      ];
      default: return [];
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
            {isWorkSprint && <Flame className="h-5 w-5 text-tks-flame" />}
            Tasks
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            {isWorkSprint ? "Focus. Execute. Ship." : "Manage your life execution pipeline"}
          </p>
        </div>
      </div>

      {/* Quick Capture Bar (always visible, optimized for sprint) */}
      <div className="flex gap-2">
        <input
          type="text"
          value={quickInput}
          onChange={(e) => setQuickInput(e.target.value)}
          placeholder={isWorkSprint ? "Quick capture — type and hit Enter" : "Quick capture a new task..."}
          className="flex-1 h-10 rounded-lg border border-input bg-secondary/30 px-4 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-tks-gold/50"
          onKeyDown={(e) => {
            if (e.key === "Enter") handleQuickCapture();
          }}
        />
        <VoiceRecorder
          compact
          onTranscription={(text) => setQuickInput(prev => prev ? prev + " " + text : text)}
        />
        <Button
          onClick={handleQuickCapture}
          disabled={!quickInput.trim() || isQuickCapturing}
          className="bg-tks-gold hover:bg-tks-gold-light text-background h-10"
          size="sm"
        >
          {isQuickCapturing ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Send className="h-4 w-4" />
          )}
        </Button>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="bg-secondary/50">
          <TabsTrigger value="all">All</TabsTrigger>
          {TASK_STATUSES.map(s => (
            <TabsTrigger key={s} value={s} className="capitalize">{s}</TabsTrigger>
          ))}
        </TabsList>

        <TabsContent value={activeTab} className="mt-4">
          {tasksQuery.isLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-6 w-6 animate-spin text-tks-gold" />
            </div>
          ) : tasks.length === 0 ? (
            <Card className="border-dashed">
              <CardContent className="flex flex-col items-center justify-center py-12 text-center">
                <p className="text-sm text-muted-foreground">No tasks found</p>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-2">
              {tasks.map((task: any) => {
                const statusCfg = STATUS_CONFIG[task.status] || STATUS_CONFIG.inbox;
                const actions = isWorkSprint ? getSprintActions(task.status) : getNextActions(task.status);
                return (
                  <Card
                    key={task.id}
                    className="hover:border-tks-gold/30 transition-all cursor-pointer"
                    onClick={() => setLocation(`/tasks/${task.id}`)}
                  >
                    <CardContent className="py-3 px-4">
                      <div className="flex items-center justify-between gap-3">
                        <div className="flex-1 min-w-0 space-y-1.5">
                          <div className="flex items-center gap-2">
                            <span
                              className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium uppercase tracking-wider shrink-0"
                              style={{ color: statusCfg.color, backgroundColor: statusCfg.color + "15" }}
                            >
                              {task.status}
                            </span>
                            <h3 className="text-sm font-medium truncate">{task.title}</h3>
                            {task.priorityScore != null && (
                              <span className="text-[10px] text-muted-foreground bg-secondary px-1.5 py-0.5 rounded shrink-0">
                                P{Math.round(task.priorityScore)}
                              </span>
                            )}
                          </div>
                          {/* Show TKS metadata only in Life OS mode */}
                          {!isWorkSprint && (
                            <div className="flex flex-wrap items-center gap-1.5">
                              <FoundationBadge foundationId={task.foundationId} />
                              <ArchetypeBadge archetype={task.archetype} />
                              {task.durationMinutes && (
                                <span className="text-[10px] text-muted-foreground">{task.durationMinutes}min</span>
                              )}
                            </div>
                          )}
                          {!isWorkSprint && (
                            <div className="w-32">
                              <DWPCompact
                                desireStage={task.desireStage}
                                wisdomStage={task.wisdomStage}
                                powerStage={task.powerStage}
                              />
                            </div>
                          )}
                        </div>
                        <div className="flex gap-1 shrink-0" onClick={(e) => e.stopPropagation()}>
                          <Button
                            size="sm"
                            variant="outline"
                            className="h-7 text-xs border-purple-500/30 text-purple-400 hover:bg-purple-500/10"
                            onClick={() => setLocation(`/tasks/${task.id}?scenarioize=true`)}
                            title="Scenarioize — visualize outcomes"
                          >
                            <ImagePlay className="h-3 w-3 mr-1" />
                            Scenarioize
                          </Button>
                          {actions.map((action) => (
                            <Button
                              key={action.status}
                              size="sm"
                              variant="outline"
                              className={`h-7 text-xs ${action.color}`}
                              onClick={() => updateTask.mutate({ id: task.id, status: action.status as any })}
                            >
                              <action.icon className="h-3 w-3 mr-1" />
                              {action.label}
                            </Button>
                          ))}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
