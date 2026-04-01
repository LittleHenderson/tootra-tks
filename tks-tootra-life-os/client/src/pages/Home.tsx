import { useAuth } from "@/_core/hooks/useAuth";
import { trpc } from "@/lib/trpc";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useState } from "react";
import { FoundationBadge, ArchetypeBadge, ConfidenceBadge } from "@/components/FoundationBadge";
import { EquationDisplay } from "@/components/EquationChip";
import { DWPCompact } from "@/components/DWPDisplay";
import { useLocation } from "wouter";
import { Loader2, Send, Inbox, Zap, Triangle, ArrowRight, CheckCircle2, XCircle, Mic, ImagePlay } from "lucide-react";
import { FOUNDATIONS } from "../../../shared/tks";
import { toast } from "sonner";
import VoiceRecorder from "@/components/VoiceRecorder";
import ReadAloud from "@/components/ReadAloud";


export default function Home() {
  const { user } = useAuth();
  const [, setLocation] = useLocation();
  const [rawInput, setRawInput] = useState("");
  const [isCapturing, setIsCapturing] = useState(false);


  const tasksQuery = trpc.tasks.list.useQuery({ status: "inbox" });
  const createTask = trpc.tasks.create.useMutation({
    onSuccess: (data) => {
      setRawInput("");
      setIsCapturing(false);
      tasksQuery.refetch();
      toast.success(`Task mapped to ${FOUNDATIONS.find(f => f.id === data.task?.foundationId)?.name || "Unknown"} Foundation`);
    },
    onError: (err) => {
      setIsCapturing(false);
      toast.error("Failed to interpret task: " + err.message);
    },
  });

  const updateTask = trpc.tasks.update.useMutation({
    onSuccess: () => tasksQuery.refetch(),
  });

  const handleCapture = () => {
    if (!rawInput.trim()) return;
    setIsCapturing(true);
    createTask.mutate({ rawInput: rawInput.trim() });
  };

  const inboxTasks = tasksQuery.data || [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
            <Triangle className="h-6 w-6 text-tks-gold fill-tks-gold/10" />
            Welcome{user?.name ? `, ${user.name}` : ""}
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            Speak your intention. The system maps your life.
          </p>
        </div>
      </div>

      {/* Capture Input */}
      <Card className="border-tks-gold/20 bg-card">
        <CardContent className="pt-6">
          <div className="space-y-3">
            <label className="text-sm font-medium text-foreground">
              What do you need to do?
            </label>
            <div className="flex gap-2">
              <textarea
                value={rawInput}
                onChange={(e) => setRawInput(e.target.value)}
                placeholder="e.g. 'Study React hooks before building the dashboard' or 'Fix the leaky faucet — it started dripping again'"
                className="flex-1 min-h-[80px] rounded-lg border border-input bg-secondary/30 px-4 py-3 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-tks-gold/50 resize-none"
                onKeyDown={(e) => {
                  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) handleCapture();
                }}
              />
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <p className="text-xs text-muted-foreground">
                  Press <kbd className="px-1 py-0.5 rounded bg-secondary text-xs">Ctrl+Enter</kbd> to capture
                </p>
                <VoiceRecorder
                  onTranscription={(text) => setRawInput(prev => prev ? prev + " " + text : text)}
                />
              </div>
              <Button
                onClick={handleCapture}
                disabled={!rawInput.trim() || isCapturing}
                className="bg-tks-gold hover:bg-tks-gold-light text-background"
              >
                {isCapturing ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin mr-2" />
                    Interpreting...
                  </>
                ) : (
                  <>
                    <Send className="h-4 w-4 mr-2" />
                    Capture & Map
                  </>
                )}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Last Interpretation Result */}
      {createTask.data && (
        <Card className="border-tks-gold/30 bg-tks-gold/5">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium text-tks-gold flex items-center gap-2">
                <Zap className="h-4 w-4" />
                TKS Interpretation Complete
              </CardTitle>
              <ReadAloud
                compact
                text={`Task: ${createTask.data.task?.title || ""}. ${createTask.data.task?.description || ""}. Foundation: ${FOUNDATIONS.find(f => f.id === createTask.data.task?.foundationId)?.name || "Unknown"}. Archetype: ${createTask.data.task?.archetype || "Unknown"}. ${createTask.data.task?.dwpDiagnosis || ""}`}
              />
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex flex-wrap items-center gap-2">
              <FoundationBadge foundationId={createTask.data.task?.foundationId ?? null} size="md" />
              <ArchetypeBadge archetype={createTask.data.task?.archetype ?? null} />
              <ConfidenceBadge confidence={createTask.data.task?.mappingConfidence ?? null} />
            </div>
            {createTask.data.task?.equationAst ? (
              <EquationDisplay tokens={createTask.data.task.equationAst as any[]} />
            ) : null}
            <DWPCompact
              desireStage={createTask.data.task?.desireStage ?? null}
              wisdomStage={createTask.data.task?.wisdomStage ?? null}
              powerStage={createTask.data.task?.powerStage ?? null}
            />
          </CardContent>
        </Card>
      )}

      {/* Inbox */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <Inbox className="h-5 w-5 text-muted-foreground" />
            Inbox
            {inboxTasks.length > 0 && (
              <span className="text-xs bg-tks-gold/20 text-tks-gold px-2 py-0.5 rounded-full">
                {inboxTasks.length}
              </span>
            )}
          </h2>
          <Button variant="outline" size="sm" onClick={() => setLocation("/tasks")} className="text-xs">
            View All Tasks <ArrowRight className="h-3 w-3 ml-1" />
          </Button>
        </div>

        {tasksQuery.isLoading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="h-6 w-6 animate-spin text-tks-gold" />
          </div>
        ) : inboxTasks.length === 0 ? (
          <Card className="border-dashed">
            <CardContent className="flex flex-col items-center justify-center py-12 text-center">
              <Inbox className="h-10 w-10 text-muted-foreground/30 mb-3" />
              <p className="text-sm text-muted-foreground">Your inbox is empty</p>
              <p className="text-xs text-muted-foreground/60 mt-1">Capture a task above to get started</p>
            </CardContent>
          </Card>
        ) : (
          <div className="space-y-2">
            {inboxTasks.map((task: any) => (
              <Card
                key={task.id}
                className="hover:border-tks-gold/30 transition-all cursor-pointer"
                onClick={() => setLocation(`/tasks/${task.id}`)}
              >
                <CardContent className="py-3 px-4">
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex-1 min-w-0 space-y-2">
                      <div className="flex items-center gap-2">
                        <h3 className="text-sm font-medium truncate">{task.title}</h3>
                        {task.priorityScore != null && (
                          <span className="text-[10px] text-muted-foreground bg-secondary px-1.5 py-0.5 rounded">
                            P{Math.round(task.priorityScore)}
                          </span>
                        )}
                      </div>
                      <div className="flex flex-wrap items-center gap-1.5">
                        <FoundationBadge foundationId={task.foundationId} />
                        <ArchetypeBadge archetype={task.archetype} />
                      </div>
                      <div className="w-32">
                        <DWPCompact
                          desireStage={task.desireStage}
                          wisdomStage={task.wisdomStage}
                          powerStage={task.powerStage}
                        />
                      </div>
                    </div>
                    <div className="flex gap-1 shrink-0">
                      <Button
                        size="sm"
                        variant="outline"
                        className="h-7 text-xs border-purple-500/30 text-purple-400 hover:bg-purple-500/10"
                        onClick={(e) => {
                          e.stopPropagation();
                          setLocation(`/tasks/${task.id}?scenarioize=true`);
                        }}
                        title="Scenarioize — visualize outcomes"
                      >
                        <ImagePlay className="h-3 w-3 mr-1" />
                        Scenarioize
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        className="h-7 text-xs border-green-500/30 text-green-500 hover:bg-green-500/10"
                        onClick={(e) => {
                          e.stopPropagation();
                          updateTask.mutate({ id: task.id, status: "scheduled" });
                          toast.success("Task scheduled");
                        }}
                      >
                        <CheckCircle2 className="h-3 w-3 mr-1" />
                        Schedule
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        className="h-7 text-xs border-destructive/30 text-destructive hover:bg-destructive/10"
                        onClick={(e) => {
                          e.stopPropagation();
                          updateTask.mutate({ id: task.id, status: "dropped" });
                          toast.info("Task dropped");
                        }}
                      >
                        <XCircle className="h-3 w-3" />
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
