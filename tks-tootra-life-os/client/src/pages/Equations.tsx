import { trpc } from "@/lib/trpc";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { EquationDisplay, EquationCanonical } from "@/components/EquationChip";
import { FoundationBadge, ArchetypeBadge } from "@/components/FoundationBadge";
import { useLocation } from "wouter";
import { Loader2, Zap } from "lucide-react";
import { TOKEN_TYPES } from "../../../shared/tks";
import type { TokenType } from "../../../shared/tks";

export default function Equations() {
  const [, setLocation] = useLocation();
  const tasksQuery = trpc.tasks.list.useQuery({});
  const tasks = (tasksQuery.data || []).filter((t: any) => t.equationAst && (t.equationAst as any[]).length > 0);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
          <Zap className="h-6 w-6 text-tks-gold" />
          TKS Equations
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          Every task expressed as a TKS equation — the language of intentional action
        </p>
      </div>

      {/* Token Legend */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium">Token Types</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {Object.entries(TOKEN_TYPES).map(([key, def]) => (
              <span
                key={key}
                className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium border"
                style={{
                  backgroundColor: def.bgColor,
                  borderColor: def.color + "40",
                  color: def.color,
                }}
              >
                <span className="text-[10px]">{def.icon}</span>
                <span className="font-semibold">{key}</span>
                <span className="opacity-60 text-[10px]">{def.label}</span>
              </span>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Equations List */}
      {tasksQuery.isLoading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-6 w-6 animate-spin text-tks-gold" />
        </div>
      ) : tasks.length === 0 ? (
        <Card className="border-dashed">
          <CardContent className="flex flex-col items-center justify-center py-12 text-center">
            <Zap className="h-10 w-10 text-muted-foreground/30 mb-3" />
            <p className="text-sm text-muted-foreground">No equations yet</p>
            <p className="text-xs text-muted-foreground/60 mt-1">Capture tasks to generate TKS equations</p>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-3">
          {tasks.map((task: any) => (
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
                  <span className="text-[10px] text-muted-foreground capitalize">{task.status}</span>
                </div>
                <EquationDisplay tokens={task.equationAst as any[]} />
                {task.equationCanonical && (
                  <EquationCanonical equation={task.equationCanonical} />
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
