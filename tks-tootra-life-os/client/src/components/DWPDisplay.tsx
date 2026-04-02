import { DWP_STAGES } from "../../../shared/tks";

const PHASE_COLORS = {
  desire: { bg: "#E03C1F", light: "#E03C1F20" },
  wisdom: { bg: "#2B4E9B", light: "#2B4E9B20" },
  power: { bg: "#C8962E", light: "#C8962E20" },
};

function StageIndicator({
  phase,
  currentStageId,
  compact = false,
}: {
  phase: "desire" | "wisdom" | "power";
  currentStageId: string | null;
  compact?: boolean;
}) {
  const stages = DWP_STAGES[phase];
  const colors = PHASE_COLORS[phase];
  const currentIndex = stages.findIndex(s => s.id === currentStageId);

  return (
    <div className={compact ? "" : "space-y-1.5"}>
      {!compact && (
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold uppercase tracking-wider" style={{ color: colors.bg }}>
            {phase}
          </span>
          <span className="text-xs text-muted-foreground">
            {currentIndex >= 0 ? stages[currentIndex].name : "Unknown"}
          </span>
        </div>
      )}
      <div className="flex gap-1">
        {stages.map((stage, i) => {
          const isActive = i <= currentIndex;
          const isCurrent = stage.id === currentStageId;
          return (
            <div
              key={stage.id}
              className={`relative transition-all ${compact ? "h-1.5 flex-1 rounded-full" : "h-2 flex-1 rounded-full"}`}
              style={{
                backgroundColor: isActive ? colors.bg : colors.light,
                boxShadow: isCurrent ? `0 0 8px ${colors.bg}40` : undefined,
              }}
              title={`${stage.name}: ${stage.meaning}`}
            />
          );
        })}
      </div>
    </div>
  );
}

export function DWPCompact({
  desireStage,
  wisdomStage,
  powerStage,
}: {
  desireStage: string | null;
  wisdomStage: string | null;
  powerStage: string | null;
}) {
  return (
    <div className="space-y-1">
      <StageIndicator phase="desire" currentStageId={desireStage} compact />
      <StageIndicator phase="wisdom" currentStageId={wisdomStage} compact />
      <StageIndicator phase="power" currentStageId={powerStage} compact />
    </div>
  );
}

export function DWPFull({
  desireStage,
  wisdomStage,
  powerStage,
  diagnosis,
}: {
  desireStage: string | null;
  wisdomStage: string | null;
  powerStage: string | null;
  diagnosis?: string | null;
}) {
  return (
    <div className="space-y-4">
      <StageIndicator phase="desire" currentStageId={desireStage} />
      <StageIndicator phase="wisdom" currentStageId={wisdomStage} />
      <StageIndicator phase="power" currentStageId={powerStage} />
      {diagnosis && (
        <div className="mt-3 p-3 rounded-lg bg-tks-flame/5 border border-tks-flame/20">
          <p className="text-xs font-medium text-tks-flame mb-1">D/W/P Diagnosis</p>
          <p className="text-xs text-muted-foreground">{diagnosis}</p>
        </div>
      )}
    </div>
  );
}

export function DWPMismatchAlert({
  desireStage,
  wisdomStage,
  powerStage,
}: {
  desireStage: string | null;
  wisdomStage: string | null;
  powerStage: string | null;
}) {
  const dIdx = DWP_STAGES.desire.findIndex(s => s.id === desireStage);
  const wIdx = DWP_STAGES.wisdom.findIndex(s => s.id === wisdomStage);
  const pIdx = DWP_STAGES.power.findIndex(s => s.id === powerStage);

  const mismatches: string[] = [];
  if (dIdx >= 2 && wIdx <= 0) mismatches.push("High desire but low wisdom — you want it but don't know how");
  if (dIdx >= 2 && pIdx <= 0) mismatches.push("High desire but no power — motivated but lacking resources");
  if (wIdx >= 2 && dIdx <= 0) mismatches.push("High wisdom but low desire — you know how but lack motivation");
  if (pIdx >= 2 && dIdx <= 0) mismatches.push("Resources available but no desire — everything ready but no drive");

  if (mismatches.length === 0) return null;

  return (
    <div className="p-3 rounded-lg bg-tks-flame/10 border border-tks-flame/30">
      <p className="text-xs font-semibold text-tks-flame mb-1">Mismatch Detected</p>
      {mismatches.map((m, i) => (
        <p key={i} className="text-xs text-muted-foreground">{m}</p>
      ))}
    </div>
  );
}
