import { FOUNDATIONS, SUB_FOUNDATIONS, TASK_ARCHETYPES } from "../../../shared/tks";

export function FoundationBadge({ foundationId, size = "sm" }: { foundationId: number | null; size?: "sm" | "md" | "lg" }) {
  const foundation = FOUNDATIONS.find(f => f.id === foundationId);
  if (!foundation) return null;

  const sizeClasses = {
    sm: "px-2 py-0.5 text-xs",
    md: "px-3 py-1 text-sm",
    lg: "px-4 py-1.5 text-base",
  };

  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full font-medium border ${sizeClasses[size]}`}
      style={{
        backgroundColor: foundation.color + "15",
        borderColor: foundation.color + "30",
        color: foundation.color,
      }}
    >
      <span>{foundation.symbol}</span>
      <span>{foundation.name}</span>
    </span>
  );
}

export function SubFoundationBadge({ subFoundationId }: { subFoundationId: string | null }) {
  const sf = SUB_FOUNDATIONS.find(s => s.id === subFoundationId);
  if (!sf) return null;
  const foundation = FOUNDATIONS.find(f => f.id === sf.foundationId);

  return (
    <span
      className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium border"
      style={{
        backgroundColor: (foundation?.color || "#888") + "10",
        borderColor: (foundation?.color || "#888") + "25",
        color: foundation?.color || "#888",
      }}
    >
      {sf.name}
    </span>
  );
}

export function ArchetypeBadge({ archetype }: { archetype: string | null }) {
  const arch = TASK_ARCHETYPES.find(a => a.id === archetype);
  if (!arch) return null;

  return (
    <span
      className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium border"
      style={{
        backgroundColor: arch.color + "15",
        borderColor: arch.color + "30",
        color: arch.color,
      }}
    >
      <span>{arch.icon}</span>
      <span>{arch.name}</span>
    </span>
  );
}

export function ConfidenceBadge({ confidence }: { confidence: number | null }) {
  if (confidence === null || confidence === undefined) return null;
  const pct = Math.round(confidence * 100);
  const color = pct >= 80 ? "#4CAF50" : pct >= 50 ? "#C8962E" : "#E03C1F";

  return (
    <span
      className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium"
      style={{ color }}
    >
      {pct}% confidence
    </span>
  );
}
