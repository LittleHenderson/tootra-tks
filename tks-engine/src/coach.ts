import { TksState, FoundationId, createDefaultState } from "./types";
import { RpmCheckResult, checkPrereqs } from "./rpm";
import {
  FOUNDATIONS,
  FoundationMeta,
  getDefaultPatternsForFoundation,
  NOETIC_PATTERNS,
  NoeticPattern,
} from "./semantics";
import {
  AiVitals,
  AiToTksSummary,
  AiToTksMappingOptions,
  applyAiVitalsToFoundations,
  summarizeAiVitals,
} from "./aiVitals";

export interface CoachSuggestion {
  goalFoundation: FoundationId;
  foundationMeta: FoundationMeta;
  rpm: RpmCheckResult;
  currentActivation: number;
  missingAcquisitions: number[];
  suggestedPatterns: NoeticPattern[];
  exampleExpressions: string[];
  narrative: string;
}

export interface AiCoachReport {
  vitals: AiVitals;
  summary: AiToTksSummary;
  foundationCoaches: Partial<Record<FoundationId, CoachSuggestion>>;
  narrative: string;
}

function sequencesEqual(a: number[], b: number[]): boolean {
  if (a.length !== b.length) return false;
  return a.every((v, i) => v === b[i]);
}

function mapWorld(world: string): string {
  switch (world) {
    case "A":
      return "Spirit";
    case "B":
      return "Mind";
    case "C":
      return "Heart";
    case "D":
      return "Body/World";
    default:
      return "Domain";
  }
}

function mapAxis(axis: string): string {
  switch (axis) {
    case "a":
      return "Desire";
    case "b":
      return "Wisdom";
    case "c":
      return "Power";
    case "d":
      return "Manifestation";
    default:
      return "Flow";
  }
}

function formatList(values: (string | number)[]): string {
  return values.length === 0 ? "(none)" : values.join(", ");
}

function pickPatternName(patterns: NoeticPattern[]): string {
  if (patterns.length === 0) return "";
  const first = patterns[0];
  if (first.label.toLowerCase().includes("aware")) {
    return "Aware-Vibrate-Rhythm";
  }
  return first.label;
}

function buildNarrative(
  goal: FoundationId,
  meta: FoundationMeta,
  rpm: RpmCheckResult,
  activation: number,
  patterns: NoeticPattern[],
  exampleExpressions: string[]
): string {
  const worldLabel = mapWorld(meta.world);
  const axisLabel = mapAxis(meta.axis);
  const activationStr = activation.toFixed(2);
  const patternName = pickPatternName(patterns);
  const example = exampleExpressions[0] ?? "";

  const baseIntro = `Goal ${goal} - ${worldLabel} / ${axisLabel}: ${meta.label}, activation ${activationStr}.`;

  if (rpm.missing.length > 0) {
    const missingList = formatList(rpm.missing);
    const rpmLine = `Missing acquisitions: ${missingList}.`;
    const gateLine = `Seal Desire, Wisdom, and Power for this domain before pushing further Noetics; then run the ${patternName || "Aware-Vibrate-Rhythm"} stream through this gate via ${example || goal}.`;
    return `${baseIntro} ${rpmLine} ${gateLine}`;
  }

  if (rpm.required.length > 0) {
    const rpmLine = `All required acquisitions present for this goal.`;
    const flowLine = `Turn awareness, vibration, and rhythm toward ${goal} using ${example || goal} until this foundation stabilizes.`;
    return `${baseIntro} ${rpmLine} ${flowLine}`;
  }

  const noRpmLine = `This foundation opens without RPM gates; ride the ${patternName || "Aware-Vibrate-Rhythm"} stream through ${example || goal} to deepen the flow.`;
  return `${baseIntro} ${noRpmLine}`;
}

export function coachForFoundation(goal: FoundationId, state: TksState): CoachSuggestion {
  const foundationMeta = FOUNDATIONS[goal];
  const rpm = checkPrereqs(goal, state);
  const currentActivation = state.foundations[goal] ?? 0;
  const missingAcquisitions = rpm.missing;

  let suggestedPatterns = getDefaultPatternsForFoundation(goal);
  if (suggestedPatterns.length === 0) {
    const fallback = NOETIC_PATTERNS.find((p) => sequencesEqual(p.sequence, [1, 4, 7]));
    if (fallback) {
      suggestedPatterns = [fallback];
    }
  }

  const exampleExpressions = suggestedPatterns.slice(0, 3).map((pattern) => {
    const seq = pattern.sequence.join(":");
    return `<<${seq}>>(${goal})`;
  });

  const narrative = buildNarrative(
    goal,
    foundationMeta,
    rpm,
    currentActivation,
    suggestedPatterns,
    exampleExpressions
  );

  return {
    goalFoundation: goal,
    foundationMeta,
    rpm,
    currentActivation,
    missingAcquisitions,
    suggestedPatterns,
    exampleExpressions,
    narrative,
  };
}

export function coachAiVitals(
  vitals: AiVitals,
  options?: AiToTksMappingOptions
): AiCoachReport {
  const state = createDefaultState();
  applyAiVitalsToFoundations(vitals, state, options);
  const summary = summarizeAiVitals(vitals);

  const foundationCoaches: Partial<Record<FoundationId, CoachSuggestion>> = {};
  const targetIds: FoundationId[] = ["3d", "4d", "5d", "6d"];
  for (const id of targetIds) {
    foundationCoaches[id] = coachForFoundation(id, state);
  }

  const f3 = summary.foundations["3d"] ?? 0;
  const f4 = summary.foundations["4d"] ?? 0;
  const f5 = summary.foundations["5d"] ?? 0;
  const f6 = summary.foundations["6d"] ?? 0;

  const missingLines: string[] = [];
  for (const id of targetIds) {
    const coach = foundationCoaches[id];
    if (coach && coach.missingAcquisitions.length > 0) {
      missingLines.push(`Missing acquisitions for ${id}: ${coach.missingAcquisitions.join(", ")}`);
    }
  }

  const baseLine = `AI run ${vitals.id}: health(3d)=${f3.toFixed(
    2
  )}, relations(4d)=${f4.toFixed(2)}, power(5d)=${f5.toFixed(2)}, material(6d)=${f6.toFixed(
    2
  )}.`;
  const rpmLine =
    missingLines.length > 0
      ? missingLines.join(" ")
      : "No RPM gaps detected for 3d/4d/5d/6d.";
  const recLine =
    "Recommend stabilizing power and material foundations with standard Aware–Vibrate–Rhythm patterns.";

  const narrative = `${baseLine} ${rpmLine} ${recLine}`;

  return {
    vitals,
    summary,
    foundationCoaches,
    narrative,
  };
}
