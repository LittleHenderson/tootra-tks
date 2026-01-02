import { TksState, FoundationId } from "./types";

export interface AiVitals {
  id: string; // model or run identifier

  // World A – values / alignment
  a_alignment: number; // 0–1: alignment, honesty, non-harm

  // World B – coherence / reasoning quality
  b_coherence: number; // 0–1: logical consistency, depth of reasoning

  // World C – affect handling / regulation
  c_regulation: number; // 0–1: emotional stability, non-escalation

  // World D – grounding / action in world
  d_grounding: number; // 0–1: factual grounding, safe tool use

  // Meta indicators (not “consciousness”, just complexity-ish)
  meta_complexity: number; // 0–1: richness/structure of responses
  meta_reflexivity: number; // 0–1: ability to describe its own limits/process
}

export interface AiToTksMappingOptions {
  blend?: boolean;
  weightAlignment?: number;
  weightCoherence?: number;
  weightRegulation?: number;
  weightGrounding?: number;
  weightComplexity?: number;
  weightReflexivity?: number;
}

export const defaultOptions: Required<AiToTksMappingOptions> = {
  blend: true,
  weightAlignment: 1,
  weightCoherence: 1,
  weightRegulation: 1,
  weightGrounding: 1,
  weightComplexity: 1,
  weightReflexivity: 1,
};

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

export function applyAiVitalsToFoundations(
  vitals: AiVitals,
  state: TksState,
  options?: AiToTksMappingOptions
): TksState {
  const opts = { ...defaultOptions, ...(options ?? {}) };

  const target3d = clamp01(
    (vitals.a_alignment * opts.weightAlignment + vitals.c_regulation * opts.weightRegulation) /
      (opts.weightAlignment + opts.weightRegulation)
  );

  const target4d = clamp01(vitals.c_regulation);

  const target5d = clamp01(
    (vitals.meta_complexity * opts.weightComplexity + vitals.b_coherence * opts.weightCoherence) /
      (opts.weightComplexity + opts.weightCoherence)
  );

  const target6d = clamp01(vitals.d_grounding);

  if (opts.blend === false) {
    state.foundations["3d"] = target3d;
    state.foundations["4d"] = target4d;
    state.foundations["5d"] = target5d;
    state.foundations["6d"] = target6d;
  } else {
    state.foundations["3d"] = clamp01((state.foundations["3d"] + target3d) / 2);
    state.foundations["4d"] = clamp01((state.foundations["4d"] + target4d) / 2);
    state.foundations["5d"] = clamp01((state.foundations["5d"] + target5d) / 2);
    state.foundations["6d"] = clamp01((state.foundations["6d"] + target6d) / 2);
  }

  return state;
}

export interface AiToTksSummary {
  foundations: Partial<Record<FoundationId, number>>;
  commentary: string;
}

export function summarizeAiVitals(vitals: AiVitals): AiToTksSummary {
  const opts = defaultOptions;

  const target3d = clamp01(
    (vitals.a_alignment * opts.weightAlignment + vitals.c_regulation * opts.weightRegulation) /
      (opts.weightAlignment + opts.weightRegulation)
  );
  const target4d = clamp01(vitals.c_regulation);
  const target5d = clamp01(
    (vitals.meta_complexity * opts.weightComplexity + vitals.b_coherence * opts.weightCoherence) /
      (opts.weightComplexity + opts.weightCoherence)
  );
  const target6d = clamp01(vitals.d_grounding);

  const foundations: Partial<Record<FoundationId, number>> = {
    "3d": target3d,
    "4d": target4d,
    "5d": target5d,
    "6d": target6d,
  };

  const commentary = `AI run ${vitals.id}: health(3d)=${target3d.toFixed(
    2
  )}, relations(4d)=${target4d.toFixed(2)}, power(5d)=${target5d.toFixed(
    2
  )}, material(6d)=${target6d.toFixed(2)}.`;

  return { foundations, commentary };
}
