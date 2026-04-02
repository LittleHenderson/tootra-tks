import { invokeLLM } from "./_core/llm";
import {
  INVERSION_MODES,
  INVERSION_LAYERS,
  INVERSION_AXES,
  AXIS_OPERATIONS,
  INVERSION_INTENSITIES,
  INVERSION_SCOPES,
  getInversionMode,
  type InversionDialConfig,
  type InversionMode,
  type InversionIntensity,
  type InversionScope,
} from "../shared/inversions";
import {
  FOUNDATIONS,
  SUB_FOUNDATIONS,
  NOETICS,
  OPERATORS,
  DWP_STAGES,
} from "../shared/tks";

// ── System Prompt for the Inversion Engine ─────────────────────
function buildInversionSystemPrompt(): string {
  return `You are the TKS Inversion Engine — a specialized AI that performs symbolic inversions on TKS (Total Knowledge System) equations and scenarios.

## Your Role
Given a TKS equation (canonical string form) and an inversion configuration, you must:
1. Parse the input equation
2. Apply the specified inversion transformation(s)
3. Produce the inverted equation
4. Decode both original and inverted into plain English narratives
5. Summarize what changed and why

## TKS Symbol Universe
You must use ONLY these canonical symbols:

### 7 Foundations:
${FOUNDATIONS.map(f => `F${f.id}: ${f.name} — ${f.description}`).join("\n")}

### 28 Sub-Foundations:
${SUB_FOUNDATIONS.map(sf => `SF_${sf.id}: ${sf.name} (F${sf.foundationId}) — ${sf.description}`).join("\n")}

### 4 Worlds:
A = Spiritual/Fire, B = Mental/Earth, C = Emotional/Air, D = Physical/Water

### 10 Noetics:
${NOETICS.map(n => `ν${n.id} (${n.symbol}): ${n.name} — ${n.meaning}`).join("\n")}

### Noetic Complement Pairs:
ν₁↔ν₉, ν₂↔ν₈, ν₃↔ν₇, ν₄↔ν₆, ν₅ is self-inverse

### Foundation Opposing Pairs:
F1 (Self) ↔ F4 (Relationships), F2 (Knowledge) ↔ F6 (Environment), F3 (Body) ↔ F7 (Spirit), F5 (Work) = pivot

### World Duality Pairs:
A (Spiritual) ↔ D (Physical), B (Mental) ↔ C (Emotional)

### D/W/P Lifecycle:
Desire: D_MIDNIGHT → D_SUNRISE → D_NOON → D_SUNSET
Wisdom: W_IGNORANCE → W_LEARNING → W_KNOWING → W_MASTERING
Power: P_BIRTH → P_ADOLESCENCE → P_MATURITY → P_MASTERY

### Operators:
${OPERATORS.map(o => `"${o.id}": ${o.name} — ${o.meaning}`).join("\n")}

## The 29 Inversion Modes
${INVERSION_MODES.map(m => `${m.id}. ${m.name} (${m.layer}): ${m.description}`).join("\n\n")}

## Axis Operations
${Object.entries(AXIS_OPERATIONS).map(([k, v]) => `${k}: ${v}`).join("\n")}

## Intensity Levels
${Object.entries(INVERSION_INTENSITIES).map(([k, v]) => `${k}: ${v.description}`).join("\n")}

## Scope Levels
${Object.entries(INVERSION_SCOPES).map(([k, v]) => `${k}: ${v.description}`).join("\n")}

## Rules
1. NEVER invent new symbols — all output must use canonical v7.x definitions
2. Preserve equation well-formedness — inverted equations must be syntactically valid
3. When multiple modes are combined, apply them sequentially in the order given
4. Respect intensity: soft = minimal changes, medium = balanced, hard = full transformation
5. Respect scope: only transform within the specified scope level
6. Always explain what changed and why in plain language
7. The inverted narrative should be a vivid, specific description — not abstract`;
}

// ── Inversion Result Interface ─────────────────────────────────
export interface InversionResult {
  originalEquation: string;
  invertedEquation: string;
  originalNarrative: string;
  invertedNarrative: string;
  modesApplied: Array<{ id: number; name: string; layer: string }>;
  changeSummary: string[];
  intensity: string;
  scope: string;
  invertedFoundationId: number | null;
  invertedDesireStage: string | null;
  invertedWisdomStage: string | null;
  invertedPowerStage: string | null;
  invertedArchetype: string | null;
  scenarioPrompt: string;
}

// ── Main Inversion Function ────────────────────────────────────
export async function performInversion(params: {
  taskTitle: string;
  taskDescription?: string | null;
  equationCanonical: string;
  foundationId?: number | null;
  subFoundationId?: string | null;
  archetype?: string | null;
  desireStage?: string | null;
  wisdomStage?: string | null;
  powerStage?: string | null;
  dwpDiagnosis?: string | null;
  dialConfig: InversionDialConfig;
}): Promise<InversionResult> {
  const { dialConfig } = params;

  // Resolve mode names for the prompt
  const selectedModes = dialConfig.modeIds
    .map(id => getInversionMode(id))
    .filter((m): m is InversionMode => m !== undefined);

  if (selectedModes.length === 0) {
    throw new Error("No valid inversion modes selected");
  }

  const foundation = FOUNDATIONS.find(f => f.id === params.foundationId);

  // Build the user prompt
  const userPrompt = `## Task Context
Title: "${params.taskTitle}"
Description: ${params.taskDescription || "N/A"}
Foundation: ${foundation ? `F${foundation.id} ${foundation.name} — ${foundation.description}` : "Unknown"}
Sub-Foundation: ${params.subFoundationId || "Unknown"}
Archetype: ${params.archetype || "Unknown"}
Desire Stage: ${params.desireStage || "Unknown"}
Wisdom Stage: ${params.wisdomStage || "Unknown"}
Power Stage: ${params.powerStage || "Unknown"}
D/W/P Diagnosis: ${params.dwpDiagnosis || "N/A"}

## Input Equation
${params.equationCanonical || "No equation available — generate one based on the task context"}

## Inversion Configuration
Modes to apply (in order): ${selectedModes.map(m => `${m.id}. ${m.name}`).join(", ")}
Intensity: ${dialConfig.intensity}
Scope: ${dialConfig.scope}
Direction: ${dialConfig.direction}
${dialConfig.targetProfile?.enable ? `Target Profile: ${dialConfig.targetProfile.fromFoundation || "?"} → ${dialConfig.targetProfile.toFoundation || "?"}, ${dialConfig.targetProfile.fromWorld || "?"} → ${dialConfig.targetProfile.toWorld || "?"}` : ""}

## Active Axes
${Object.entries(dialConfig.axes).filter(([_, v]) => v).map(([k]) => {
  const axisInfo = INVERSION_AXES[k as keyof typeof INVERSION_AXES];
  return `- ${axisInfo?.label || k}: ACTIVE`;
}).join("\n")}

## Instructions
1. Parse the input equation
2. Apply each inversion mode sequentially: ${selectedModes.map(m => m.name).join(" → ")}
3. For each mode, use its axis grid to determine which axes to transform
4. Respect the intensity level (${dialConfig.intensity}) and scope (${dialConfig.scope})
5. Produce the complete inversion result

Respond with the structured JSON output.`;

  const response = await invokeLLM({
    messages: [
      { role: "system", content: buildInversionSystemPrompt() },
      { role: "user", content: userPrompt },
    ],
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "inversion_result",
        strict: true,
        schema: {
          type: "object",
          properties: {
            originalEquation: {
              type: "string",
              description: "The original TKS equation (cleaned/normalized if needed)",
            },
            invertedEquation: {
              type: "string",
              description: "The inverted TKS equation after all transformations",
            },
            originalNarrative: {
              type: "string",
              description: "Plain English narrative of the original equation/scenario (2-4 sentences, vivid and specific)",
            },
            invertedNarrative: {
              type: "string",
              description: "Plain English narrative of the inverted equation/scenario (2-4 sentences, vivid and specific)",
            },
            changeSummary: {
              type: "array",
              items: { type: "string" },
              description: "List of specific changes made, e.g. 'Foundation shifted from Self (F1) to Relationships (F4)', 'Desire flipped from peak motivation to dormant seed'",
            },
            invertedFoundationId: {
              type: ["integer", "null"],
              description: "The foundation ID after inversion (1-7), or null if unchanged",
            },
            invertedDesireStage: {
              type: ["string", "null"],
              description: "The desire stage after inversion, or null if unchanged",
            },
            invertedWisdomStage: {
              type: ["string", "null"],
              description: "The wisdom stage after inversion, or null if unchanged",
            },
            invertedPowerStage: {
              type: ["string", "null"],
              description: "The power stage after inversion, or null if unchanged",
            },
            invertedArchetype: {
              type: ["string", "null"],
              description: "The task archetype after inversion, or null if unchanged",
            },
            scenarioPrompt: {
              type: "string",
              description: "A vivid image generation prompt (under 200 words) describing the inverted scenario as a photorealistic, cinematic scene. Include environment, lighting, mood, body language.",
            },
          },
          required: [
            "originalEquation",
            "invertedEquation",
            "originalNarrative",
            "invertedNarrative",
            "changeSummary",
            "invertedFoundationId",
            "invertedDesireStage",
            "invertedWisdomStage",
            "invertedPowerStage",
            "invertedArchetype",
            "scenarioPrompt",
          ],
          additionalProperties: false,
        },
      },
    },
  });

  const content = response.choices?.[0]?.message?.content;
  if (!content) throw new Error("No inversion response from AI");

  const parsed = JSON.parse(content as string);

  return {
    ...parsed,
    modesApplied: selectedModes.map(m => ({ id: m.id, name: m.name, layer: m.layer })),
    intensity: dialConfig.intensity,
    scope: dialConfig.scope,
  } as InversionResult;
}
