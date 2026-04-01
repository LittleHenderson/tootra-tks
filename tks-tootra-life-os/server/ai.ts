import { invokeLLM } from "./_core/llm";
import { FOUNDATIONS, SUB_FOUNDATIONS, NOETICS, OPERATORS, DWP_STAGES, TASK_ARCHETYPES, POWER_CATEGORIES } from "../shared/tks";

const TKS_SYSTEM_PROMPT = `You are the TKS (Total Knowledge System) Interpreter. Your role is to analyze a user's natural language task and produce a structured TKS interpretation.

## TKS Foundations (7):
${FOUNDATIONS.map(f => `${f.id}. ${f.name} — ${f.description}`).join("\n")}

## Sub-Foundations (28):
${SUB_FOUNDATIONS.map(sf => `${sf.id}: ${sf.name} (Foundation ${sf.foundationId}) — ${sf.description}`).join("\n")}

## Task Archetypes (6):
- execution: Do the thing. Concrete action producing a result. Signal words: do, finish, complete, submit, send, ship.
- prep: Get ready. Gathering resources/prerequisites. Signal: prepare, set up, gather, before I can, first I need.
- maintenance: Keep doing. Recurring rhythm-driven tasks. Signal: every day, weekly, routine, habit, keep up.
- repair: Fix what broke. Reactive to failure/disruption. Signal: fix, repair, resolve, broke, went wrong.
- study: Learn before doing. Knowledge acquisition. Signal: study, learn, research, how do I, figure out.
- audit: Check how it went. Review and retrospective. Signal: review, evaluate, reflect, how did it go.

## D/W/P Lifecycle:
Desire stages: D_MIDNIGHT (dormant), D_SUNRISE (awakening), D_NOON (peak), D_SUNSET (fading)
Wisdom stages: W_IGNORANCE (unknown), W_LEARNING (acquiring), W_KNOWING (sufficient), W_MASTERING (expert)
Power stages: P_BIRTH (no resources), P_ADOLESCENCE (partial), P_MATURITY (all available), P_MASTERY (surplus)

## 10 Noetics:
${NOETICS.map(n => `${n.symbol} ${n.name}: ${n.meaning}`).join("\n")}

## 11 Operators:
${OPERATORS.map(o => `"${o.id}" ${o.name}: ${o.meaning}`).join("\n")}

## Power Categories (8):
${POWER_CATEGORIES.map(p => `${p.id}: ${p.name} — ${p.description}`).join("\n")}

## TKS Equation Token Types (13):
- ACT(action_name): The action to perform
- TIME(iso_datetime): When to do it
- DUR(minutes): How long it takes
- RHY(cadence): Recurrence pattern (daily, weekly, monthly, etc.)
- MAP(foundation_id, sub_foundation_id): Foundation mapping
- NOE(^n): Noetic tag
- LINK(type, target): Dependency link
- D(stage): Desire stage
- W(stage): Wisdom stage
- P(stage): Power stage
- REQ(resource): Required resource
- HAVE(resource): Available resource
- LACK(resource): Missing resource

## Your Task:
Given a natural language task description, produce a JSON response with:
1. A clean task title
2. Foundation and sub-foundation mapping with confidence (0-1)
3. Task archetype
4. D/W/P stage assessment
5. A TKS equation in canonical string form
6. The equation as a structured AST (array of tokens)
7. Noetic tags
8. Power inventory assessment (which of the 8 categories are HAVE/LACK/REQ)
9. Priority urgency assessment
10. Duration estimate in minutes
11. D/W/P diagnosis (any mismatches or blocks detected)`;

export interface TKSInterpretation {
  title: string;
  description: string;
  foundationId: number;
  subFoundationId: string;
  mappingConfidence: number;
  archetype: string;
  desireStage: string;
  wisdomStage: string;
  powerStage: string;
  equationCanonical: string;
  equationAst: any[];
  noetics: string[];
  powerInventory: Array<{ category: string; status: "have" | "lack" | "partial"; note: string; suggestedSubtask?: string }>;
  urgency: string;
  durationMinutes: number;
  dwpDiagnosis: string;
  priorityScore: number;
}

export async function interpretTask(rawInput: string): Promise<TKSInterpretation> {
  const response = await invokeLLM({
    messages: [
      { role: "system", content: TKS_SYSTEM_PROMPT },
      { role: "user", content: `Interpret this task: "${rawInput}"` },
    ],
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "tks_interpretation",
        strict: true,
        schema: {
          type: "object",
          properties: {
            title: { type: "string", description: "Clean, concise task title" },
            description: { type: "string", description: "Brief description of what needs to be done" },
            foundationId: { type: "integer", description: "Primary foundation ID (1-7)" },
            subFoundationId: { type: "string", description: "Sub-foundation ID like '1a', '3b', etc." },
            mappingConfidence: { type: "number", description: "Confidence in the mapping (0.0 to 1.0)" },
            archetype: { type: "string", description: "Task archetype: execution, prep, maintenance, repair, study, or audit" },
            desireStage: { type: "string", description: "D_MIDNIGHT, D_SUNRISE, D_NOON, or D_SUNSET" },
            wisdomStage: { type: "string", description: "W_IGNORANCE, W_LEARNING, W_KNOWING, or W_MASTERING" },
            powerStage: { type: "string", description: "P_BIRTH, P_ADOLESCENCE, P_MATURITY, or P_MASTERY" },
            equationCanonical: { type: "string", description: "TKS equation in canonical string form" },
            equationAst: {
              type: "array",
              description: "Equation as array of token objects",
              items: {
                type: "object",
                properties: {
                  type: { type: "string", description: "Token type: ACT, TIME, DUR, RHY, MAP, NOE, LINK, D, W, P, REQ, HAVE, LACK" },
                  value: { type: "string", description: "Token value" },
                  operator: { type: "string", description: "Operator before this token (o, ->, <-, +, etc.) or empty for first token" },
                },
                required: ["type", "value", "operator"],
                additionalProperties: false,
              },
            },
            noetics: { type: "array", items: { type: "string" }, description: "Noetic symbols like ^0, ^2, etc." },
            powerInventory: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  category: { type: "string" },
                  status: { type: "string", description: "have, lack, or partial" },
                  note: { type: "string" },
                  suggestedSubtask: { type: "string" },
                },
                required: ["category", "status", "note", "suggestedSubtask"],
                additionalProperties: false,
              },
            },
            urgency: { type: "string", description: "low, medium, high, or critical" },
            durationMinutes: { type: "integer", description: "Estimated duration in minutes" },
            dwpDiagnosis: { type: "string", description: "Analysis of D/W/P alignment and any mismatches" },
            priorityScore: { type: "number", description: "Calculated priority score 0-100" },
          },
          required: ["title", "description", "foundationId", "subFoundationId", "mappingConfidence", "archetype", "desireStage", "wisdomStage", "powerStage", "equationCanonical", "equationAst", "noetics", "powerInventory", "urgency", "durationMinutes", "dwpDiagnosis", "priorityScore"],
          additionalProperties: false,
        },
      },
    },
  });

  const content = response.choices?.[0]?.message?.content;
  if (!content) throw new Error("No response from AI");
  return JSON.parse(content as string) as TKSInterpretation;
}

export async function generateSubtasksForGap(taskTitle: string, category: string): Promise<string[]> {
  const response = await invokeLLM({
    messages: [
      { role: "system", content: "You generate practical subtasks to close power gaps. Return a JSON array of 2-4 actionable subtask strings." },
      { role: "user", content: `Task: "${taskTitle}"\nPower gap category: ${category}\nGenerate subtasks to acquire this missing resource.` },
    ],
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "subtasks",
        strict: true,
        schema: {
          type: "object",
          properties: {
            subtasks: { type: "array", items: { type: "string" } },
          },
          required: ["subtasks"],
          additionalProperties: false,
        },
      },
    },
  });

  const content = response.choices?.[0]?.message?.content;
  if (!content) return [];
  const parsed = JSON.parse(content as string);
  return parsed.subtasks || [];
}

export interface ScenarioPromptResult {
  positivePrompt: string;
  negativePrompt: string;
  positiveDescription: string;
  negativeDescription: string;
}

export async function buildScenarioPrompts(task: {
  title: string;
  description?: string | null;
  foundationId?: number | null;
  archetype?: string | null;
  desireStage?: string | null;
  wisdomStage?: string | null;
  powerStage?: string | null;
  dwpDiagnosis?: string | null;
}): Promise<ScenarioPromptResult> {
  const foundation = FOUNDATIONS.find(f => f.id === task.foundationId);

  const response = await invokeLLM({
    messages: [
      {
        role: "system",
        content: `You are a cinematic scenario writer for the TKS Life Execution System. Given a task with its TKS context, you create two vivid, realistic scene descriptions — one showing the POSITIVE outcome if the person follows through, and one showing the NEGATIVE consequence if they don't.

Rules:
- Write image generation prompts that create photorealistic, cinematic scenes
- The scenes should feel personal and emotionally resonant
- Include specific environmental details, lighting, mood, and body language
- Reference the TKS Foundation context to ground the scenario in the right life domain
- Keep prompts under 200 words each
- Also write a short 1-2 sentence human-readable description for each scenario`
      },
      {
        role: "user",
        content: `Task: "${task.title}"
Description: ${task.description || "N/A"}
Foundation: ${foundation?.name || "Unknown"} — ${foundation?.description || ""}
Archetype: ${task.archetype || "Unknown"}
Desire Stage: ${task.desireStage || "Unknown"}
Wisdom Stage: ${task.wisdomStage || "Unknown"}
Power Stage: ${task.powerStage || "Unknown"}
D/W/P Diagnosis: ${task.dwpDiagnosis || "N/A"}

Generate the two scenario prompts.`
      },
    ],
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "scenario_prompts",
        strict: true,
        schema: {
          type: "object",
          properties: {
            positivePrompt: { type: "string", description: "Image generation prompt for the positive outcome scenario — photorealistic, cinematic" },
            negativePrompt: { type: "string", description: "Image generation prompt for the negative consequence scenario — photorealistic, cinematic" },
            positiveDescription: { type: "string", description: "1-2 sentence human-readable description of the positive outcome" },
            negativeDescription: { type: "string", description: "1-2 sentence human-readable description of the negative consequence" },
          },
          required: ["positivePrompt", "negativePrompt", "positiveDescription", "negativeDescription"],
          additionalProperties: false,
        },
      },
    },
  });

  const content = response.choices?.[0]?.message?.content;
  if (!content) throw new Error("No scenario response from AI");
  return JSON.parse(content as string) as ScenarioPromptResult;
}
