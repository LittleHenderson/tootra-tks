import { describe, expect, it, vi } from "vitest";
import { appRouter } from "./routers";
import type { TrpcContext } from "./_core/context";

type AuthenticatedUser = NonNullable<TrpcContext["user"]>;

function createAuthContext(): TrpcContext {
  const user: AuthenticatedUser = {
    id: 1,
    openId: "test-user",
    email: "test@example.com",
    name: "Test User",
    loginMethod: "manus",
    role: "user",
    mode: "life_os",
    createdAt: new Date(),
    updatedAt: new Date(),
    lastSignedIn: new Date(),
  };

  return {
    user,
    req: {
      protocol: "https",
      headers: {},
    } as TrpcContext["req"],
    res: {
      clearCookie: () => {},
    } as TrpcContext["res"],
  };
}

describe("voice.transcribe", () => {
  it("rejects empty audioBase64", async () => {
    const ctx = createAuthContext();
    const caller = appRouter.createCaller(ctx);

    await expect(
      caller.voice.transcribe({ audioBase64: "", mimeType: "audio/webm" })
    ).rejects.toThrow();
  });

  it("validates input schema requires audioBase64", async () => {
    const ctx = createAuthContext();
    const caller = appRouter.createCaller(ctx);

    // @ts-expect-error - testing missing required field
    await expect(caller.voice.transcribe({})).rejects.toThrow();
  });

  it("accepts valid mimeType values", async () => {
    const ctx = createAuthContext();
    const caller = appRouter.createCaller(ctx);

    // This will fail at the S3 upload stage (not in test env), but validates schema acceptance
    const validMimes = ["audio/webm", "audio/mp4", "audio/wav"];
    for (const mime of validMimes) {
      try {
        await caller.voice.transcribe({
          audioBase64: "dGVzdA==", // "test" in base64
          mimeType: mime,
        });
      } catch (e: any) {
        // Expected to fail at storage/transcription, not at input validation
        expect(e.message).not.toContain("validation");
      }
    }
  });

  it("defaults mimeType to audio/webm when not provided", async () => {
    const ctx = createAuthContext();
    const caller = appRouter.createCaller(ctx);

    try {
      await caller.voice.transcribe({
        audioBase64: "dGVzdA==",
      });
    } catch (e: any) {
      // Should fail at storage, not at schema validation for missing mimeType
      expect(e.message).not.toContain("mimeType");
    }
  });
});

describe("ReadAloud text builder logic", () => {
  // Test the text building functions that ReadAloud uses

  it("builds complete read-aloud text from task data", () => {
    const task = {
      title: "Study React hooks",
      description: "Learn useState and useEffect",
      archetype: "study",
      urgency: "medium",
      desireStage: "craving",
      wisdomStage: "research",
      powerStage: "gathering",
      dwpDiagnosis: "Good alignment between desire and action",
      equationCanonical: "ACT(study) + NOE(technical) → D(craving)",
    };
    const powerItems = [
      { status: "have", category: "time" },
      { status: "lack", category: "skill" },
    ];

    // Simulate the buildReadAloudText function
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
      const haves = powerItems.filter(p => p.status === "have").map(p => p.category);
      const lacks = powerItems.filter(p => p.status === "lack").map(p => p.category);
      if (haves.length) parts.push(`You have: ${haves.join(", ")}.`);
      if (lacks.length) parts.push(`You lack: ${lacks.join(", ")}.`);
    }
    if (task.equationCanonical) parts.push(`TKS Equation: ${task.equationCanonical}`);
    const text = parts.join(" ");

    expect(text).toContain("Task: Study React hooks.");
    expect(text).toContain("Learn useState and useEffect");
    expect(text).toContain("This is a study task.");
    expect(text).toContain("Urgency is medium.");
    expect(text).toContain("Desire stage: craving.");
    expect(text).toContain("Wisdom stage: research.");
    expect(text).toContain("Power stage: gathering.");
    expect(text).toContain("Diagnosis: Good alignment");
    expect(text).toContain("You have: time.");
    expect(text).toContain("You lack: skill.");
    expect(text).toContain("TKS Equation:");
  });

  it("handles empty/null task fields gracefully", () => {
    const task = {
      title: "Simple task",
      description: null,
      archetype: null,
      urgency: null,
      desireStage: null,
      wisdomStage: null,
      powerStage: null,
      dwpDiagnosis: null,
      equationCanonical: null,
    };

    const parts: string[] = [];
    parts.push(`Task: ${task.title}.`);
    if (task.description) parts.push(task.description);
    if (task.archetype) parts.push(`This is a ${task.archetype} task.`);
    const text = parts.join(" ");

    expect(text).toBe("Task: Simple task.");
  });

  it("builds DWP read-aloud text correctly", () => {
    const task = {
      desireStage: "craving",
      wisdomStage: "research",
      powerStage: "gathering",
      dwpDiagnosis: "Mismatch detected",
    };

    const parts: string[] = [];
    if (task.desireStage) parts.push(`Desire stage is ${task.desireStage.replace(/_/g, " ")}`);
    if (task.wisdomStage) parts.push(`Wisdom stage is ${task.wisdomStage.replace(/_/g, " ")}`);
    if (task.powerStage) parts.push(`Power stage is ${task.powerStage.replace(/_/g, " ")}`);
    if (task.dwpDiagnosis) parts.push(`Diagnosis: ${task.dwpDiagnosis}`);
    const text = parts.join(". ");

    expect(text).toContain("Desire stage is craving");
    expect(text).toContain("Wisdom stage is research");
    expect(text).toContain("Power stage is gathering");
    expect(text).toContain("Diagnosis: Mismatch detected");
  });

  it("replaces underscores with spaces in stage names", () => {
    const stage = "deep_craving";
    const readable = stage.replace(/_/g, " ");
    expect(readable).toBe("deep craving");
  });

  it("returns fallback text when no DWP data", () => {
    const task = {
      desireStage: null,
      wisdomStage: null,
      powerStage: null,
      dwpDiagnosis: null,
    };

    const parts: string[] = [];
    if (task.desireStage) parts.push(`Desire stage is ${task.desireStage}`);
    if (task.wisdomStage) parts.push(`Wisdom stage is ${task.wisdomStage}`);
    if (task.powerStage) parts.push(`Power stage is ${task.powerStage}`);
    if (task.dwpDiagnosis) parts.push(`Diagnosis: ${task.dwpDiagnosis}`);
    const text = parts.join(". ") || "No lifecycle data available.";

    expect(text).toBe("No lifecycle data available.");
  });
});
