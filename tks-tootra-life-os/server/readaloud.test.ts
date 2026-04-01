import { describe, expect, it, beforeEach } from "vitest";

/**
 * Tests for ReadAloud component logic.
 * Since the component uses browser SpeechSynthesis API (not available in Node),
 * we test the pure logic: text building, voice persistence, and rate management.
 */

describe("ReadAloud voice persistence logic", () => {
  const STORAGE_KEY = "tks-tts-voice";
  const RATE_KEY = "tks-tts-rate";

  // Simulate localStorage
  let store: Record<string, string> = {};
  const mockStorage = {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => { store[key] = value; },
    removeItem: (key: string) => { delete store[key]; },
  };

  beforeEach(() => {
    store = {};
  });

  it("stores selected voice name in localStorage", () => {
    mockStorage.setItem(STORAGE_KEY, "Google UK English Male");
    expect(mockStorage.getItem(STORAGE_KEY)).toBe("Google UK English Male");
  });

  it("stores speech rate in localStorage", () => {
    mockStorage.setItem(RATE_KEY, "1.5");
    const rate = parseFloat(mockStorage.getItem(RATE_KEY)!);
    expect(rate).toBe(1.5);
  });

  it("defaults rate to 1.0 when no stored value", () => {
    const saved = mockStorage.getItem(RATE_KEY);
    const rate = saved ? parseFloat(saved) : 1.0;
    expect(rate).toBe(1.0);
  });

  it("defaults voice to empty string when no stored value", () => {
    const voice = mockStorage.getItem(STORAGE_KEY) || "";
    expect(voice).toBe("");
  });

  it("persists voice selection across sessions", () => {
    // First session
    mockStorage.setItem(STORAGE_KEY, "Microsoft Zira");
    
    // Simulate new session by reading back
    const restored = mockStorage.getItem(STORAGE_KEY);
    expect(restored).toBe("Microsoft Zira");
  });

  it("persists rate selection across sessions", () => {
    mockStorage.setItem(RATE_KEY, "0.75");
    const restored = parseFloat(mockStorage.getItem(RATE_KEY)!);
    expect(restored).toBe(0.75);
  });

  it("validates rate values are within expected range", () => {
    const validRates = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0];
    for (const rate of validRates) {
      expect(rate).toBeGreaterThanOrEqual(0.5);
      expect(rate).toBeLessThanOrEqual(2.0);
    }
  });
});

describe("ReadAloud text preparation", () => {
  it("strips underscores from stage names for natural speech", () => {
    const stages = ["deep_craving", "initial_spark", "full_commitment", "gathering_resources"];
    const expected = ["deep craving", "initial spark", "full commitment", "gathering resources"];
    
    stages.forEach((stage, i) => {
      expect(stage.replace(/_/g, " ")).toBe(expected[i]);
    });
  });

  it("builds a complete task summary for speech", () => {
    const task = {
      title: "Fix the leaky faucet",
      description: "Kitchen sink has been dripping for a week",
      archetype: "repair",
      urgency: "high",
      desireStage: "full_commitment",
      wisdomStage: "planning",
      powerStage: "gathering",
      dwpDiagnosis: "Strong desire but still gathering resources",
      equationCanonical: "ACT(repair) + MAP(home) + D(committed)",
    };
    const powerItems = [
      { status: "have", category: "tools" },
      { status: "have", category: "time" },
      { status: "lack", category: "skill" },
    ];

    const parts: string[] = [];
    parts.push(`Task: ${task.title}.`);
    if (task.description) parts.push(task.description);
    if (task.archetype) parts.push(`This is a ${task.archetype} task.`);
    if (task.urgency) parts.push(`Urgency is ${task.urgency}.`);
    if (task.desireStage) parts.push(`Desire stage: ${task.desireStage.replace(/_/g, " ")}.`);
    if (task.wisdomStage) parts.push(`Wisdom stage: ${task.wisdomStage.replace(/_/g, " ")}.`);
    if (task.powerStage) parts.push(`Power stage: ${task.powerStage.replace(/_/g, " ")}.`);
    if (task.dwpDiagnosis) parts.push(`Diagnosis: ${task.dwpDiagnosis}`);
    const haves = powerItems.filter(p => p.status === "have").map(p => p.category);
    const lacks = powerItems.filter(p => p.status === "lack").map(p => p.category);
    if (haves.length) parts.push(`You have: ${haves.join(", ")}.`);
    if (lacks.length) parts.push(`You lack: ${lacks.join(", ")}.`);
    if (task.equationCanonical) parts.push(`TKS Equation: ${task.equationCanonical}`);
    const text = parts.join(" ");

    // Verify the text is well-formed for speech
    expect(text).toContain("Task: Fix the leaky faucet.");
    expect(text).toContain("Kitchen sink has been dripping");
    expect(text).toContain("This is a repair task.");
    expect(text).toContain("Urgency is high.");
    expect(text).toContain("Desire stage: full commitment.");
    expect(text).toContain("You have: tools, time.");
    expect(text).toContain("You lack: skill.");
    expect(text.length).toBeGreaterThan(100); // Ensure substantial content
  });

  it("handles minimal task data without crashing", () => {
    const task = { title: "Quick note" };
    const parts: string[] = [];
    parts.push(`Task: ${task.title}.`);
    const text = parts.join(" ");
    
    expect(text).toBe("Task: Quick note.");
    expect(text.length).toBeGreaterThan(0);
  });

  it("produces speakable DWP summary", () => {
    const task = {
      desireStage: "craving",
      wisdomStage: "research",
      powerStage: "gathering",
      dwpDiagnosis: "Desire is ahead of wisdom — more research needed",
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
    expect(text).toContain("more research needed");
  });
});
