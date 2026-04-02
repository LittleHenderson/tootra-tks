import { describe, expect, it } from "vitest";
import { calculatePriorityScore, _scorers } from "./priority";

const { scoreDeadlineProximity, scoreFoundationNeglect, scoreDesireStage, scorePowerCompleteness, scoreUrgency, scoreRhythm } = _scorers;

describe("Priority Scoring Engine", () => {
  describe("scoreDeadlineProximity", () => {
    it("returns 100 for overdue tasks", () => {
      const past = new Date(Date.now() - 3600000);
      expect(scoreDeadlineProximity(past)).toBe(100);
    });

    it("returns 30 for tasks with no deadline", () => {
      expect(scoreDeadlineProximity(null)).toBe(30);
    });

    it("returns high score for tasks due within 4 hours", () => {
      const soon = new Date(Date.now() + 2 * 3600000);
      expect(scoreDeadlineProximity(soon)).toBe(95);
    });

    it("returns lower score for tasks due in a week+", () => {
      const far = new Date(Date.now() + 10 * 24 * 3600000);
      expect(scoreDeadlineProximity(far)).toBe(20);
    });
  });

  describe("scoreFoundationNeglect", () => {
    it("returns 50 when no foundation is assigned", () => {
      expect(scoreFoundationNeglect(null, {})).toBe(50);
    });

    it("returns high score for neglected foundation", () => {
      const counts = { 1: 10, 2: 10, 3: 10, 4: 10, 5: 10, 6: 10, 7: 1 };
      expect(scoreFoundationNeglect(7, counts)).toBeGreaterThanOrEqual(80);
    });

    it("returns low score for over-indexed foundation", () => {
      const counts = { 1: 20, 2: 5, 3: 5, 4: 5, 5: 5, 6: 5, 7: 5 };
      expect(scoreFoundationNeglect(1, counts)).toBeLessThanOrEqual(25);
    });
  });

  describe("scoreDesireStage", () => {
    it("returns 100 for D_NOON (peak desire)", () => {
      expect(scoreDesireStage("D_NOON")).toBe(100);
    });

    it("returns 15 for D_MIDNIGHT (dormant)", () => {
      expect(scoreDesireStage("D_MIDNIGHT")).toBe(15);
    });

    it("returns 50 for unknown stage", () => {
      expect(scoreDesireStage(null)).toBe(50);
    });
  });

  describe("scorePowerCompleteness", () => {
    it("returns 100 when all items are have", () => {
      const items = [{ status: "have" }, { status: "have" }, { status: "have" }];
      expect(scorePowerCompleteness(items)).toBe(100);
    });

    it("returns 0 when all items are lack", () => {
      const items = [{ status: "lack" }, { status: "lack" }];
      expect(scorePowerCompleteness(items)).toBe(0);
    });

    it("returns 50 for empty power items", () => {
      expect(scorePowerCompleteness([])).toBe(50);
    });

    it("returns 50 for all partial items", () => {
      const items = [{ status: "partial" }, { status: "partial" }];
      expect(scorePowerCompleteness(items)).toBe(50);
    });
  });

  describe("scoreUrgency", () => {
    it("returns 100 for critical", () => {
      expect(scoreUrgency("critical")).toBe(100);
    });

    it("returns 20 for low", () => {
      expect(scoreUrgency("low")).toBe(20);
    });
  });

  describe("scoreRhythm", () => {
    it("returns 40 for non-recurring tasks", () => {
      expect(scoreRhythm(null)).toBe(40);
    });

    it("returns 65 for recurring tasks", () => {
      expect(scoreRhythm("weekly")).toBe(65);
    });
  });

  describe("calculatePriorityScore", () => {
    it("returns a number between 0 and 100", () => {
      const score = calculatePriorityScore({
        dueAt: null,
        foundationId: 1,
        foundationCounts: { 1: 5, 2: 5, 3: 5, 4: 5, 5: 5, 6: 5, 7: 5 },
        desireStage: "D_SUNRISE",
        powerItems: [{ status: "have" }, { status: "lack" }],
        urgency: "medium",
        recurrence: null,
      });
      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(100);
    });

    it("high urgency overdue task scores higher than low urgency distant task", () => {
      const highScore = calculatePriorityScore({
        dueAt: new Date(Date.now() - 3600000),
        foundationId: 1,
        foundationCounts: {},
        desireStage: "D_NOON",
        powerItems: [{ status: "have" }],
        urgency: "critical",
        recurrence: null,
      });
      const lowScore = calculatePriorityScore({
        dueAt: new Date(Date.now() + 30 * 24 * 3600000),
        foundationId: 1,
        foundationCounts: { 1: 50 },
        desireStage: "D_MIDNIGHT",
        powerItems: [{ status: "lack" }],
        urgency: "low",
        recurrence: null,
      });
      expect(highScore).toBeGreaterThan(lowScore);
    });

    it("weights sum correctly in the output", () => {
      // All factors at 50 should produce ~50
      const score = calculatePriorityScore({
        dueAt: null, // 30
        foundationId: null, // 50
        foundationCounts: {},
        desireStage: null, // 50
        powerItems: [], // 50
        urgency: "medium", // 50
        recurrence: null, // 40
      });
      // Expected: 30*0.25 + 50*0.20 + 50*0.15 + 50*0.15 + 50*0.15 + 40*0.10 = 7.5+10+7.5+7.5+7.5+4 = 44
      expect(score).toBe(44);
    });
  });
});
