/**
 * Deterministic Priority Scoring Engine
 * Computes a 0-100 score using the 6 weighted factors from TKS PRIORITY_WEIGHTS.
 */
import { PRIORITY_WEIGHTS, DWP_STAGES, FOUNDATIONS } from "../shared/tks";

interface PriorityInput {
  dueAt: Date | null;
  foundationId: number | null;
  foundationCounts: Record<number, number>; // how many tasks per foundation
  desireStage: string | null;
  powerItems: Array<{ status: string }>;
  urgency: string | null;
  recurrence: string | null;
}

/**
 * Compute deadline proximity score (0-100).
 * Closer deadlines score higher. No deadline = 30 (neutral).
 */
function scoreDeadlineProximity(dueAt: Date | null): number {
  if (!dueAt) return 30;
  const now = Date.now();
  const due = dueAt.getTime();
  const hoursLeft = (due - now) / (1000 * 60 * 60);
  if (hoursLeft <= 0) return 100;   // overdue
  if (hoursLeft <= 4) return 95;
  if (hoursLeft <= 12) return 85;
  if (hoursLeft <= 24) return 75;
  if (hoursLeft <= 48) return 60;
  if (hoursLeft <= 168) return 40;  // within a week
  return 20;
}

/**
 * Compute foundation neglect score (0-100).
 * Foundations with fewer tasks relative to the mean score higher.
 */
function scoreFoundationNeglect(foundationId: number | null, foundationCounts: Record<number, number>): number {
  if (!foundationId) return 50;
  const counts = FOUNDATIONS.map(f => foundationCounts[f.id] || 0);
  const total = counts.reduce((a, b) => a + b, 0);
  if (total === 0) return 50;
  const mean = total / 7;
  const thisCount = foundationCounts[foundationId] || 0;
  // If this foundation is below average, it's neglected → higher score
  if (mean === 0) return 50;
  const ratio = thisCount / mean;
  if (ratio <= 0.25) return 100;
  if (ratio <= 0.5) return 80;
  if (ratio <= 0.75) return 60;
  if (ratio <= 1.0) return 40;
  if (ratio <= 1.5) return 25;
  return 10; // over-indexed
}

/**
 * Compute desire stage score (0-100).
 * Higher desire = higher priority.
 */
function scoreDesireStage(desireStage: string | null): number {
  const map: Record<string, number> = {
    D_NOON: 100,
    D_SUNRISE: 75,
    D_SUNSET: 40,
    D_MIDNIGHT: 15,
  };
  return map[desireStage || ""] ?? 50;
}

/**
 * Compute power completeness score (0-100).
 * More "have" items = more ready = higher priority.
 */
function scorePowerCompleteness(powerItems: Array<{ status: string }>): number {
  if (powerItems.length === 0) return 50;
  let score = 0;
  for (const item of powerItems) {
    if (item.status === "have") score += 100;
    else if (item.status === "partial") score += 50;
    // "lack" = 0
  }
  return Math.round(score / powerItems.length);
}

/**
 * Compute urgency score (0-100).
 */
function scoreUrgency(urgency: string | null): number {
  const map: Record<string, number> = {
    critical: 100,
    high: 80,
    medium: 50,
    low: 20,
  };
  return map[urgency || ""] ?? 50;
}

/**
 * Compute rhythm score (0-100).
 * Recurring tasks get a moderate boost.
 */
function scoreRhythm(recurrence: string | null): number {
  if (!recurrence) return 40;
  // Recurring tasks have built-in rhythm — moderate priority
  return 65;
}

/**
 * Calculate the final weighted priority score (0-100).
 */
export function calculatePriorityScore(input: PriorityInput): number {
  const dp = scoreDeadlineProximity(input.dueAt);
  const fn = scoreFoundationNeglect(input.foundationId, input.foundationCounts);
  const ds = scoreDesireStage(input.desireStage);
  const pc = scorePowerCompleteness(input.powerItems);
  const ur = scoreUrgency(input.urgency);
  const rh = scoreRhythm(input.recurrence);

  const score =
    dp * PRIORITY_WEIGHTS.deadlineProximity +
    fn * PRIORITY_WEIGHTS.foundationNeglect +
    ds * PRIORITY_WEIGHTS.desireStage +
    pc * PRIORITY_WEIGHTS.powerCompleteness +
    ur * PRIORITY_WEIGHTS.urgency +
    rh * PRIORITY_WEIGHTS.rhythm;

  return Math.round(Math.max(0, Math.min(100, score)));
}

// Export individual scorers for testing
export const _scorers = {
  scoreDeadlineProximity,
  scoreFoundationNeglect,
  scoreDesireStage,
  scorePowerCompleteness,
  scoreUrgency,
  scoreRhythm,
};
