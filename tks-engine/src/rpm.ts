import { AcquisitionId, FoundationId, TksState } from "./types";

export interface RpmCheckResult {
  goal: FoundationId;
  required: AcquisitionId[];
  present: AcquisitionId[];
  missing: AcquisitionId[];
  notes: string[];
}

/**
 * Returns the list of acquisitions required to work on a foundation goal.
 * v1 hardcoded mapping:
 *  - 5d (worldly power/influence/control) -> 13, 14, 15
 *  - 6d (material food/thing/money) -> 16, 17, 18
 *  - others -> none
 */
export function getRequiredAcquisitions(goal: FoundationId): AcquisitionId[] {
  if (goal === "5d") {
    return [13, 14, 15];
  }
  if (goal === "6d") {
    return [16, 17, 18];
  }
  return [];
}

/**
 * Performs a prerequisite check for a foundation goal based on RPM v1 mapping.
 * Does not mutate state.
 */
export function checkPrereqs(goal: FoundationId, state: TksState): RpmCheckResult {
  const required = getRequiredAcquisitions(goal);
  const present: AcquisitionId[] = [];
  const missing: AcquisitionId[] = [];

  for (const id of required) {
    if (state.acquisitions[id]) {
      present.push(id);
    } else {
      missing.push(id);
    }
  }

  const notes: string[] = [];
  if (required.length === 0) {
    notes.push("No RPM prerequisites defined for this foundation in v1.");
  } else if (missing.length === 0) {
    notes.push("All RPM prerequisites satisfied for this foundation in v1.");
  } else {
    notes.push(`Missing acquisitions: ${missing.join(", ")}`);
  }

  return {
    goal,
    required,
    present,
    missing,
    notes,
  };
}
