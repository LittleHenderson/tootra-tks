/**
 * TKS Core Engine v0 - Type Definitions
 *
 * These types are canonical and must not be renamed without explicit user instruction.
 */

// ============================================================================
// FOUNDATIONS (28)
// Format: digit (1-7) + letter (a-d)
// Example: "3d" = physical life/vitality/health
// ============================================================================

export type FoundationId =
  | "1a" | "2a" | "3a" | "4a" | "5a" | "6a" | "7a"
  | "1b" | "2b" | "3b" | "4b" | "5b" | "6b" | "7b"
  | "1c" | "2c" | "3c" | "4c" | "5c" | "6c" | "7c"
  | "1d" | "2d" | "3d" | "4d" | "5d" | "6d" | "7d";

export const ALL_FOUNDATION_IDS: FoundationId[] = [
  "1a", "2a", "3a", "4a", "5a", "6a", "7a",
  "1b", "2b", "3b", "4b", "5b", "6b", "7b",
  "1c", "2c", "3c", "4c", "5c", "6c", "7c",
  "1d", "2d", "3d", "4d", "5d", "6d", "7d",
];

export function isFoundationId(value: string): value is FoundationId {
  return /^[1-7][a-d]$/.test(value);
}

// ============================================================================
// IDEAS / ELEMENTS (40)
// Format: letter (A-D) + digit (1-10)
// Example: "B2" = mental positive (semantics are domain-specific)
// ============================================================================

export type IdeaId =
  | "A1" | "A2" | "A3" | "A4" | "A5" | "A6" | "A7" | "A8" | "A9" | "A10"
  | "B1" | "B2" | "B3" | "B4" | "B5" | "B6" | "B7" | "B8" | "B9" | "B10"
  | "C1" | "C2" | "C3" | "C4" | "C5" | "C6" | "C7" | "C8" | "C9" | "C10"
  | "D1" | "D2" | "D3" | "D4" | "D5" | "D6" | "D7" | "D8" | "D9" | "D10";

export const ALL_IDEA_IDS: IdeaId[] = [
  "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10",
  "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10",
  "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
  "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10",
];

export function isIdeaId(value: string): value is IdeaId {
  return /^[A-D](10|[1-9])$/.test(value);
}

// ============================================================================
// ACQUISITIONS (22)
// Integers 1-22, representing acquired states
// ============================================================================

export type AcquisitionId =
  | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11
  | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22;

export const ALL_ACQUISITION_IDS: AcquisitionId[] = [
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
  12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
];

export function isAcquisitionId(value: number): value is AcquisitionId {
  return Number.isInteger(value) && value >= 1 && value <= 22;
}

// ============================================================================
// NOETICS (10)
// Indices 0-9, representing structural tags and transformation functions
// ============================================================================

export type NoeticIndex = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9;

export const ALL_NOETIC_INDICES: NoeticIndex[] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

export function isNoeticIndex(value: number): value is NoeticIndex {
  return Number.isInteger(value) && value >= 0 && value <= 9;
}

// ============================================================================
// TKS VALUE
// The set of possible values in the TKS system
// ============================================================================

// Note: TksExpression is defined in ast.ts to avoid circular dependency.
// TksValue can hold AST nodes via the 'object' type check in evaluator.

export type TksValue =
  | number
  | boolean
  | string
  | FoundationId
  | IdeaId
  | StructuralValue;

// Structural values for non-boolean operations
export interface StructuralValue {
  kind: "not" | "and" | "or" | "implication" | "aware" | "vibration" | "rhythm" | "noetic";
  inner?: TksValue;
  left?: TksValue;
  right?: TksValue;
  from?: TksValue;
  to?: TksValue;
  noeticIndex?: NoeticIndex;
}

// ============================================================================
// TKS STATE
// The complete state model for the TKS system
// ============================================================================

export interface TksState {
  foundations: Record<FoundationId, number>;     // 0.0 - 1.0 (activation level)
  acquisitions: Record<AcquisitionId, boolean>;  // true = acquired, false = not
  currentNoetic: NoeticIndex;                    // global Noetic focus
  currentElement?: IdeaId;                       // e.g. "A8" for today's element
  vars: Record<string, TksValue>;                // REPL variables
}

/**
 * Creates a default initialized TKS state
 */
export function createDefaultState(): TksState {
  const foundations: Record<FoundationId, number> = {} as Record<FoundationId, number>;
  for (const id of ALL_FOUNDATION_IDS) {
    foundations[id] = 0.0;
  }

  const acquisitions: Record<AcquisitionId, boolean> = {} as Record<AcquisitionId, boolean>;
  for (const id of ALL_ACQUISITION_IDS) {
    acquisitions[id] = false;
  }

  return {
    foundations,
    acquisitions,
    currentNoetic: 0,
    currentElement: undefined,
    vars: {},
  };
}
