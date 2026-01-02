/**
 * TKS Core Engine v0 - Evaluator
 *
 * Evaluates TKS expressions according to the semantics specification.
 * Implements stub semantics for Noetics (v0).
 */

import {
  TksExpression,
  AtomicExpr,
  exprToString,
  isAtomicExpr,
} from './ast';
import {
  TksState,
  TksValue,
  NoeticIndex,
  StructuralValue,
  FoundationId,
  isFoundationId,
} from './types';

// ============================================================================
// EVALUATION RESULT
// ============================================================================

export interface EvaluationResult {
  value: TksValue;
  state: TksState;
  trace: string[];
}

// ============================================================================
// NOETIC FUNCTIONS
// ============================================================================

/**
 * Extracts foundation targets embedded in a value for numeric noetic updates.
 */
function getFoundationTargets(value: TksValue): FoundationId[] {
  const results = new Set<FoundationId>();

  const collect = (v: TksValue): void => {
    if (typeof v === "string" && isFoundationId(v)) {
      results.add(v);
      return;
    }

    if (typeof v === "object" && v !== null && "kind" in v) {
      const sv = v as StructuralValue;
      if (sv.kind === "aware" || sv.kind === "vibration" || sv.kind === "rhythm") {
        if (sv.inner !== undefined) {
          collect(sv.inner);
        }
        return;
      }

      if (sv.kind === "and" || sv.kind === "or") {
        if (sv.left !== undefined) {
          collect(sv.left);
        }
        if (sv.right !== undefined) {
          collect(sv.right);
        }
        return;
      }
    }
  };

  collect(value);
  return Array.from(results);
}

export type NoeticFn = (value: TksValue, state: TksState) => {
  value: TksValue;
  state: TksState;
  note: string;
};

/**
 * Numeric deltas applied to foundations when Noetics are used.
 * These represent simple doctrine-aligned increments for v0 numeric semantics.
 */
const NOETIC_DELTA: Record<NoeticIndex, number> = {
  0: 0.0,
  1: 0.05,
  2: 0.0,
  3: 0.0,
  4: 0.10,
  5: 0.0,
  6: 0.0,
  7: 0.15,
  8: 0.0,
  9: 0.0,
};

/**
 * Applies numeric foundation deltas for a given noetic index and value.
 * Mutates state in-place, clamping to [0, 1].
 */
function applyFoundationDelta(noeticIndex: NoeticIndex, state: TksState, value: TksValue): void {
  const delta = NOETIC_DELTA[noeticIndex];
  if (delta === 0) {
    return;
  }

  const targets = getFoundationTargets(value);
  for (const id of targets) {
    const oldVal = state.foundations[id] ?? 0;
    const newVal = Math.max(0, Math.min(1, oldVal + delta));
    state.foundations[id] = newVal;
  }
}

/**
 * v0 Noetic function implementations:
 * - IŤ0: identity
 * - IŤ1: awareness-tag
 * - IŤ4: vibration-tag
 * - IŤ7: rhythm-tag
 * - All others: no-op (v0)
 */
export const applyNoetic: Record<NoeticIndex, NoeticFn> = {
  0: (value, state) => {
    applyFoundationDelta(0, state, value);
    return {
      value,
      state,
      note: "ν0: identity (IŤ0)",
    };
  },

  1: (value, state) => {
    applyFoundationDelta(1, state, value);
    const wrapped = {
      kind: "aware",
      inner: value,
    } as StructuralValue;
    return {
      value: wrapped,
      state,
      note: "ν1: awareness applied (IŤ1)",
    };
  },

  2: (value, state) => {
    applyFoundationDelta(2, state, value);
    return {
      value,
      state,
      note: "ν2: no-op (v0) (IŤ2)",
    };
  },

  3: (value, state) => {
    applyFoundationDelta(3, state, value);
    return {
      value,
      state,
      note: "ν3: no-op (v0) (IŤ3)",
    };
  },

  4: (value, state) => {
    applyFoundationDelta(4, state, value);
    const wrapped = {
      kind: "vibration",
      inner: value,
    } as StructuralValue;
    return {
      value: wrapped,
      state,
      note: "ν4: vibration applied (IŤ4)",
    };
  },

  5: (value, state) => {
    applyFoundationDelta(5, state, value);
    return {
      value,
      state,
      note: "ν5: no-op (v0) (IŤ5)",
    };
  },

  6: (value, state) => {
    applyFoundationDelta(6, state, value);
    return {
      value,
      state,
      note: "ν6: no-op (v0) (IŤ6)",
    };
  },

  7: (value, state) => {
    applyFoundationDelta(7, state, value);
    const wrapped = {
      kind: "rhythm",
      inner: value,
    } as StructuralValue;
    return {
      value: wrapped,
      state,
      note: "ν7: rhythm applied (IŤ7)",
    };
  },

  8: (value, state) => {
    applyFoundationDelta(8, state, value);
    return {
      value,
      state,
      note: "ν8: no-op (v0) (IŤ8)",
    };
  },

  9: (value, state) => {
    applyFoundationDelta(9, state, value);
    return {
      value,
      state,
      note: "ν9: no-op (v0) (IŤ9)",
    };
  },
};

// ============================================================================
// MAIN EVALUATOR
// ============================================================================

/**
 * Evaluates a TKS expression within the given state
 */
export function evaluate(expr: TksExpression, state: TksState): EvaluationResult {
  const trace: string[] = [];

  const result = evaluateExpr(expr, state, trace);

  return {
    value: result.value,
    state: result.state,
    trace,
  };
}

function evaluateExpr(
  expr: TksExpression,
  state: TksState,
  trace: string[]
): { value: TksValue; state: TksState } {
  switch (expr.kind) {
    case "foundation":
      return evaluateFoundation(expr, state, trace);

    case "idea":
      return evaluateIdea(expr, state, trace);

    case "ident":
      return evaluateIdent(expr, state, trace);

    case "integer":
      return evaluateInteger(expr, state, trace);

    case "unary":
      return evaluateUnary(expr, state, trace);

    case "binary":
      return evaluateBinary(expr, state, trace);

    case "implication":
      return evaluateImplication(expr, state, trace);

    case "fractalApplication":
      return evaluateFractal(expr, state, trace);

    case "noeticTerm":
      return evaluateNoeticTerm(expr, state, trace);

    default:
      throw new EvaluationError(`Unknown expression kind: ${(expr as any).kind}`);
  }
}

// ============================================================================
// ATOMIC EVALUATION
// ============================================================================

function evaluateFoundation(
  expr: { kind: "foundation"; id: string },
  state: TksState,
  trace: string[]
): { value: TksValue; state: TksState } {
  trace.push(`foundation ${expr.id}`);
  return { value: expr.id, state };
}

function evaluateIdea(
  expr: { kind: "idea"; id: string },
  state: TksState,
  trace: string[]
): { value: TksValue; state: TksState } {
  trace.push(`idea ${expr.id}`);
  return { value: expr.id, state };
}

function evaluateIdent(
  expr: { kind: "ident"; name: string },
  state: TksState,
  trace: string[]
): { value: TksValue; state: TksState } {
  const { name } = expr;

  if (!(name in state.vars)) {
    throw new EvaluationError(`Unknown identifier: ${name}`);
  }

  const value = state.vars[name];
  trace.push(`ident ${name} = ${valueToString(value)}`);
  return { value, state };
}

function evaluateInteger(
  expr: { kind: "integer"; value: number },
  state: TksState,
  trace: string[]
): { value: TksValue; state: TksState } {
  trace.push(`integer ${expr.value}`);
  return { value: expr.value, state };
}

// ============================================================================
// UNARY AND BINARY OPERATORS
// ============================================================================

function evaluateUnary(
  expr: { kind: "unary"; op: "!"; expr: TksExpression },
  state: TksState,
  trace: string[]
): { value: TksValue; state: TksState } {
  const inner = evaluateExpr(expr.expr, state, trace);

  // If the value is a boolean, apply logical NOT
  if (typeof inner.value === "boolean") {
    const result = !inner.value;
    trace.push(`NOT ${inner.value} = ${result}`);
    return { value: result, state: inner.state };
  }

  // If the value is a number (treat 0 as false, non-zero as true)
  if (typeof inner.value === "number") {
    const boolVal = inner.value !== 0;
    const result = !boolVal;
    trace.push(`NOT ${inner.value} (as bool: ${boolVal}) = ${result}`);
    return { value: result, state: inner.state };
  }

  // For non-boolean values, throw an error (per spec)
  throw new EvaluationError(
    `Cannot apply NOT to non-boolean value: ${valueToString(inner.value)}`
  );
}

function evaluateBinary(
  expr: { kind: "binary"; op: "&" | "|"; left: TksExpression; right: TksExpression },
  state: TksState,
  trace: string[]
): { value: TksValue; state: TksState } {
  const leftResult = evaluateExpr(expr.left, state, trace);
  const rightResult = evaluateExpr(expr.right, leftResult.state, trace);

  const leftVal = leftResult.value;
  const rightVal = rightResult.value;

  // Convert to booleans if both are numbers or booleans
  const leftBool = toBoolOrNull(leftVal);
  const rightBool = toBoolOrNull(rightVal);

  if (leftBool !== null && rightBool !== null) {
    let result: boolean;
    if (expr.op === "&") {
      result = leftBool && rightBool;
      trace.push(`${leftBool} AND ${rightBool} = ${result}`);
    } else {
      result = leftBool || rightBool;
      trace.push(`${leftBool} OR ${rightBool} = ${result}`);
    }
    return { value: result, state: rightResult.state };
  }

  // Otherwise return structural value
  const structural: StructuralValue = {
    kind: expr.op === "&" ? "and" : "or",
    left: leftVal,
    right: rightVal,
  };
  trace.push(`${expr.op === "&" ? "AND" : "OR"} structural: (${valueToString(leftVal)}, ${valueToString(rightVal)})`);
  return { value: structural, state: rightResult.state };
}

function evaluateImplication(
  expr: { kind: "implication"; left: TksExpression; right: TksExpression },
  state: TksState,
  trace: string[]
): { value: TksValue; state: TksState } {
  const leftResult = evaluateExpr(expr.left, state, trace);
  const rightResult = evaluateExpr(expr.right, leftResult.state, trace);

  // For v0: treat as structural implication, no state update
  const structural: StructuralValue = {
    kind: "implication",
    from: leftResult.value,
    to: rightResult.value,
  };

  trace.push(`implication: ${valueToString(leftResult.value)} -> ${valueToString(rightResult.value)}`);
  return { value: structural, state: rightResult.state };
}

// ============================================================================
// FRACTAL EVALUATION
// ============================================================================

function evaluateFractal(
  expr: { kind: "fractalApplication"; noetics: NoeticIndex[]; inner: TksExpression },
  state: TksState,
  trace: string[]
): { value: TksValue; state: TksState } {
  // First evaluate the inner expression
  const innerResult = evaluateExpr(expr.inner, state, trace);

  let value = innerResult.value;
  let currentState = innerResult.state;

  // Apply each noetic in sequence
  for (const n of expr.noetics) {
    const fn = applyNoetic[n];
    const before = value;
    const out = fn(value, currentState);
    value = out.value;
    currentState = out.state;
    trace.push(`Applied ν${n} (IŤ${n}) to ${valueToString(before)} -> ${valueToString(value)} | ${out.note}`);
  }

  return { value, state: currentState };
}

// ============================================================================
// NOETIC TERM EVALUATION
// ============================================================================

function evaluateNoeticTerm(
  expr: { kind: "noeticTerm"; base: AtomicExpr; noetic: NoeticIndex },
  state: TksState,
  trace: string[]
): { value: TksValue; state: TksState } {
  // Evaluate the atomic base
  const baseResult = evaluateExpr(expr.base, state, trace);

  // Apply the single noetic
  const fn = applyNoetic[expr.noetic];
  const before = baseResult.value;
  const out = fn(before, baseResult.state);

  trace.push(`Applied ^${expr.noetic} (ν${expr.noetic} / IŤ${expr.noetic}) to ${valueToString(before)} | ${out.note}`);

  return { value: out.value, state: out.state };
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Converts a value to boolean if it's a number or boolean, otherwise returns null
 */
function toBoolOrNull(value: TksValue): boolean | null {
  if (typeof value === "boolean") {
    return value;
  }
  if (typeof value === "number") {
    return value !== 0;
  }
  return null;
}

/**
 * Converts a TKS value to a string representation
 */
export function valueToString(value: TksValue): string {
  if (value === null || value === undefined) {
    return "undefined";
  }

  if (typeof value === "boolean") {
    return String(value);
  }

  if (typeof value === "number") {
    return String(value);
  }

  if (typeof value === "string") {
    return value;
  }

  // Check if it's an AST node (has a 'kind' property)
  if (typeof value === "object" && "kind" in value) {
    const obj = value as { kind: string };

    // Structural values
    if (obj.kind === "not") {
      const sv = value as StructuralValue;
      return `NOT(${valueToString(sv.inner!)})`;
    }
    if (obj.kind === "and") {
      const sv = value as StructuralValue;
      return `AND(${valueToString(sv.left!)}, ${valueToString(sv.right!)})`;
    }
    if (obj.kind === "or") {
      const sv = value as StructuralValue;
      return `OR(${valueToString(sv.left!)}, ${valueToString(sv.right!)})`;
    }
    if (obj.kind === "implication") {
      const sv = value as StructuralValue;
      return `IMPL(${valueToString(sv.from!)} -> ${valueToString(sv.to!)})`;
    }
    if (obj.kind === "aware") {
      const sv = value as StructuralValue;
      return `AWARE(${valueToString(sv.inner!)})`;
    }
    if (obj.kind === "vibration") {
      const sv = value as StructuralValue;
      return `VIBRATION(${valueToString(sv.inner!)})`;
    }
    if (obj.kind === "rhythm") {
      const sv = value as StructuralValue;
      return `RHYTHM(${valueToString(sv.inner!)})`;
    }
    if (obj.kind === "noetic") {
      const sv = value as StructuralValue;
      return `NOETIC[IŤ${sv.noeticIndex}](${valueToString(sv.inner!)})`;
    }

    // AST expression nodes - use exprToString
    if (isExpressionKind(obj.kind)) {
      return exprToString(value as TksExpression);
    }

    return JSON.stringify(value);
  }

  return JSON.stringify(value);
}

function isExpressionKind(kind: string): boolean {
  return [
    "implication",
    "binary",
    "unary",
    "fractalApplication",
    "noeticTerm",
    "foundation",
    "idea",
    "ident",
    "integer",
  ].includes(kind);
}

// ============================================================================
// EVALUATION ERROR
// ============================================================================

export class EvaluationError extends Error {
  constructor(message: string) {
    super(`Evaluation error: ${message}`);
    this.name = "EvaluationError";
  }
}
