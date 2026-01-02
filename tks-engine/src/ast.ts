/**
 * TKS Core Engine v0 - Abstract Syntax Tree Types
 *
 * These types represent the parsed structure of TKS expressions.
 * The structure matches the grammar specification exactly.
 */

import { FoundationId, IdeaId, NoeticIndex } from './types';

// ============================================================================
// MAIN EXPRESSION TYPE
// ============================================================================

export type TksExpression =
  | ImplicationExpr
  | BinaryExpr
  | UnaryExpr
  | FractalApplicationExpr
  | NoeticTermExpr
  | AtomicExpr;

// ============================================================================
// IMPLICATION EXPRESSION
// e.g., A -> B -> C (right-associative)
// ============================================================================

export interface ImplicationExpr {
  kind: "implication";
  left: TksExpression;
  right: TksExpression;
}

// ============================================================================
// BINARY EXPRESSION
// Conjunction (&) and Disjunction (|)
// ============================================================================

export interface BinaryExpr {
  kind: "binary";
  op: "&" | "|";
  left: TksExpression;
  right: TksExpression;
}

// ============================================================================
// UNARY EXPRESSION
// Negation (!)
// ============================================================================

export interface UnaryExpr {
  kind: "unary";
  op: "!";
  expr: TksExpression;
}

// ============================================================================
// FRACTAL APPLICATION EXPRESSION
// e.g., <<1:4:7>>(3d) or ⟨1:4:7⟩(3d)
// ============================================================================

export interface FractalApplicationExpr {
  kind: "fractalApplication";
  noetics: NoeticIndex[];    // from the noetic_list
  inner: TksExpression;
}

// ============================================================================
// NOETIC TERM EXPRESSION
// An atomic with optional noetic index
// e.g., 3d^2, B2^4, x^7
// ============================================================================

export interface NoeticTermExpr {
  kind: "noeticTerm";
  base: AtomicExpr;
  noetic: NoeticIndex;
}

// ============================================================================
// ATOMIC EXPRESSIONS
// The base terminal expressions
// ============================================================================

export type AtomicExpr =
  | FoundationAtom
  | IdeaAtom
  | IdentAtom
  | IntegerAtom;

export interface FoundationAtom {
  kind: "foundation";
  id: FoundationId;
}

export interface IdeaAtom {
  kind: "idea";
  id: IdeaId;
}

export interface IdentAtom {
  kind: "ident";
  name: string;
}

export interface IntegerAtom {
  kind: "integer";
  value: number;
}

// ============================================================================
// AST HELPER FUNCTIONS
// ============================================================================

/**
 * Creates an implication expression
 */
export function implication(left: TksExpression, right: TksExpression): ImplicationExpr {
  return { kind: "implication", left, right };
}

/**
 * Creates a binary expression (& or |)
 */
export function binary(op: "&" | "|", left: TksExpression, right: TksExpression): BinaryExpr {
  return { kind: "binary", op, left, right };
}

/**
 * Creates a unary expression (!)
 */
export function unary(expr: TksExpression): UnaryExpr {
  return { kind: "unary", op: "!", expr };
}

/**
 * Creates a fractal application expression
 */
export function fractalApplication(noetics: NoeticIndex[], inner: TksExpression): FractalApplicationExpr {
  return { kind: "fractalApplication", noetics, inner };
}

/**
 * Creates a noetic term expression
 */
export function noeticTerm(base: AtomicExpr, noetic: NoeticIndex): NoeticTermExpr {
  return { kind: "noeticTerm", base, noetic };
}

/**
 * Creates a foundation atom
 */
export function foundation(id: FoundationId): FoundationAtom {
  return { kind: "foundation", id };
}

/**
 * Creates an idea atom
 */
export function idea(id: IdeaId): IdeaAtom {
  return { kind: "idea", id };
}

/**
 * Creates an identifier atom
 */
export function ident(name: string): IdentAtom {
  return { kind: "ident", name };
}

/**
 * Creates an integer atom
 */
export function integer(value: number): IntegerAtom {
  return { kind: "integer", value };
}

/**
 * Pretty-prints a TKS expression to a string
 */
export function exprToString(expr: TksExpression): string {
  switch (expr.kind) {
    case "implication":
      return `(${exprToString(expr.left)} -> ${exprToString(expr.right)})`;
    case "binary":
      return `(${exprToString(expr.left)} ${expr.op} ${exprToString(expr.right)})`;
    case "unary":
      return `!${exprToString(expr.expr)}`;
    case "fractalApplication":
      return `<<${expr.noetics.join(":")}>>( ${exprToString(expr.inner)})`;
    case "noeticTerm":
      return `${atomToString(expr.base)}^${expr.noetic}`;
    case "foundation":
    case "idea":
    case "ident":
    case "integer":
      return atomToString(expr);
  }
}

function atomToString(atom: AtomicExpr): string {
  switch (atom.kind) {
    case "foundation":
      return atom.id;
    case "idea":
      return atom.id;
    case "ident":
      return atom.name;
    case "integer":
      return String(atom.value);
  }
}

/**
 * Checks if an expression is an atomic expression
 */
export function isAtomicExpr(expr: TksExpression): expr is AtomicExpr {
  return (
    expr.kind === "foundation" ||
    expr.kind === "idea" ||
    expr.kind === "ident" ||
    expr.kind === "integer"
  );
}
