/**
 * TKS Core Engine v0
 *
 * Main entry point - exports all public types and functions.
 */

// Types
export {
  FoundationId,
  IdeaId,
  AcquisitionId,
  NoeticIndex,
  TksValue,
  TksState,
  StructuralValue,
  ALL_FOUNDATION_IDS,
  ALL_IDEA_IDS,
  ALL_ACQUISITION_IDS,
  ALL_NOETIC_INDICES,
  isFoundationId,
  isIdeaId,
  isAcquisitionId,
  isNoeticIndex,
  createDefaultState,
} from './types';

// AST
export {
  TksExpression,
  ImplicationExpr,
  BinaryExpr,
  UnaryExpr,
  FractalApplicationExpr,
  NoeticTermExpr,
  AtomicExpr,
  FoundationAtom,
  IdeaAtom,
  IdentAtom,
  IntegerAtom,
  implication,
  binary,
  unary,
  fractalApplication,
  noeticTerm,
  foundation,
  idea,
  ident,
  integer,
  exprToString,
  isAtomicExpr,
} from './ast';

// Lexer
export {
  Lexer,
  Token,
  TokenType,
  LexerError,
} from './lexer';

// Parser
export {
  Parser,
  Command,
  LetCommand,
  EvalCommand,
  ShowStateCommand,
  HelpCommand,
  CoachCommand,
  ExprCommand,
  ParseError,
  parseExpression,
  parseCommand,
} from './parser';

// Evaluator
export {
  evaluate,
  EvaluationResult,
  NoeticFn,
  applyNoetic,
  valueToString,
  EvaluationError,
} from './evaluator';

// REPL
export { TksRepl } from './repl';
