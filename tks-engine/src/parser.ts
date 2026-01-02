/**
 * TKS Core Engine v0 - Parser
 *
 * Parses TKS expressions according to the grammar specification.
 * Implements operator precedence: ! > & > | > ->
 */

import { Lexer, Token, TokenType, LexerError } from './lexer';
import {
  TksExpression,
  AtomicExpr,
  ImplicationExpr,
  BinaryExpr,
  UnaryExpr,
  FractalApplicationExpr,
  NoeticTermExpr,
  FoundationAtom,
  IdeaAtom,
  IdentAtom,
  IntegerAtom,
} from './ast';
import { FoundationId, IdeaId, NoeticIndex, isNoeticIndex, isFoundationId, isIdeaId } from './types';

// ============================================================================
// COMMAND TYPES (for REPL)
// ============================================================================

export type Command =
  | LetCommand
  | EvalCommand
  | ShowStateCommand
  | HelpCommand
  | CoachCommand
  | ExprCommand;

export interface LetCommand {
  kind: "let";
  name: string;
  expr: TksExpression;
}

export interface EvalCommand {
  kind: "eval";
  expr: TksExpression;
}

export interface ShowStateCommand {
  kind: "showState";
}

export interface HelpCommand {
  kind: "help";
}

export interface CoachCommand {
  kind: "coach";
  id: string;
}

export interface ExprCommand {
  kind: "expr";
  expr: TksExpression;
}

// ============================================================================
// PARSER CLASS
// ============================================================================

export class Parser {
  private tokens: Token[];
  private current: number = 0;

  constructor(input: string) {
    const lexer = new Lexer(input);
    this.tokens = lexer.tokenize();
  }

  /**
   * Parses a REPL command
   */
  parseCommand(): Command {
    // Check for 'let' command
    if (this.check("LET")) {
      return this.parseLetCommand();
    }

    // Check for 'eval' command
    if (this.check("EVAL")) {
      return this.parseEvalCommand();
    }

    // Check for 'show state' command
    if (this.check("SHOW")) {
      return this.parseShowCommand();
    }

    // Check for 'help' command
    if (this.check("HELP")) {
      this.advance();
      return { kind: "help" };
    }

    // Check for 'coach' command
    if (this.check("COACH")) {
      this.advance();
      const idToken = this.expect("IDENT", "Expected foundation id after 'coach'");
      return { kind: "coach", id: idToken.value };
    }

    // Otherwise, treat as a bare expression
    const expr = this.parseExpression();
    this.expectEnd();
    return { kind: "expr", expr };
  }

  private parseLetCommand(): LetCommand {
    this.advance(); // consume 'let'

    const nameToken = this.expect("IDENT", "Expected identifier after 'let'");
    const name = nameToken.value;

    this.expect("EQUALS", "Expected '=' after identifier");

    const expr = this.parseExpression();
    this.expectEnd();

    return { kind: "let", name, expr };
  }

  private parseEvalCommand(): EvalCommand {
    this.advance(); // consume 'eval'
    const expr = this.parseExpression();
    this.expectEnd();
    return { kind: "eval", expr };
  }

  private parseShowCommand(): Command {
    this.advance(); // consume 'show'

    if (this.check("STATE")) {
      this.advance();
      this.expectEnd();
      return { kind: "showState" };
    }

    throw this.error("Expected 'state' after 'show'");
  }

  /**
   * Parses a TKS expression (entry point for expression parsing)
   */
  parseExpression(): TksExpression {
    return this.parseImplication();
  }

  /**
   * Parses implication (->)
   * Right-associative: a -> b -> c = a -> (b -> c)
   */
  private parseImplication(): TksExpression {
    let left = this.parseDisjunction();

    if (this.check("IMPLIES")) {
      this.advance();
      const right = this.parseImplication(); // right-associative recursion
      return {
        kind: "implication",
        left,
        right,
      } as ImplicationExpr;
    }

    return left;
  }

  /**
   * Parses disjunction (|)
   * Left-associative: a | b | c = (a | b) | c
   */
  private parseDisjunction(): TksExpression {
    let left = this.parseConjunction();

    while (this.check("OR")) {
      this.advance();
      const right = this.parseConjunction();
      left = {
        kind: "binary",
        op: "|",
        left,
        right,
      } as BinaryExpr;
    }

    return left;
  }

  /**
   * Parses conjunction (&)
   * Left-associative: a & b & c = (a & b) & c
   */
  private parseConjunction(): TksExpression {
    let left = this.parseUnary();

    while (this.check("AND")) {
      this.advance();
      const right = this.parseUnary();
      left = {
        kind: "binary",
        op: "&",
        left,
        right,
      } as BinaryExpr;
    }

    return left;
  }

  /**
   * Parses unary expressions (!)
   */
  private parseUnary(): TksExpression {
    if (this.check("NOT")) {
      this.advance();
      const expr = this.parseUnary(); // right-recursive for multiple !
      return {
        kind: "unary",
        op: "!",
        expr,
      } as UnaryExpr;
    }

    return this.parsePrimary();
  }

  /**
   * Parses primary expressions:
   * - fractal_application
   * - noetic_term
   * - atomic
   * - parenthesized expression
   */
  private parsePrimary(): TksExpression {
    // Check for fractal application: << ... >> ( expr )
    if (this.check("FRACTAL_OPEN")) {
      return this.parseFractalApplication();
    }

    // Check for parenthesized expression
    if (this.check("LPAREN")) {
      this.advance();
      const expr = this.parseExpression();
      this.expect("RPAREN", "Expected ')' after expression");
      return expr;
    }

    // Parse atomic, possibly with noetic term suffix
    return this.parseNoeticTermOrAtomic();
  }

  /**
   * Parses a fractal application: << noetic_list >> ( expr )
   */
  private parseFractalApplication(): FractalApplicationExpr {
    this.advance(); // consume <<

    const noetics = this.parseNoeticList();

    this.expect("FRACTAL_CLOSE", "Expected '>>' after noetic list");
    this.expect("LPAREN", "Expected '(' after fractal");

    const inner = this.parseExpression();

    this.expect("RPAREN", "Expected ')' after fractal argument");

    return {
      kind: "fractalApplication",
      noetics,
      inner,
    };
  }

  /**
   * Parses a noetic list: idx : idx : idx ...
   */
  private parseNoeticList(): NoeticIndex[] {
    const noetics: NoeticIndex[] = [];

    const first = this.parseNoeticIndex();
    noetics.push(first);

    while (this.check("COLON")) {
      this.advance();
      const next = this.parseNoeticIndex();
      noetics.push(next);
    }

    return noetics;
  }

  /**
   * Parses a single noetic index (0-9)
   */
  private parseNoeticIndex(): NoeticIndex {
    const token = this.expect("INTEGER", "Expected noetic index (0-9)");
    const value = parseInt(token.value, 10);

    if (!isNoeticIndex(value)) {
      throw this.errorAt(token, `Invalid noetic index: ${value}. Must be 0-9.`);
    }

    return value;
  }

  /**
   * Parses a noetic term (atomic^idx) or just an atomic
   */
  private parseNoeticTermOrAtomic(): TksExpression {
    const atomic = this.parseAtomic();

    // Check for noetic suffix: ^idx
    if (this.check("CARET")) {
      this.advance();
      const token = this.expect("INTEGER", "Expected noetic index after '^'");
      const value = parseInt(token.value, 10);

      if (!isNoeticIndex(value)) {
        throw this.errorAt(token, `Invalid noetic index: ${value}. Must be 0-9.`);
      }

      return {
        kind: "noeticTerm",
        base: atomic,
        noetic: value,
      } as NoeticTermExpr;
    }

    return atomic;
  }

  /**
   * Parses an atomic expression: FOUNDATION | IDEA_REF | IDENT | INTEGER
   */
  private parseAtomic(): AtomicExpr {
    const token = this.peek();

    if (this.check("FOUNDATION")) {
      this.advance();
      if (!isFoundationId(token.value)) {
        throw this.errorAt(token, `Invalid foundation ID: ${token.value}`);
      }
      return {
        kind: "foundation",
        id: token.value as FoundationId,
      } as FoundationAtom;
    }

    if (this.check("IDEA_REF")) {
      this.advance();
      if (!isIdeaId(token.value)) {
        throw this.errorAt(token, `Invalid idea ID: ${token.value}`);
      }
      return {
        kind: "idea",
        id: token.value as IdeaId,
      } as IdeaAtom;
    }

    if (this.check("IDENT")) {
      this.advance();
      return {
        kind: "ident",
        name: token.value,
      } as IdentAtom;
    }

    if (this.check("INTEGER")) {
      this.advance();
      return {
        kind: "integer",
        value: parseInt(token.value, 10),
      } as IntegerAtom;
    }

    throw this.error(`Unexpected token: ${token.type} (${token.value})`);
  }

  // ============================================================================
  // HELPER METHODS
  // ============================================================================

  private peek(): Token {
    return this.tokens[this.current];
  }

  private check(type: TokenType): boolean {
    return this.peek().type === type;
  }

  private advance(): Token {
    const token = this.peek();
    if (token.type !== "EOF") {
      this.current++;
    }
    return token;
  }

  private expect(type: TokenType, message: string): Token {
    const token = this.peek();
    if (token.type !== type) {
      throw this.errorAt(token, `${message}. Got ${token.type} (${token.value})`);
    }
    return this.advance();
  }

  private expectEnd(): void {
    const token = this.peek();
    if (token.type !== "EOF") {
      throw this.errorAt(token, `Unexpected token after expression: ${token.type} (${token.value})`);
    }
  }

  private error(message: string): ParseError {
    const token = this.peek();
    return new ParseError(message, token.line, token.column);
  }

  private errorAt(token: Token, message: string): ParseError {
    return new ParseError(message, token.line, token.column);
  }
}

// ============================================================================
// PARSE ERROR
// ============================================================================

export class ParseError extends Error {
  line: number;
  column: number;

  constructor(message: string, line: number, column: number) {
    super(`Parse error at line ${line}, column ${column}: ${message}`);
    this.name = "ParseError";
    this.line = line;
    this.column = column;
  }
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

/**
 * Parses a TKS expression string
 */
export function parseExpression(input: string): TksExpression {
  const parser = new Parser(input);
  const expr = parser.parseExpression();
  return expr;
}

/**
 * Parses a REPL command string
 */
export function parseCommand(input: string): Command {
  const parser = new Parser(input);
  return parser.parseCommand();
}
