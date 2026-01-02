/**
 * TKS Core Engine v0 - Lexer/Tokenizer
 *
 * Tokenizes TKS expression strings according to the grammar specification.
 */

// ============================================================================
// TOKEN TYPES
// ============================================================================

export type TokenType =
  | "FOUNDATION"     // 1a, 3d, 7c, etc.
  | "IDEA_REF"       // A1, B10, D9, etc.
  | "INTEGER"        // 0, 1, 42, etc.
  | "IDENT"          // variable names
  | "AND"            // &
  | "OR"             // |
  | "NOT"            // !
  | "IMPLIES"        // ->
  | "CARET"          // ^
  | "COLON"          // :
  | "LPAREN"         // (
  | "RPAREN"         // )
  | "FRACTAL_OPEN"   // << or ⟨
  | "FRACTAL_CLOSE"  // >> or ⟩
  | "EQUALS"         // =
  | "LET"            // let keyword
  | "EVAL"           // eval keyword
  | "SHOW"           // show keyword
  | "STATE"          // state keyword
  | "HELP"           // help keyword
  | "COACH"          // coach keyword
  | "EOF";           // end of input

export interface Token {
  type: TokenType;
  value: string;
  position: number;    // character position in input
  line: number;        // line number (1-indexed)
  column: number;      // column number (1-indexed)
}

// ============================================================================
// LEXER CLASS
// ============================================================================

export class Lexer {
  private input: string;
  private position: number = 0;
  private line: number = 1;
  private column: number = 1;

  constructor(input: string) {
    this.input = input;
  }

  /**
   * Tokenizes the entire input and returns all tokens
   */
  tokenize(): Token[] {
    const tokens: Token[] = [];
    let token = this.nextToken();
    while (token.type !== "EOF") {
      tokens.push(token);
      token = this.nextToken();
    }
    tokens.push(token); // include EOF
    return tokens;
  }

  /**
   * Returns the next token from the input
   */
  nextToken(): Token {
    this.skipWhitespace();

    if (this.position >= this.input.length) {
      return this.makeToken("EOF", "");
    }

    const startPos = this.position;
    const startLine = this.line;
    const startCol = this.column;

    // Check for two-character operators first
    if (this.match("->")) {
      return { type: "IMPLIES", value: "->", position: startPos, line: startLine, column: startCol };
    }
    if (this.match("<<")) {
      return { type: "FRACTAL_OPEN", value: "<<", position: startPos, line: startLine, column: startCol };
    }
    if (this.match(">>")) {
      return { type: "FRACTAL_CLOSE", value: ">>", position: startPos, line: startLine, column: startCol };
    }

    // Unicode fractal brackets (optional support)
    if (this.match("⟨")) {
      return { type: "FRACTAL_OPEN", value: "⟨", position: startPos, line: startLine, column: startCol };
    }
    if (this.match("⟩")) {
      return { type: "FRACTAL_CLOSE", value: "⟩", position: startPos, line: startLine, column: startCol };
    }

    // Single-character operators
    const char = this.peek();
    switch (char) {
      case "&":
        this.advance();
        return { type: "AND", value: "&", position: startPos, line: startLine, column: startCol };
      case "|":
        this.advance();
        return { type: "OR", value: "|", position: startPos, line: startLine, column: startCol };
      case "!":
        this.advance();
        return { type: "NOT", value: "!", position: startPos, line: startLine, column: startCol };
      case "^":
        this.advance();
        return { type: "CARET", value: "^", position: startPos, line: startLine, column: startCol };
      case ":":
        this.advance();
        return { type: "COLON", value: ":", position: startPos, line: startLine, column: startCol };
      case "(":
        this.advance();
        return { type: "LPAREN", value: "(", position: startPos, line: startLine, column: startCol };
      case ")":
        this.advance();
        return { type: "RPAREN", value: ")", position: startPos, line: startLine, column: startCol };
      case "=":
        this.advance();
        return { type: "EQUALS", value: "=", position: startPos, line: startLine, column: startCol };
    }

    // Check for FOUNDATION: digit (1-7) followed by letter (a-d)
    if (this.isDigit(char)) {
      const value = this.readNumber();

      // Check if this is a foundation (single digit 1-7 followed by a-d)
      if (value.length === 1 && /[1-7]/.test(value)) {
        const nextChar = this.peek();
        if (nextChar && /[a-d]/.test(nextChar)) {
          this.advance();
          const foundationId = value + nextChar;
          return { type: "FOUNDATION", value: foundationId, position: startPos, line: startLine, column: startCol };
        }
      }

      return { type: "INTEGER", value, position: startPos, line: startLine, column: startCol };
    }

    // Check for IDEA_REF: A-D followed by 1-10
    if (/[A-D]/.test(char)) {
      this.advance();
      if (this.isDigit(this.peek())) {
        const digits = this.readNumber();
        const num = parseInt(digits, 10);
        if (num >= 1 && num <= 10) {
          return { type: "IDEA_REF", value: char + digits, position: startPos, line: startLine, column: startCol };
        }
        // Backtrack - this is not a valid idea ref
        // Treat as identifier starting with the letter
        const rest = this.readIdentifierRest();
        return { type: "IDENT", value: char + digits + rest, position: startPos, line: startLine, column: startCol };
      }
      // Just a letter A-D followed by non-digit, treat as start of identifier
      const rest = this.readIdentifierRest();
      return { type: "IDENT", value: char + rest, position: startPos, line: startLine, column: startCol };
    }

    // Check for identifiers and keywords
    if (this.isLetter(char) || char === "_") {
      const value = this.readIdentifier();

      // Check for keywords
      switch (value) {
        case "let":
          return { type: "LET", value, position: startPos, line: startLine, column: startCol };
        case "eval":
          return { type: "EVAL", value, position: startPos, line: startLine, column: startCol };
        case "show":
          return { type: "SHOW", value, position: startPos, line: startLine, column: startCol };
        case "state":
          return { type: "STATE", value, position: startPos, line: startLine, column: startCol };
        case "help":
          return { type: "HELP", value, position: startPos, line: startLine, column: startCol };
        case "coach":
          return { type: "COACH", value, position: startPos, line: startLine, column: startCol };
        default:
          return { type: "IDENT", value, position: startPos, line: startLine, column: startCol };
      }
    }

    // Unknown character
    throw new LexerError(
      `Unexpected character '${char}'`,
      startLine,
      startCol
    );
  }

  private makeToken(type: TokenType, value: string): Token {
    return {
      type,
      value,
      position: this.position,
      line: this.line,
      column: this.column,
    };
  }

  private peek(offset: number = 0): string {
    const pos = this.position + offset;
    if (pos >= this.input.length) {
      return "\0";
    }
    return this.input[pos];
  }

  private advance(): string {
    const char = this.input[this.position];
    this.position++;
    if (char === "\n") {
      this.line++;
      this.column = 1;
    } else {
      this.column++;
    }
    return char;
  }

  private match(expected: string): boolean {
    if (this.position + expected.length > this.input.length) {
      return false;
    }
    const actual = this.input.slice(this.position, this.position + expected.length);
    if (actual === expected) {
      for (let i = 0; i < expected.length; i++) {
        this.advance();
      }
      return true;
    }
    return false;
  }

  private skipWhitespace(): void {
    while (this.position < this.input.length) {
      const char = this.peek();
      if (char === " " || char === "\t" || char === "\r" || char === "\n") {
        this.advance();
      } else {
        break;
      }
    }
  }

  private isDigit(char: string): boolean {
    return /[0-9]/.test(char);
  }

  private isLetter(char: string): boolean {
    return /[a-zA-Z]/.test(char);
  }

  private isAlphanumeric(char: string): boolean {
    return this.isLetter(char) || this.isDigit(char) || char === "_";
  }

  private readNumber(): string {
    let value = "";
    while (this.isDigit(this.peek())) {
      value += this.advance();
    }
    return value;
  }

  private readIdentifier(): string {
    let value = "";
    while (this.isAlphanumeric(this.peek())) {
      value += this.advance();
    }
    return value;
  }

  private readIdentifierRest(): string {
    let value = "";
    while (this.isAlphanumeric(this.peek())) {
      value += this.advance();
    }
    return value;
  }
}

// ============================================================================
// LEXER ERROR
// ============================================================================

export class LexerError extends Error {
  line: number;
  column: number;

  constructor(message: string, line: number, column: number) {
    super(`Lexer error at line ${line}, column ${column}: ${message}`);
    this.name = "LexerError";
    this.line = line;
    this.column = column;
  }
}
