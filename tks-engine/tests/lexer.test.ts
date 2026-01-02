/**
 * TKS Core Engine v0 - Lexer Tests
 */

import { Lexer, LexerError } from '../src/lexer';

describe('Lexer', () => {
  describe('Foundation tokens', () => {
    test('tokenizes valid foundations (1a-7d)', () => {
      const lexer = new Lexer('1a 3d 7c');
      const tokens = lexer.tokenize();

      expect(tokens[0]).toMatchObject({ type: 'FOUNDATION', value: '1a' });
      expect(tokens[1]).toMatchObject({ type: 'FOUNDATION', value: '3d' });
      expect(tokens[2]).toMatchObject({ type: 'FOUNDATION', value: '7c' });
    });

    test('does not tokenize 8a as foundation', () => {
      const lexer = new Lexer('8a');
      const tokens = lexer.tokenize();

      // 8 is an integer, 'a' is an identifier
      expect(tokens[0]).toMatchObject({ type: 'INTEGER', value: '8' });
      expect(tokens[1]).toMatchObject({ type: 'IDENT', value: 'a' });
    });

    test('does not tokenize 1e as foundation (only a-d valid)', () => {
      const lexer = new Lexer('1e');
      const tokens = lexer.tokenize();

      // 1 is an integer, 'e' is an identifier
      expect(tokens[0]).toMatchObject({ type: 'INTEGER', value: '1' });
      expect(tokens[1]).toMatchObject({ type: 'IDENT', value: 'e' });
    });
  });

  describe('Idea tokens', () => {
    test('tokenizes valid ideas (A1-D10)', () => {
      const lexer = new Lexer('A1 B5 C10 D9');
      const tokens = lexer.tokenize();

      expect(tokens[0]).toMatchObject({ type: 'IDEA_REF', value: 'A1' });
      expect(tokens[1]).toMatchObject({ type: 'IDEA_REF', value: 'B5' });
      expect(tokens[2]).toMatchObject({ type: 'IDEA_REF', value: 'C10' });
      expect(tokens[3]).toMatchObject({ type: 'IDEA_REF', value: 'D9' });
    });

    test('does not tokenize E1 as idea (only A-D valid)', () => {
      const lexer = new Lexer('E1');
      const tokens = lexer.tokenize();

      expect(tokens[0]).toMatchObject({ type: 'IDENT', value: 'E1' });
    });
  });

  describe('Operators', () => {
    test('tokenizes all operators', () => {
      const lexer = new Lexer('& | ! -> ^ : = ( )');
      const tokens = lexer.tokenize();

      expect(tokens[0]).toMatchObject({ type: 'AND', value: '&' });
      expect(tokens[1]).toMatchObject({ type: 'OR', value: '|' });
      expect(tokens[2]).toMatchObject({ type: 'NOT', value: '!' });
      expect(tokens[3]).toMatchObject({ type: 'IMPLIES', value: '->' });
      expect(tokens[4]).toMatchObject({ type: 'CARET', value: '^' });
      expect(tokens[5]).toMatchObject({ type: 'COLON', value: ':' });
      expect(tokens[6]).toMatchObject({ type: 'EQUALS', value: '=' });
      expect(tokens[7]).toMatchObject({ type: 'LPAREN', value: '(' });
      expect(tokens[8]).toMatchObject({ type: 'RPAREN', value: ')' });
    });

    test('tokenizes fractal brackets', () => {
      const lexer = new Lexer('<< >>');
      const tokens = lexer.tokenize();

      expect(tokens[0]).toMatchObject({ type: 'FRACTAL_OPEN', value: '<<' });
      expect(tokens[1]).toMatchObject({ type: 'FRACTAL_CLOSE', value: '>>' });
    });

    test('tokenizes Unicode fractal brackets', () => {
      const lexer = new Lexer('⟨ ⟩');
      const tokens = lexer.tokenize();

      expect(tokens[0]).toMatchObject({ type: 'FRACTAL_OPEN', value: '⟨' });
      expect(tokens[1]).toMatchObject({ type: 'FRACTAL_CLOSE', value: '⟩' });
    });
  });

  describe('Keywords', () => {
    test('tokenizes REPL keywords', () => {
      const lexer = new Lexer('let eval show state help');
      const tokens = lexer.tokenize();

      expect(tokens[0]).toMatchObject({ type: 'LET', value: 'let' });
      expect(tokens[1]).toMatchObject({ type: 'EVAL', value: 'eval' });
      expect(tokens[2]).toMatchObject({ type: 'SHOW', value: 'show' });
      expect(tokens[3]).toMatchObject({ type: 'STATE', value: 'state' });
      expect(tokens[4]).toMatchObject({ type: 'HELP', value: 'help' });
    });
  });

  describe('Identifiers and integers', () => {
    test('tokenizes identifiers', () => {
      const lexer = new Lexer('health_goal myVar _private');
      const tokens = lexer.tokenize();

      expect(tokens[0]).toMatchObject({ type: 'IDENT', value: 'health_goal' });
      expect(tokens[1]).toMatchObject({ type: 'IDENT', value: 'myVar' });
      expect(tokens[2]).toMatchObject({ type: 'IDENT', value: '_private' });
    });

    test('tokenizes integers', () => {
      const lexer = new Lexer('0 1 42 100');
      const tokens = lexer.tokenize();

      expect(tokens[0]).toMatchObject({ type: 'INTEGER', value: '0' });
      expect(tokens[1]).toMatchObject({ type: 'INTEGER', value: '1' });
      expect(tokens[2]).toMatchObject({ type: 'INTEGER', value: '42' });
      expect(tokens[3]).toMatchObject({ type: 'INTEGER', value: '100' });
    });
  });

  describe('Complex expressions', () => {
    test('tokenizes fractal application', () => {
      const lexer = new Lexer('<<1:4:7>>(3d)');
      const tokens = lexer.tokenize();

      expect(tokens[0]).toMatchObject({ type: 'FRACTAL_OPEN' });
      expect(tokens[1]).toMatchObject({ type: 'INTEGER', value: '1' });
      expect(tokens[2]).toMatchObject({ type: 'COLON' });
      expect(tokens[3]).toMatchObject({ type: 'INTEGER', value: '4' });
      expect(tokens[4]).toMatchObject({ type: 'COLON' });
      expect(tokens[5]).toMatchObject({ type: 'INTEGER', value: '7' });
      expect(tokens[6]).toMatchObject({ type: 'FRACTAL_CLOSE' });
      expect(tokens[7]).toMatchObject({ type: 'LPAREN' });
      expect(tokens[8]).toMatchObject({ type: 'FOUNDATION', value: '3d' });
      expect(tokens[9]).toMatchObject({ type: 'RPAREN' });
    });

    test('tokenizes noetic term', () => {
      const lexer = new Lexer('3d^2');
      const tokens = lexer.tokenize();

      expect(tokens[0]).toMatchObject({ type: 'FOUNDATION', value: '3d' });
      expect(tokens[1]).toMatchObject({ type: 'CARET' });
      expect(tokens[2]).toMatchObject({ type: 'INTEGER', value: '2' });
    });
  });

  describe('Error handling', () => {
    test('throws on unexpected character', () => {
      const lexer = new Lexer('3d @ x');
      expect(() => lexer.tokenize()).toThrow(LexerError);
    });
  });
});
