/**
 * TKS Core Engine v0 - Parser Tests
 */

import { parseExpression, parseCommand, ParseError } from '../src/parser';
import { TksExpression } from '../src/ast';

describe('Parser', () => {
  describe('Atomic expressions', () => {
    test('parses foundation', () => {
      const expr = parseExpression('3d');
      expect(expr).toMatchObject({
        kind: 'foundation',
        id: '3d',
      });
    });

    test('parses all foundation variations', () => {
      expect(parseExpression('1a')).toMatchObject({ kind: 'foundation', id: '1a' });
      expect(parseExpression('7d')).toMatchObject({ kind: 'foundation', id: '7d' });
      expect(parseExpression('4b')).toMatchObject({ kind: 'foundation', id: '4b' });
    });

    test('parses idea reference', () => {
      const expr = parseExpression('B2');
      expect(expr).toMatchObject({
        kind: 'idea',
        id: 'B2',
      });
    });

    test('parses ideas with two digits', () => {
      expect(parseExpression('A10')).toMatchObject({ kind: 'idea', id: 'A10' });
      expect(parseExpression('D10')).toMatchObject({ kind: 'idea', id: 'D10' });
    });

    test('parses integer', () => {
      const expr = parseExpression('42');
      expect(expr).toMatchObject({
        kind: 'integer',
        value: 42,
      });
    });

    test('parses identifier', () => {
      const expr = parseExpression('health_goal');
      expect(expr).toMatchObject({
        kind: 'ident',
        name: 'health_goal',
      });
    });
  });

  describe('Unary expressions', () => {
    test('parses NOT', () => {
      const expr = parseExpression('!1');
      expect(expr).toMatchObject({
        kind: 'unary',
        op: '!',
        expr: { kind: 'integer', value: 1 },
      });
    });

    test('parses double NOT', () => {
      const expr = parseExpression('!!1');
      expect(expr).toMatchObject({
        kind: 'unary',
        op: '!',
        expr: {
          kind: 'unary',
          op: '!',
          expr: { kind: 'integer', value: 1 },
        },
      });
    });
  });

  describe('Binary expressions', () => {
    test('parses AND', () => {
      const expr = parseExpression('1 & 0');
      expect(expr).toMatchObject({
        kind: 'binary',
        op: '&',
        left: { kind: 'integer', value: 1 },
        right: { kind: 'integer', value: 0 },
      });
    });

    test('parses OR', () => {
      const expr = parseExpression('1 | 0');
      expect(expr).toMatchObject({
        kind: 'binary',
        op: '|',
        left: { kind: 'integer', value: 1 },
        right: { kind: 'integer', value: 0 },
      });
    });

    test('respects operator precedence (& before |)', () => {
      // a | b & c should be a | (b & c)
      const expr = parseExpression('1 | 2 & 3');
      expect(expr).toMatchObject({
        kind: 'binary',
        op: '|',
        left: { kind: 'integer', value: 1 },
        right: {
          kind: 'binary',
          op: '&',
          left: { kind: 'integer', value: 2 },
          right: { kind: 'integer', value: 3 },
        },
      });
    });

    test('respects operator precedence (! before &)', () => {
      // !a & b should be (!a) & b
      const expr = parseExpression('!1 & 2');
      expect(expr).toMatchObject({
        kind: 'binary',
        op: '&',
        left: {
          kind: 'unary',
          op: '!',
          expr: { kind: 'integer', value: 1 },
        },
        right: { kind: 'integer', value: 2 },
      });
    });
  });

  describe('Implication', () => {
    test('parses implication', () => {
      const expr = parseExpression('A1 -> B2');
      expect(expr).toMatchObject({
        kind: 'implication',
        left: { kind: 'idea', id: 'A1' },
        right: { kind: 'idea', id: 'B2' },
      });
    });

    test('implication is right-associative', () => {
      // a -> b -> c should be a -> (b -> c)
      const expr = parseExpression('A1 -> B2 -> C3');
      expect(expr).toMatchObject({
        kind: 'implication',
        left: { kind: 'idea', id: 'A1' },
        right: {
          kind: 'implication',
          left: { kind: 'idea', id: 'B2' },
          right: { kind: 'idea', id: 'C3' },
        },
      });
    });

    test('implication has lowest precedence', () => {
      // a & b -> c | d should be (a & b) -> (c | d)
      const expr = parseExpression('1 & 2 -> 3 | 4');
      expect(expr).toMatchObject({
        kind: 'implication',
        left: {
          kind: 'binary',
          op: '&',
        },
        right: {
          kind: 'binary',
          op: '|',
        },
      });
    });
  });

  describe('Parentheses', () => {
    test('parentheses override precedence', () => {
      // (a | b) & c - parentheses force | to bind first
      const expr = parseExpression('(1 | 2) & 3');
      expect(expr).toMatchObject({
        kind: 'binary',
        op: '&',
        left: {
          kind: 'binary',
          op: '|',
          left: { kind: 'integer', value: 1 },
          right: { kind: 'integer', value: 2 },
        },
        right: { kind: 'integer', value: 3 },
      });
    });
  });

  describe('Noetic terms', () => {
    test('parses noetic term with foundation', () => {
      const expr = parseExpression('3d^2');
      expect(expr).toMatchObject({
        kind: 'noeticTerm',
        base: { kind: 'foundation', id: '3d' },
        noetic: 2,
      });
    });

    test('parses noetic term with idea', () => {
      const expr = parseExpression('B2^4');
      expect(expr).toMatchObject({
        kind: 'noeticTerm',
        base: { kind: 'idea', id: 'B2' },
        noetic: 4,
      });
    });

    test('parses noetic term with identifier', () => {
      const expr = parseExpression('health^7');
      expect(expr).toMatchObject({
        kind: 'noeticTerm',
        base: { kind: 'ident', name: 'health' },
        noetic: 7,
      });
    });

    test('rejects invalid noetic index', () => {
      expect(() => parseExpression('3d^10')).toThrow(ParseError);
    });
  });

  describe('Fractal applications', () => {
    test('parses fractal with single noetic', () => {
      const expr = parseExpression('<<1>>(3d)');
      expect(expr).toMatchObject({
        kind: 'fractalApplication',
        noetics: [1],
        inner: { kind: 'foundation', id: '3d' },
      });
    });

    test('parses fractal with multiple noetics', () => {
      const expr = parseExpression('<<1:4:7>>(3d)');
      expect(expr).toMatchObject({
        kind: 'fractalApplication',
        noetics: [1, 4, 7],
        inner: { kind: 'foundation', id: '3d' },
      });
    });

    test('parses nested fractal expression', () => {
      const expr = parseExpression('<<1:4:7>>(A1 & B2)');
      expect(expr).toMatchObject({
        kind: 'fractalApplication',
        noetics: [1, 4, 7],
        inner: {
          kind: 'binary',
          op: '&',
          left: { kind: 'idea', id: 'A1' },
          right: { kind: 'idea', id: 'B2' },
        },
      });
    });

    test('parses Unicode fractal brackets', () => {
      const expr = parseExpression('⟨1:4:7⟩(3d)');
      expect(expr).toMatchObject({
        kind: 'fractalApplication',
        noetics: [1, 4, 7],
        inner: { kind: 'foundation', id: '3d' },
      });
    });
  });

  describe('REPL commands', () => {
    test('parses let command', () => {
      const cmd = parseCommand('let health = 3d');
      expect(cmd).toMatchObject({
        kind: 'let',
        name: 'health',
        expr: { kind: 'foundation', id: '3d' },
      });
    });

    test('parses eval command', () => {
      const cmd = parseCommand('eval 3d');
      expect(cmd).toMatchObject({
        kind: 'eval',
        expr: { kind: 'foundation', id: '3d' },
      });
    });

    test('parses show state command', () => {
      const cmd = parseCommand('show state');
      expect(cmd).toMatchObject({
        kind: 'showState',
      });
    });

    test('parses help command', () => {
      const cmd = parseCommand('help');
      expect(cmd).toMatchObject({
        kind: 'help',
      });
    });

    test('parses bare expression as expr command', () => {
      const cmd = parseCommand('3d & 1a');
      expect(cmd).toMatchObject({
        kind: 'expr',
        expr: {
          kind: 'binary',
          op: '&',
        },
      });
    });
  });

  describe('Error handling', () => {
    test('throws on invalid syntax', () => {
      expect(() => parseExpression('3d &')).toThrow(ParseError);
    });

    test('throws on unclosed parenthesis', () => {
      expect(() => parseExpression('(3d')).toThrow(ParseError);
    });

    test('throws on unclosed fractal', () => {
      expect(() => parseExpression('<<1:4:7(3d)')).toThrow(ParseError);
    });
  });
});
