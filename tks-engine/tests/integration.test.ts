/**
 * TKS Core Engine v0 - Integration Tests
 *
 * These tests verify the complete pipeline from parsing to evaluation,
 * matching the examples specified in the v0 specification.
 */

import { parseExpression, parseCommand } from '../src/parser';
import { evaluate, valueToString } from '../src/evaluator';
import { createDefaultState, TksState } from '../src/types';
import { TksRepl } from '../src/repl';

describe('Specification Examples', () => {
  let state: TksState;

  beforeEach(() => {
    state = createDefaultState();
  });

  /**
   * G.1: Parsing and evaluating a bare foundation
   * Input: 3d
   * Expected: value "3d", state unchanged
   */
  describe('G.1: Bare foundation', () => {
    test('evaluates 3d to string "3d" with state unchanged', () => {
      const expr = parseExpression('3d');
      const originalState = JSON.parse(JSON.stringify(state));

      const result = evaluate(expr, state);

      expect(result.value).toBe('3d');
      expect(result.state).toEqual(originalState);
    });
  });

  /**
   * G.2: Parsing and evaluating a fractal
   * Input: eval <<1:4:7>>(3d)
   * Expected:
   *   - No crash
   *   - Trace includes 3 applications: ν1, ν4, ν7
   *   - Result value is whatever Noetic-tagging functions produce
   */
  describe('G.2: Fractal evaluation', () => {
    test('evaluates <<1:4:7>>(3d) without crash', () => {
      const expr = parseExpression('<<1:4:7>>(3d)');

      expect(() => evaluate(expr, state)).not.toThrow();
    });

    test('trace includes ν1, ν4, ν7 applications', () => {
      const expr = parseExpression('<<1:4:7>>(3d)');
      const result = evaluate(expr, state);

      const traceText = result.trace.join('\n');
      expect(traceText).toContain('ν1');
      expect(traceText).toContain('ν4');
      expect(traceText).toContain('ν7');
    });

    test('result is nested structural value from noetic applications', () => {
      const expr = parseExpression('<<1:4:7>>(3d)');
      const result = evaluate(expr, state);

      // ν1 -> AWARE, ν4 -> VIBRATION, ν7 -> RHYTHM (applied in sequence)
      expect(result.value).toMatchObject({
        kind: 'rhythm',
        inner: {
          kind: 'vibration',
          inner: {
            kind: 'aware',
            inner: '3d',
          },
        },
      });
    });
  });

  /**
   * G.3: Variables
   * let health = 3d
   * eval <<1:4:7>>(health)
   */
  describe('G.3: Variables', () => {
    test('can define and use variables with fractals', () => {
      // Set up variable
      state.vars['health'] = '3d';

      // Evaluate fractal with variable
      const expr = parseExpression('<<1:4:7>>(health)');
      const result = evaluate(expr, state);

      // Should produce same result as direct 3d
      expect(result.value).toMatchObject({
        kind: 'rhythm',
        inner: {
          kind: 'vibration',
          inner: {
            kind: 'aware',
            inner: '3d',
          },
        },
      });
    });

    test('REPL workflow: let health = 3d then eval <<1:4:7>>(health)', () => {
      const repl = new TksRepl();

      // Define variable
      repl.processLine('let health = 3d');

      // Evaluate fractal
      const output = repl.processLine('eval <<1:4:7>>(health)');

      // Verify trace is shown
      expect(output.some(line => line.includes('ν1'))).toBe(true);
      expect(output.some(line => line.includes('ν4'))).toBe(true);
      expect(output.some(line => line.includes('ν7'))).toBe(true);
    });
  });

  /**
   * G.4: Boolean logic
   * eval 1 & 0 -> Result: false
   * eval !1 -> error or false (choose behavior and keep consistent)
   */
  describe('G.4: Boolean logic', () => {
    test('1 & 0 evaluates to false', () => {
      const expr = parseExpression('1 & 0');
      const result = evaluate(expr, state);

      expect(result.value).toBe(false);
    });

    test('!1 evaluates to false (number treated as boolean)', () => {
      // We chose: numbers are converted to boolean (0 = false, non-zero = true)
      const expr = parseExpression('!1');
      const result = evaluate(expr, state);

      expect(result.value).toBe(false);
    });

    test('!0 evaluates to true', () => {
      const expr = parseExpression('!0');
      const result = evaluate(expr, state);

      expect(result.value).toBe(true);
    });

    test('1 | 0 evaluates to true', () => {
      const expr = parseExpression('1 | 0');
      const result = evaluate(expr, state);

      expect(result.value).toBe(true);
    });

    test('0 & 0 evaluates to false', () => {
      const expr = parseExpression('0 & 0');
      const result = evaluate(expr, state);

      expect(result.value).toBe(false);
    });

    test('complex boolean: (1 | 0) & !0', () => {
      const expr = parseExpression('(1 | 0) & !0');
      const result = evaluate(expr, state);

      // (true) & (true) = true
      expect(result.value).toBe(true);
    });
  });

  describe('All Foundation IDs', () => {
    const foundations = [
      '1a', '2a', '3a', '4a', '5a', '6a', '7a',
      '1b', '2b', '3b', '4b', '5b', '6b', '7b',
      '1c', '2c', '3c', '4c', '5c', '6c', '7c',
      '1d', '2d', '3d', '4d', '5d', '6d', '7d',
    ];

    test.each(foundations)('parses and evaluates foundation %s', (id) => {
      const expr = parseExpression(id);
      const result = evaluate(expr, state);

      expect(expr).toMatchObject({ kind: 'foundation', id });
      expect(result.value).toBe(id);
    });
  });

  describe('All Idea IDs', () => {
    const ideas = [
      'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10',
      'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10',
      'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10',
      'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10',
    ];

    test.each(ideas)('parses and evaluates idea %s', (id) => {
      const expr = parseExpression(id);
      const result = evaluate(expr, state);

      expect(expr).toMatchObject({ kind: 'idea', id });
      expect(result.value).toBe(id);
    });
  });

  describe('All Noetic Indices', () => {
    test.each([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])('applies noetic index %i', (idx) => {
      const expr = parseExpression(`3d^${idx}`);
      const result = evaluate(expr, state);

      expect(result.trace.some(line => line.includes(`ν${idx}`))).toBe(true);
    });
  });

  describe('Operator Precedence', () => {
    test('! binds tighter than &', () => {
      // !1 & 1 should be (!1) & 1 = false & true = false
      const expr = parseExpression('!1 & 1');
      const result = evaluate(expr, state);

      expect(result.value).toBe(false);
    });

    test('& binds tighter than |', () => {
      // 1 | 0 & 0 should be 1 | (0 & 0) = true | false = true
      const expr = parseExpression('1 | 0 & 0');
      const result = evaluate(expr, state);

      expect(result.value).toBe(true);
    });

    test('| binds tighter than ->', () => {
      // Parse: 1 | 0 -> 0 | 1 should be (1 | 0) -> (0 | 1)
      const expr = parseExpression('1 | 0 -> 0 | 1');

      expect(expr).toMatchObject({
        kind: 'implication',
        left: { kind: 'binary', op: '|' },
        right: { kind: 'binary', op: '|' },
      });
    });

    test('parentheses override precedence', () => {
      // (0 | 1) & 0 should be false
      const expr = parseExpression('(0 | 1) & 0');
      const result = evaluate(expr, state);

      expect(result.value).toBe(false);
    });
  });

  describe('Complex Expressions', () => {
    test('nested fractals in implication', () => {
      const expr = parseExpression('<<1>>(3d) -> <<4>>(1a)');
      const result = evaluate(expr, state);

      expect(result.value).toMatchObject({
        kind: 'implication',
        from: { kind: 'aware', inner: '3d' },
        to: { kind: 'vibration', inner: '1a' },
      });
    });

    test('noetic term in binary expression', () => {
      const expr = parseExpression('3d^1 & 1a^4');
      const result = evaluate(expr, state);

      expect(result.value).toMatchObject({
        kind: 'and',
        left: { kind: 'aware', inner: '3d' },
        right: { kind: 'vibration', inner: '1a' },
      });
    });

    test('fractal with complex inner expression', () => {
      const expr = parseExpression('<<1:4:7>>(A1 & B2 | C3)');
      const result = evaluate(expr, state);

      // Should not crash and should apply all noetics
      expect(result.trace.filter(t => t.includes('Applied ν')).length).toBe(3);
    });
  });
});
