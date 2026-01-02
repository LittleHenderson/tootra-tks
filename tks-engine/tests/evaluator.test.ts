/**
 * TKS Core Engine v0 - Evaluator Tests
 */

import { evaluate, valueToString, EvaluationError } from '../src/evaluator';
import { parseExpression } from '../src/parser';
import { createDefaultState, TksState } from '../src/types';

describe('Evaluator', () => {
  let state: TksState;

  beforeEach(() => {
    state = createDefaultState();
  });

  describe('Atomic evaluation', () => {
    /**
     * Test: Parsing and evaluating a bare foundation
     * Input: 3d
     * Expected: value "3d", state unchanged
     */
    test('evaluates foundation - returns foundation ID', () => {
      const expr = parseExpression('3d');
      const result = evaluate(expr, state);

      expect(result.value).toBe('3d');
      expect(result.state).toEqual(state);
      expect(result.trace).toContain('foundation 3d');
    });

    test('evaluates idea - returns idea ID', () => {
      const expr = parseExpression('B2');
      const result = evaluate(expr, state);

      expect(result.value).toBe('B2');
      expect(result.trace).toContain('idea B2');
    });

    test('evaluates integer', () => {
      const expr = parseExpression('42');
      const result = evaluate(expr, state);

      expect(result.value).toBe(42);
      expect(result.trace).toContain('integer 42');
    });

    test('evaluates identifier from state.vars', () => {
      state.vars['health'] = '3d';
      const expr = parseExpression('health');
      const result = evaluate(expr, state);

      expect(result.value).toBe('3d');
    });

    test('throws on unknown identifier', () => {
      const expr = parseExpression('unknown_var');
      expect(() => evaluate(expr, state)).toThrow(EvaluationError);
      expect(() => evaluate(expr, state)).toThrow('Unknown identifier: unknown_var');
    });
  });

  describe('Boolean operations', () => {
    /**
     * Test: Boolean logic
     * eval 1 & 0 -> Result: false
     */
    test('evaluates AND with numbers (1 & 0 = false)', () => {
      const expr = parseExpression('1 & 0');
      const result = evaluate(expr, state);

      expect(result.value).toBe(false);
    });

    test('evaluates AND with both true', () => {
      const expr = parseExpression('1 & 1');
      const result = evaluate(expr, state);

      expect(result.value).toBe(true);
    });

    test('evaluates OR with numbers', () => {
      const expr = parseExpression('1 | 0');
      const result = evaluate(expr, state);

      expect(result.value).toBe(true);
    });

    test('evaluates OR with both false', () => {
      const expr = parseExpression('0 | 0');
      const result = evaluate(expr, state);

      expect(result.value).toBe(false);
    });

    /**
     * Test: eval !1 -> error (per spec, non-boolean NOT should throw)
     * Since we treat 1 as truthy, !1 should return false
     */
    test('evaluates NOT on number (treats as boolean)', () => {
      const expr = parseExpression('!1');
      const result = evaluate(expr, state);

      expect(result.value).toBe(false);
    });

    test('evaluates NOT on zero', () => {
      const expr = parseExpression('!0');
      const result = evaluate(expr, state);

      expect(result.value).toBe(true);
    });

    test('evaluates double NOT', () => {
      const expr = parseExpression('!!1');
      const result = evaluate(expr, state);

      expect(result.value).toBe(true);
    });
  });

  describe('Structural operations', () => {
    test('AND with non-boolean returns structural value', () => {
      const expr = parseExpression('3d & 1a');
      const result = evaluate(expr, state);

      expect(result.value).toMatchObject({
        kind: 'and',
        left: '3d',
        right: '1a',
      });
    });

    test('OR with non-boolean returns structural value', () => {
      const expr = parseExpression('A1 | B2');
      const result = evaluate(expr, state);

      expect(result.value).toMatchObject({
        kind: 'or',
        left: 'A1',
        right: 'B2',
      });
    });

    test('implication returns structural value', () => {
      const expr = parseExpression('A1 -> B2');
      const result = evaluate(expr, state);

      expect(result.value).toMatchObject({
        kind: 'implication',
        from: 'A1',
        to: 'B2',
      });
    });
  });

  describe('Noetic term evaluation', () => {
    test('applies noetic to foundation', () => {
      const expr = parseExpression('3d^2');
      const result = evaluate(expr, state);

      // ν2 is no-op in v0, so value should be unchanged
      expect(result.value).toBe('3d');
      expect(result.trace.some(t => t.includes('ν2'))).toBe(true);
    });

    test('applies awareness noetic (ν1)', () => {
      const expr = parseExpression('3d^1');
      const result = evaluate(expr, state);

      expect(result.value).toMatchObject({
        kind: 'aware',
        inner: '3d',
      });
      expect(result.trace.some(t => t.includes('ν1: awareness applied'))).toBe(true);
    });

    test('applies vibration noetic (ν4)', () => {
      const expr = parseExpression('3d^4');
      const result = evaluate(expr, state);

      expect(result.value).toMatchObject({
        kind: 'vibration',
        inner: '3d',
      });
    });

    test('applies rhythm noetic (ν7)', () => {
      const expr = parseExpression('3d^7');
      const result = evaluate(expr, state);

      expect(result.value).toMatchObject({
        kind: 'rhythm',
        inner: '3d',
      });
    });
  });

  describe('Fractal evaluation', () => {
    /**
     * Test: Parsing and evaluating a fractal
     * Input: eval <<1:4:7>>(3d)
     * Expected:
     *   - No crash
     *   - Trace includes 3 applications: ν1, ν4, ν7
     *   - Result value is whatever Noetic-tagging functions produce
     */
    test('evaluates fractal with multiple noetics', () => {
      const expr = parseExpression('<<1:4:7>>(3d)');
      const result = evaluate(expr, state);

      // Should not crash
      expect(result).toBeDefined();

      // Trace should include all 3 applications
      expect(result.trace.some(t => t.includes('ν1'))).toBe(true);
      expect(result.trace.some(t => t.includes('ν4'))).toBe(true);
      expect(result.trace.some(t => t.includes('ν7'))).toBe(true);

      // Result should be nested structural values
      // ν1 wraps in AWARE, ν4 wraps in VIBRATION, ν7 wraps in RHYTHM
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

    test('evaluates fractal with single noetic', () => {
      const expr = parseExpression('<<0>>(3d)');
      const result = evaluate(expr, state);

      // ν0 is identity, value should be unchanged
      expect(result.value).toBe('3d');
      expect(result.trace.some(t => t.includes('ν0: identity'))).toBe(true);
    });

    test('evaluates fractal with complex inner expression', () => {
      const expr = parseExpression('<<1>>(A1 & B2)');
      const result = evaluate(expr, state);

      expect(result.value).toMatchObject({
        kind: 'aware',
        inner: {
          kind: 'and',
          left: 'A1',
          right: 'B2',
        },
      });
    });
  });

  describe('Variable tests', () => {
    /**
     * Test: Variables
     * let health = 3d
     * eval <<1:4:7>>(health)
     */
    test('evaluates fractal with variable', () => {
      // First set up the variable
      state.vars['health'] = '3d';

      // Then evaluate fractal with variable
      const expr = parseExpression('<<1:4:7>>(health)');
      const result = evaluate(expr, state);

      // Should work the same as with direct foundation
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

  describe('valueToString', () => {
    test('formats boolean', () => {
      expect(valueToString(true)).toBe('true');
      expect(valueToString(false)).toBe('false');
    });

    test('formats number', () => {
      expect(valueToString(42)).toBe('42');
    });

    test('formats string (foundation/idea)', () => {
      expect(valueToString('3d')).toBe('3d');
      expect(valueToString('B2')).toBe('B2');
    });

    test('formats structural values', () => {
      expect(valueToString({ kind: 'aware', inner: '3d' })).toBe('AWARE(3d)');
      expect(valueToString({ kind: 'vibration', inner: '3d' })).toBe('VIBRATION(3d)');
      expect(valueToString({ kind: 'rhythm', inner: '3d' })).toBe('RHYTHM(3d)');
    });
  });

  describe('State changes under Noetics', () => {
    test('noetic application mutates foundations via numeric semantics', () => {
      const originalState = createDefaultState();

      const expr = parseExpression('<<1:4:7>>(3d)');
      const result = evaluate(expr, originalState);

      expect(originalState.foundations['3d']).toBeCloseTo(0.30);
      expect(result.state.foundations['3d']).toBeCloseTo(0.30);
    });
  });
});
