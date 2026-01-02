/**
 * TKS Core Engine v0 - REPL Tests
 */

import { TksRepl } from '../src/repl';
import { createDefaultState } from '../src/types';

describe('TksRepl', () => {
  let repl: TksRepl;

  beforeEach(() => {
    repl = new TksRepl();
  });

  describe('let command', () => {
    test('defines a variable', () => {
      repl.processLine('let health = 3d');

      const state = repl.getState();
      expect(state.vars['health']).toBe('3d');
    });

    test('defines variable with complex expression', () => {
      repl.processLine('let combo = A1 & B2');

      const state = repl.getState();
      expect(state.vars['combo']).toMatchObject({
        kind: 'and',
        left: 'A1',
        right: 'B2',
      });
    });

    test('outputs variable assignment', () => {
      const output = repl.processLine('let health = 3d');

      expect(output.some(line => line.includes('health = 3d'))).toBe(true);
    });
  });

  describe('eval command', () => {
    test('evaluates expression', () => {
      const output = repl.processLine('eval 3d');

      expect(output.some(line => line.includes('Result: 3d'))).toBe(true);
    });

    test('evaluates fractal', () => {
      const output = repl.processLine('eval <<1:4:7>>(3d)');

      expect(output.some(line => line.includes('Trace:'))).toBe(true);
      expect(output.some(line => line.includes('ν1'))).toBe(true);
      expect(output.some(line => line.includes('ν4'))).toBe(true);
      expect(output.some(line => line.includes('ν7'))).toBe(true);
    });

    test('evaluates using defined variable', () => {
      repl.processLine('let health = 3d');
      const output = repl.processLine('eval <<1>>(health)');

      expect(output.some(line => line.includes('AWARE(3d)'))).toBe(true);
    });
  });

  describe('bare expression', () => {
    test('evaluates bare expression same as eval', () => {
      const output = repl.processLine('3d');

      expect(output.some(line => line.includes('Result: 3d'))).toBe(true);
    });

    test('evaluates boolean expression', () => {
      const output = repl.processLine('1 & 0');

      expect(output.some(line => line.includes('Result: false'))).toBe(true);
    });
  });

  describe('show state command', () => {
    test('shows default state', () => {
      const output = repl.processLine('show state');

      expect(output.some(line => line.includes('TKS State'))).toBe(true);
      expect(output.some(line => line.includes('Current Noetic: ν0'))).toBe(true);
      expect(output.some(line => line.includes('Variables: (none)'))).toBe(true);
    });

    test('shows defined variables', () => {
      repl.processLine('let x = 3d');
      repl.processLine('let y = B2');
      const output = repl.processLine('show state');

      expect(output.some(line => line.includes('x = 3d'))).toBe(true);
      expect(output.some(line => line.includes('y = B2'))).toBe(true);
    });
  });

  describe('help command', () => {
    test('shows help', () => {
      const output = repl.processLine('help');

      expect(output.some(line => line.includes('Commands'))).toBe(true);
      expect(output.some(line => line.includes('let'))).toBe(true);
      expect(output.some(line => line.includes('eval'))).toBe(true);
      expect(output.some(line => line.includes('show state'))).toBe(true);
      expect(output.some(line => line.includes('Fractals'))).toBe(true);
    });
  });

  describe('error handling', () => {
    test('handles parse error gracefully', () => {
      const output = repl.processLine('3d &');

      expect(output.some(line => line.includes('Parse error'))).toBe(true);
    });

    test('handles undefined variable gracefully', () => {
      const output = repl.processLine('undefined_var');

      expect(output.some(line => line.includes('Unknown identifier'))).toBe(true);
    });

    test('handles invalid noetic index gracefully', () => {
      const output = repl.processLine('3d^15');

      expect(output.some(line => line.includes('error'))).toBe(true);
    });
  });

  describe('integration tests', () => {
    test('full workflow: define, use, evaluate', () => {
      // Define a foundation
      repl.processLine('let health = 3d');

      // Define an idea
      repl.processLine('let energy = B2');

      // Combine them
      repl.processLine('let combo = health & energy');

      // Apply fractal
      const output = repl.processLine('eval <<1:4:7>>(combo)');

      // Verify state
      const state = repl.getState();
      expect(state.vars['health']).toBe('3d');
      expect(state.vars['energy']).toBe('B2');
      expect(state.vars['combo']).toMatchObject({ kind: 'and' });

      // Verify output includes fractal trace
      expect(output.some(line => line.includes('ν1'))).toBe(true);
    });
  });
});
