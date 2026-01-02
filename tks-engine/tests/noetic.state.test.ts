import { evaluate } from "../src/evaluator";
import { parseExpression } from "../src/parser";
import { createDefaultState, TksState } from "../src/types";

/**
 * Numeric semantics for Noetics applied to foundations.
 */
describe("Noetic numeric semantics", () => {
  let state: TksState;

  beforeEach(() => {
    state = createDefaultState();
  });

  test("ν1 awareness increases foundation slightly", () => {
    const expr = parseExpression("3d^1");
    evaluate(expr, state);
    expect(state.foundations["3d"]).toBeCloseTo(0.05);
  });

  test("ν1 + ν4 + ν7 stack deltas via fractal", () => {
    const expr = parseExpression("<<1:4:7>>(3d)");
    evaluate(expr, state);
    expect(state.foundations["3d"]).toBeCloseTo(0.30);
  });

  test("deltas clamp at 1.0", () => {
    state.foundations["3d"] = 0.95;
    const expr = parseExpression("<<4:7>>(3d)");
    evaluate(expr, state);
    expect(state.foundations["3d"]).toBe(1.0);
  });
});
