import { evaluate } from "../src/evaluator";
import { parseExpression } from "../src/parser";
import { createDefaultState, TksState } from "../src/types";

/**
 * Focused evaluator scenarios for fractals, boolean gatekeeping, and implication behavior.
 */
describe("Evaluator fractal + structural semantics", () => {
  let state: TksState;

  beforeEach(() => {
    state = createDefaultState();
  });

  test("fractal nests awareness, vibration, rhythm around a foundation", () => {
    const expr = parseExpression("<<1:4:7>>(3d)");
    const result = evaluate(expr, state);

    expect(result.trace.some((t) => t.includes("awareness applied"))).toBe(true);
    expect(result.trace.some((t) => t.includes("vibration applied"))).toBe(true);
    expect(result.trace.some((t) => t.includes("rhythm applied"))).toBe(true);

    expect(result.value).toMatchObject({
      kind: "rhythm",
      inner: {
        kind: "vibration",
        inner: {
          kind: "aware",
          inner: "3d",
        },
      },
    });
  });

  test("fractal nests awareness, vibration, rhythm around an idea", () => {
    const expr = parseExpression("<<1:4:7>>(B2)");
    const result = evaluate(expr, state);

    expect(result.value).toMatchObject({
      kind: "rhythm",
      inner: {
        kind: "vibration",
        inner: {
          kind: "aware",
          inner: "B2",
        },
      },
    });
  });

  test("boolean AND/OR only coerce when both sides are boolean-like", () => {
    const boolExpr = parseExpression("1 & 0 | 1");
    const boolResult = evaluate(boolExpr, state);
    expect(boolResult.value).toBe(true);

    const structuralAnd = evaluate(parseExpression("1 & A1"), state);
    expect(structuralAnd.value).toMatchObject({
      kind: "and",
      left: 1,
      right: "A1",
    });

    const structuralOr = evaluate(parseExpression("3d | 0"), state);
    expect(structuralOr.value).toMatchObject({
      kind: "or",
      left: "3d",
      right: 0,
    });
  });

  test("implication yields structural value without mutating state", () => {
    const expr = parseExpression("A1 -> B2");
    const snapshot = JSON.parse(JSON.stringify(state));

    const result = evaluate(expr, state);

    expect(result.value).toMatchObject({
      kind: "implication",
      from: "A1",
      to: "B2",
    });
    expect(result.state).toBe(state);
    expect(state).toEqual(snapshot);
  });
});
