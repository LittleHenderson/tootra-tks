import { checkPrereqs } from "../src/rpm";
import { createDefaultState, TksState } from "../src/types";

describe("RPM prerequisite checks", () => {
  let state: TksState;

  beforeEach(() => {
    state = createDefaultState();
  });

  test("5d requires 13,14,15 and reports missing", () => {
    const result = checkPrereqs("5d", state);
    expect(result.required).toEqual([13, 14, 15]);
    expect(result.present).toEqual([]);
    expect(result.missing).toEqual([13, 14, 15]);
  });

  test("5d with partial acquisitions reports present/missing correctly", () => {
    state.acquisitions[13] = true;
    state.acquisitions[14] = true;
    const result = checkPrereqs("5d", state);
    expect(result.present).toEqual([13, 14]);
    expect(result.missing).toEqual([15]);
  });

  test("6d requires 16,17,18 and reports missing", () => {
    const result = checkPrereqs("6d", state);
    expect(result.required).toEqual([16, 17, 18]);
    expect(result.present).toEqual([]);
    expect(result.missing).toEqual([16, 17, 18]);
  });

  test("6d with acquisitions present passes", () => {
    state.acquisitions[16] = true;
    state.acquisitions[17] = true;
    state.acquisitions[18] = true;
    const result = checkPrereqs("6d", state);
    expect(result.present).toEqual([16, 17, 18]);
    expect(result.missing).toEqual([]);
    expect(result.notes.some((n) => n.includes("satisfied"))).toBe(true);
  });

  test("3d has no requirements", () => {
    const result = checkPrereqs("3d", state);
    expect(result.required).toEqual([]);
    expect(result.missing).toEqual([]);
    expect(result.notes.some((n) => n.includes("No RPM prerequisites defined"))).toBe(true);
  });
});
