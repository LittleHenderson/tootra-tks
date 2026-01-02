import { TksRepl } from "../src/repl";
import { createDefaultState } from "../src/types";

/**
 * REPL state manipulation commands.
 */
describe("TksRepl state commands", () => {
  let repl: TksRepl;
  let logSpy: jest.SpyInstance;

  beforeEach(() => {
    repl = new TksRepl(createDefaultState());
    logSpy = jest.spyOn(console, "log").mockImplementation(() => {});
  });

  afterEach(() => {
    logSpy.mockRestore();
  });

  test("set/show foundations and acquisitions update state and output", () => {
    repl.processLine("set foundation 3d 0.4");
    repl.processLine("set foundation 3d 2.0"); // clamps to 1.0
    repl.processLine("set acquisition 16 true");
    repl.processLine("set acquisition 17 no");

    const state = repl.getState();
    expect(state.foundations["3d"]).toBe(1.0);
    expect(state.acquisitions[16]).toBe(true);
    expect(state.acquisitions[17]).toBe(false);

    const showFoundations = repl.processLine("show foundations");
    expect(showFoundations.some((line) => line.includes("Foundations:"))).toBe(true);
    expect(showFoundations.some((line) => line.includes("3d: 1"))).toBe(true);

    const showAcquisitions = repl.processLine("show acquisitions");
    expect(showAcquisitions.some((line) => line.includes("Acquisitions:"))).toBe(true);
    expect(showAcquisitions.some((line) => line.includes("16: true"))).toBe(true);
  });
});
