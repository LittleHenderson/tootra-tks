import { TksRepl } from "../src/repl";
import { createDefaultState } from "../src/types";

describe("REPL coach command", () => {
  let repl: TksRepl;
  let logSpy: jest.SpyInstance;

  beforeEach(() => {
    repl = new TksRepl(createDefaultState());
    logSpy = jest.spyOn(console, "log").mockImplementation(() => {});
  });

  afterEach(() => {
    logSpy.mockRestore();
  });

  test("prints coach suggestions for a foundation", () => {
    repl.processLine("coach 6d");

    const logs = logSpy.mock.calls.map((args) => args.join(" ")).join("\n");
    expect(logs).toContain("Coach report for 6d");
    expect(logs).toContain("Suggested patterns");
  });

  test("rejects unknown foundation ids", () => {
    repl.processLine("coach 99z");

    const logs = logSpy.mock.calls.map((args) => args.join(" ")).join("\n");
    expect(logs.toLowerCase()).toContain("unknown foundation id");
  });
});
