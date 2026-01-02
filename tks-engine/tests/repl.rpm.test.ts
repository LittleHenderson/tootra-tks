import { TksRepl } from "../src/repl";
import { createDefaultState } from "../src/types";

describe("REPL RPM diagnostics", () => {
  let repl: TksRepl;
  let logSpy: jest.SpyInstance;

  beforeEach(() => {
    repl = new TksRepl(createDefaultState());
    logSpy = jest.spyOn(console, "log").mockImplementation(() => {});
  });

  afterEach(() => {
    logSpy.mockRestore();
  });

  test("reports missing acquisitions for 6d", () => {
    repl.processLine("set foundation 6d 0.0");
    repl.processLine("eval 6d");

    const logs = logSpy.mock.calls.map((args) => args.join(" ")).join("\n");
    expect(logs).toContain("RPM check for goal 6d");
    expect(logs).toContain("Missing acquisitions: 16, 17, 18");
  });

  test("reports all present acquisitions for 6d", () => {
    repl.processLine("set acquisition 16 true");
    repl.processLine("set acquisition 17 true");
    repl.processLine("set acquisition 18 true");
    repl.processLine("eval 6d");

    const logs = logSpy.mock.calls.map((args) => args.join(" ")).join("\n");
    expect(logs).toContain("RPM check for goal 6d");
    expect(logs).toContain("all required acquisitions present");
  });
});
