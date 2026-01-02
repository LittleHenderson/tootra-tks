import { getAiProvider } from "../src/aiAssist";

describe("aiAssist providers", () => {
  const originalEnv = process.env.TKS_AI_PROVIDER;

  afterEach(() => {
    process.env.TKS_AI_PROVIDER = originalEnv;
  });

  test("default provider is claude when nothing is set", async () => {
    delete process.env.TKS_AI_PROVIDER;
    const provider = getAiProvider();
    expect(provider.name).toBe("claude");
    const result = await provider.complete({ prompt: "hello" });
    expect(result).toContain("[claude stub]");
    expect(result).toContain("hello");
  });

  test("env selects gemini", async () => {
    process.env.TKS_AI_PROVIDER = "gemini";
    const provider = getAiProvider();
    expect(provider.name).toBe("gemini");
    const result = await provider.complete({ prompt: "ping" });
    expect(result).toContain("[gemini stub");
    expect(result).toContain("ping");
  });

  test("config overrides env", async () => {
    process.env.TKS_AI_PROVIDER = "claude";
    const provider = getAiProvider({ provider: "openai" });
    expect(provider.name).toBe("openai");
    const result = await provider.complete({ prompt: "q", system: "s" });
    expect(result).toContain("[openai stub|system]");
    expect(result).toContain("q");
  });
});
