export type AiProviderName = "openai" | "claude" | "gemini";

export interface AiProvider {
  name: AiProviderName;
  complete(params: { prompt: string; system?: string }): Promise<string>;
}

export interface AiConfig {
  provider?: AiProviderName;
}

function normalizeProviderName(raw?: string): AiProviderName | undefined {
  if (!raw) return undefined;
  const lower = raw.toLowerCase();
  if (lower.startsWith("claude")) return "claude";
  if (lower.startsWith("gemini")) return "gemini";
  if (lower.startsWith("openai")) return "openai";
  return undefined;
}

export function getAiProvider(config?: AiConfig): AiProvider {
  const env = normalizeProviderName(process.env.TKS_AI_PROVIDER);
  const providerName: AiProviderName = config?.provider ?? env ?? "claude";

  switch (providerName) {
    case "gemini":
      return createGeminiProvider();
    case "openai":
      return createOpenAiProvider();
    case "claude":
    default:
      return createClaudeProvider();
  }
}

function createClaudeProvider(): AiProvider {
  return {
    name: "claude",
    async complete({ prompt, system }) {
      const prefix = system ? `[claude stub|system]` : "[claude stub]";
      return `${prefix} ${prompt}`;
    },
  };
}

function createGeminiProvider(): AiProvider {
  return {
    name: "gemini",
    async complete({ prompt, system }) {
      const prefix = system ? `[gemini stub|system]` : "[gemini stub]";
      return `${prefix} ${prompt}`;
    },
  };
}

function createOpenAiProvider(): AiProvider {
  return {
    name: "openai",
    async complete({ prompt, system }) {
      const prefix = system ? `[openai stub|system]` : "[openai stub]";
      return `${prefix} ${prompt}`;
    },
  };
}
