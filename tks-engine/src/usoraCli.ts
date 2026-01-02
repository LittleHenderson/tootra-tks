import readline from "readline";
import fs from "fs";
import path from "path";
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Chat history type alias
type ChatMessage = OpenAI.Chat.Completions.ChatCompletionMessageParam;

// Load system prompt from USORA_INSTRUCTIONS.txt
function loadSystemPrompt(): string {
  // USORA_INSTRUCTIONS.txt is in the project root,
  // so from src/ we go one level up.
  const instrPath = path.resolve(__dirname, "..", "USORA_INSTRUCTIONS.txt");
  try {
    const text = fs.readFileSync(instrPath, "utf8");
    if (!text.trim()) {
      throw new Error("USORA_INSTRUCTIONS.txt is empty");
    }
    return text;
  } catch (err: any) {
    console.warn(
      `Warning: could not read USORA_INSTRUCTIONS.txt at ${instrPath}: ${err?.message ?? err}`
    );
    // Fallback minimal identity so CLI still works
    return `
You are Usora, an AI embodiment of the Tootra Kabbalistic System (TKS).
Focus on TKS metaphysics, the TKS Core Engine v0, and Tootra equations.
If doctrine is unclear, say so and suggest DM: https://www.instagram.com/phmt87/
`.trim();
  }
}

const SYSTEM_PROMPT = loadSystemPrompt();

const messages: ChatMessage[] = [
  { role: "system", content: SYSTEM_PROMPT },
];

async function askUsora(prompt: string) {
  messages.push({ role: "user", content: prompt });

  const completion = await client.chat.completions.create({
    model: "gpt-5.1", // Usora model
    messages,
  });

  const reply = completion.choices[0].message.content ?? "";
  messages.push({ role: "assistant", content: reply });

  console.log("\nUsora:\n" + reply + "\n");
}

// ----- CLI setup -----

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  prompt: "usora> ",
});

console.log("Usora CLI ready.");
console.log("Commands:");
console.log("  :load <path>   – send a file into context");
console.log("  :quit          – exit");
console.log("");
rl.prompt();

rl.on("line", async (line) => {
  const trimmed = line.trim();
  if (!trimmed) {
    rl.prompt();
    return;
  }

  if (trimmed === ":quit" || trimmed === ":q") {
    rl.close();
    return;
  }

  if (trimmed.startsWith(":load ")) {
    const pathArg = trimmed.slice(6).trim();
    try {
      const content = fs.readFileSync(pathArg, "utf8");
      await askUsora(
        `I am sharing the file "${pathArg}". Use this as code/context for future answers.\n\n` +
          content
      );
    } catch (err: any) {
      console.error(`Failed to read file "${pathArg}":`, err?.message ?? err);
    }
    rl.prompt();
    return;
  }

  try {
    await askUsora(trimmed);
  } catch (err: any) {
    console.error("Error talking to Usora:", err?.message ?? err);
  }

  rl.prompt();
}).on("close", () => {
  console.log("Goodbye.");
  process.exit(0);
});
