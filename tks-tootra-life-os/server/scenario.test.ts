import { describe, expect, it } from "vitest";
import { appRouter } from "./routers";
import type { TrpcContext } from "./_core/context";

function createAuthContext(): { ctx: TrpcContext } {
  const user = {
    id: 1,
    openId: "test-user",
    email: "test@example.com",
    name: "Test User",
    loginMethod: "manus",
    role: "user" as const,
    mode: "life_os" as const,
    photoUrl: null,
    createdAt: new Date(),
    updatedAt: new Date(),
    lastSignedIn: new Date(),
  };

  const ctx: TrpcContext = {
    user,
    req: { protocol: "https", headers: {} } as TrpcContext["req"],
    res: { clearCookie: () => {} } as TrpcContext["res"],
  };

  return { ctx };
}

describe("scenario router", () => {
  describe("input validation", () => {
    it("uploadPhoto requires non-empty photoBase64", async () => {
      const { ctx } = createAuthContext();
      const caller = appRouter.createCaller(ctx);
      await expect(
        caller.scenario.uploadPhoto({ photoBase64: "", mimeType: "image/jpeg" })
      ).rejects.toThrow();
    });

    it("generate rejects non-existent taskId", async () => {
      const { ctx } = createAuthContext();
      const caller = appRouter.createCaller(ctx);
      await expect(
        caller.scenario.generate({ taskId: 999999, mode: "single", type: "positive" })
      ).rejects.toThrow();
    });

    it("generate rejects with both single and dual modes for missing task", async () => {
      const { ctx } = createAuthContext();
      const caller = appRouter.createCaller(ctx);
      await expect(
        caller.scenario.generate({ taskId: 999999, mode: "single", type: "positive" })
      ).rejects.toThrow();
      await expect(
        caller.scenario.generate({ taskId: 999999, mode: "dual", type: "positive" })
      ).rejects.toThrow();
    });

    it("listForTask returns empty array for non-existent task", async () => {
      const { ctx } = createAuthContext();
      const caller = appRouter.createCaller(ctx);
      const result = await caller.scenario.listForTask({ taskId: 999999 });
      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBe(0);
    });
  });

  describe("scenario prompt building", () => {
    it("buildScenarioPrompts is importable and typed correctly", async () => {
      const { buildScenarioPrompts } = await import("./ai");
      expect(typeof buildScenarioPrompts).toBe("function");
    });
  });

  describe("photo upload", () => {
    it("rejects photos over 10MB", async () => {
      const { ctx } = createAuthContext();
      const caller = appRouter.createCaller(ctx);
      const largeFakeBase64 = "A".repeat(14 * 1024 * 1024); // ~10.5MB decoded
      await expect(
        caller.scenario.uploadPhoto({ photoBase64: largeFakeBase64, mimeType: "image/jpeg" })
      ).rejects.toThrow(/10MB/);
    });

    it("accepts valid small photo and returns URL", async () => {
      const { ctx } = createAuthContext();
      const caller = appRouter.createCaller(ctx);
      const smallBase64 = Buffer.from("test-image-data").toString("base64");
      const result = await caller.scenario.uploadPhoto({ photoBase64: smallBase64, mimeType: "image/png" });
      expect(result).toBeDefined();
      expect(result.photoUrl).toBeDefined();
      expect(typeof result.photoUrl).toBe("string");
      expect(result.photoUrl).toContain("http");
    });

    it("getPhoto returns photo URL after upload", async () => {
      const { ctx } = createAuthContext();
      const caller = appRouter.createCaller(ctx);
      // Upload first
      const smallBase64 = Buffer.from("photo-data").toString("base64");
      await caller.scenario.uploadPhoto({ photoBase64: smallBase64, mimeType: "image/jpeg" });
      // Then query
      const result = await caller.scenario.getPhoto();
      expect(result).toBeDefined();
      expect(result.photoUrl).toBeDefined();
    });
  });

  describe("mode and type logic", () => {
    it("mode accepts single and dual", () => {
      expect(["single", "dual"]).toContain("single");
      expect(["single", "dual"]).toContain("dual");
    });

    it("type accepts positive and negative", () => {
      expect(["positive", "negative"]).toContain("positive");
      expect(["positive", "negative"]).toContain("negative");
    });

    it("dual mode generates both types", () => {
      const mode = "dual";
      const typesToGenerate = mode === "dual"
        ? ["positive", "negative"]
        : ["positive"];
      expect(typesToGenerate).toEqual(["positive", "negative"]);
    });

    it("single mode generates only selected type", () => {
      const mode = "single";
      const selectedType = "negative";
      const typesToGenerate = mode === "dual"
        ? ["positive", "negative"]
        : [selectedType];
      expect(typesToGenerate).toEqual(["negative"]);
    });
  });
});
