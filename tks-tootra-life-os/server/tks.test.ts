import { describe, expect, it } from "vitest";
import { appRouter } from "./routers";
import type { TrpcContext } from "./_core/context";
import {
  FOUNDATIONS,
  SUB_FOUNDATIONS,
  NOETICS,
  OPERATORS,
  TOKEN_TYPES,
  DWP_STAGES,
  TASK_ARCHETYPES,
  TASK_STATUSES,
  POWER_CATEGORIES,
  PRIORITY_WEIGHTS,
  MODES,
  WORLDS,
} from "../shared/tks";

// ── Helper: Create authenticated context ──────────────────
function createAuthContext(): TrpcContext {
  return {
    user: {
      id: 1,
      openId: "test-user-001",
      email: "test@example.com",
      name: "Test User",
      loginMethod: "manus",
      role: "user",
      mode: "life_os",
      createdAt: new Date(),
      updatedAt: new Date(),
      lastSignedIn: new Date(),
    },
    req: {
      protocol: "https",
      headers: {},
    } as TrpcContext["req"],
    res: {
      clearCookie: () => {},
    } as TrpcContext["res"],
  };
}

// ═══════════════════════════════════════════════════════════
// TKS Shared Types Validation
// ═══════════════════════════════════════════════════════════

describe("TKS Foundations", () => {
  it("has exactly 7 foundations", () => {
    expect(FOUNDATIONS).toHaveLength(7);
  });

  it("has unique IDs from 1 to 7", () => {
    const ids = FOUNDATIONS.map(f => f.id);
    expect(ids).toEqual([1, 2, 3, 4, 5, 6, 7]);
  });

  it("contains the correct foundation names", () => {
    const names = FOUNDATIONS.map(f => f.name);
    expect(names).toEqual(["Self", "Knowledge", "Body", "Relationships", "Work", "Environment", "Spirit"]);
  });

  it("each foundation has required fields", () => {
    for (const f of FOUNDATIONS) {
      expect(f).toHaveProperty("id");
      expect(f).toHaveProperty("name");
      expect(f).toHaveProperty("symbol");
      expect(f).toHaveProperty("world");
      expect(f).toHaveProperty("color");
      expect(f).toHaveProperty("description");
    }
  });

  it("maps to correct worlds", () => {
    const worldMap: Record<string, number[]> = {};
    for (const f of FOUNDATIONS) {
      if (!worldMap[f.world]) worldMap[f.world] = [];
      worldMap[f.world].push(f.id);
    }
    expect(worldMap["Fire"]).toEqual([1, 5]);
    expect(worldMap["Earth"]).toEqual([2, 6]);
    expect(worldMap["Air"]).toEqual([3, 7]);
    expect(worldMap["Water"]).toEqual([4]);
  });
});

describe("TKS Sub-Foundations", () => {
  it("has exactly 28 sub-foundations (4 per foundation)", () => {
    expect(SUB_FOUNDATIONS).toHaveLength(28);
  });

  it("each foundation has exactly 4 sub-foundations", () => {
    for (let i = 1; i <= 7; i++) {
      const subs = SUB_FOUNDATIONS.filter(sf => sf.foundationId === i);
      expect(subs).toHaveLength(4);
    }
  });

  it("sub-foundation IDs follow pattern NX where N is foundation ID", () => {
    for (const sf of SUB_FOUNDATIONS) {
      const expectedPrefix = sf.foundationId.toString();
      expect(sf.id.startsWith(expectedPrefix)).toBe(true);
    }
  });
});

describe("TKS Noetics", () => {
  it("has exactly 10 noetics", () => {
    expect(NOETICS).toHaveLength(10);
  });

  it("has IDs from 0 to 9", () => {
    const ids = NOETICS.map(n => n.id);
    expect(ids).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });

  it("8th Noetic is Above/Cause (not Above/Case)", () => {
    const eighth = NOETICS.find(n => n.id === 8);
    expect(eighth).toBeDefined();
    expect(eighth!.name).toBe("Above/Cause");
    expect(eighth!.name).not.toBe("Above/Case");
  });

  it("9th Noetic is Below/Effect", () => {
    const ninth = NOETICS.find(n => n.id === 9);
    expect(ninth).toBeDefined();
    expect(ninth!.name).toBe("Below/Effect");
  });

  it("symbols follow ^N pattern", () => {
    for (const n of NOETICS) {
      expect(n.symbol).toBe(`^${n.id}`);
    }
  });
});

describe("TKS Operators", () => {
  it("has exactly 11 operators", () => {
    expect(OPERATORS).toHaveLength(11);
  });

  it("includes all required operators", () => {
    const ids = OPERATORS.map(o => o.id);
    expect(ids).toContain("o");
    expect(ids).toContain("->");
    expect(ids).toContain("<-");
    expect(ids).toContain("+");
    expect(ids).toContain("-");
    expect(ids).toContain("*");
    expect(ids).toContain("/");
    expect(ids).toContain("||");
    expect(ids).toContain("?");
    expect(ids).toContain("!");
    expect(ids).toContain("~");
  });
});

describe("TKS Token Types", () => {
  it("has exactly 13 token types", () => {
    const keys = Object.keys(TOKEN_TYPES);
    expect(keys).toHaveLength(13);
  });

  it("includes all 13 required token types", () => {
    const expected = ["ACT", "TIME", "DUR", "RHY", "MAP", "NOE", "LINK", "D", "W", "P", "REQ", "HAVE", "LACK"];
    for (const key of expected) {
      expect(TOKEN_TYPES).toHaveProperty(key);
    }
  });

  it("each token type has color, bgColor, label, and icon", () => {
    for (const [key, def] of Object.entries(TOKEN_TYPES)) {
      expect(def).toHaveProperty("color");
      expect(def).toHaveProperty("bgColor");
      expect(def).toHaveProperty("label");
      expect(def).toHaveProperty("icon");
    }
  });
});

describe("D/W/P Lifecycle", () => {
  it("has 12 total stages (4 per phase)", () => {
    expect(DWP_STAGES.desire).toHaveLength(4);
    expect(DWP_STAGES.wisdom).toHaveLength(4);
    expect(DWP_STAGES.power).toHaveLength(4);
  });

  it("desire stages are in correct order", () => {
    const ids = DWP_STAGES.desire.map(s => s.id);
    expect(ids).toEqual(["D_MIDNIGHT", "D_SUNRISE", "D_NOON", "D_SUNSET"]);
  });

  it("wisdom stages are in correct order", () => {
    const ids = DWP_STAGES.wisdom.map(s => s.id);
    expect(ids).toEqual(["W_IGNORANCE", "W_LEARNING", "W_KNOWING", "W_MASTERING"]);
  });

  it("power stages are in correct order", () => {
    const ids = DWP_STAGES.power.map(s => s.id);
    expect(ids).toEqual(["P_BIRTH", "P_ADOLESCENCE", "P_MATURITY", "P_MASTERY"]);
  });

  it("each stage has card and zodiac associations", () => {
    for (const phase of ["desire", "wisdom", "power"] as const) {
      for (const stage of DWP_STAGES[phase]) {
        expect(stage).toHaveProperty("card");
        expect(stage).toHaveProperty("zodiac");
        expect(stage.card).toBeTruthy();
        expect(stage.zodiac).toBeTruthy();
      }
    }
  });
});

describe("Task Archetypes", () => {
  it("has exactly 6 archetypes", () => {
    expect(TASK_ARCHETYPES).toHaveLength(6);
  });

  it("includes all required archetypes", () => {
    const ids = TASK_ARCHETYPES.map(a => a.id);
    expect(ids).toEqual(["execution", "prep", "maintenance", "repair", "study", "audit"]);
  });
});

describe("Task Statuses", () => {
  it("has exactly 5 statuses", () => {
    expect(TASK_STATUSES).toHaveLength(5);
  });

  it("follows the correct status machine order", () => {
    expect(TASK_STATUSES).toEqual(["inbox", "scheduled", "active", "done", "dropped"]);
  });
});

describe("Power Categories", () => {
  it("has exactly 8 categories", () => {
    expect(POWER_CATEGORIES).toHaveLength(8);
  });

  it("includes all 8 required categories", () => {
    const ids = POWER_CATEGORIES.map(p => p.id);
    expect(ids).toEqual(["time", "money", "tools", "access", "body", "skill", "connections", "environment"]);
  });
});

describe("Priority Weights", () => {
  it("weights sum to 1.0", () => {
    const sum = Object.values(PRIORITY_WEIGHTS).reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1.0, 5);
  });

  it("includes all 6 required weight factors", () => {
    expect(PRIORITY_WEIGHTS).toHaveProperty("deadlineProximity");
    expect(PRIORITY_WEIGHTS).toHaveProperty("foundationNeglect");
    expect(PRIORITY_WEIGHTS).toHaveProperty("desireStage");
    expect(PRIORITY_WEIGHTS).toHaveProperty("powerCompleteness");
    expect(PRIORITY_WEIGHTS).toHaveProperty("urgency");
    expect(PRIORITY_WEIGHTS).toHaveProperty("rhythm");
  });
});

describe("Operational Modes", () => {
  it("has exactly 2 modes", () => {
    expect(MODES).toHaveLength(2);
  });

  it("includes life_os and work_sprint", () => {
    expect(MODES).toEqual(["life_os", "work_sprint"]);
  });
});

describe("Worlds", () => {
  it("has exactly 4 worlds", () => {
    expect(WORLDS).toHaveLength(4);
  });

  it("includes Fire, Earth, Air, Water", () => {
    const names = WORLDS.map(w => w.name);
    expect(names).toEqual(["Fire", "Earth", "Air", "Water"]);
  });
});

// ═══════════════════════════════════════════════════════════
// Router Tests (auth.me, settings)
// ═══════════════════════════════════════════════════════════

describe("auth.me", () => {
  it("returns user when authenticated", async () => {
    const ctx = createAuthContext();
    const caller = appRouter.createCaller(ctx);
    const result = await caller.auth.me();
    expect(result).toBeDefined();
    expect(result?.name).toBe("Test User");
    expect(result?.email).toBe("test@example.com");
  });

  it("returns null when not authenticated", async () => {
    const ctx: TrpcContext = {
      user: null,
      req: { protocol: "https", headers: {} } as TrpcContext["req"],
      res: { clearCookie: () => {} } as TrpcContext["res"],
    };
    const caller = appRouter.createCaller(ctx);
    const result = await caller.auth.me();
    expect(result).toBeNull();
  });
});

describe("settings.getMode", () => {
  it("returns user mode", async () => {
    const ctx = createAuthContext();
    const caller = appRouter.createCaller(ctx);
    const result = await caller.settings.getMode();
    expect(result).toHaveProperty("mode");
    expect(["life_os", "work_sprint"]).toContain(result.mode);
  });
});
