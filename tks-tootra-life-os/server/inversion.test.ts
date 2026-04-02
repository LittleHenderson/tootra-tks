import { describe, it, expect } from "vitest";
import {
  INVERSION_MODES,
  INVERSION_PRESETS,
  INVERSION_LAYERS,
  INVERSION_INTENSITIES,
  INVERSION_SCOPES,
  INVERSION_AXES,
  AXIS_OPERATIONS,
  getInversionMode,
  getInversionModeByKey,
  getModesByLayer,
  getPreset,
  getPresetModes,
  buildDefaultDialConfig,
  type InversionMode,
  type InversionLayer,
} from "../shared/inversions";

// ── Shared Constants Tests ──────────────────────────────────

describe("TKS Inversion Engine — Shared Constants", () => {
  it("should define exactly 29 inversion modes", () => {
    expect(INVERSION_MODES.length).toBe(29);
  });

  it("should have unique IDs for all modes", () => {
    const ids = INVERSION_MODES.map((m) => m.id);
    expect(new Set(ids).size).toBe(29);
  });

  it("should have unique keys for all modes", () => {
    const keys = INVERSION_MODES.map((m) => m.key);
    expect(new Set(keys).size).toBe(29);
  });

  it("should distribute modes across 4 layers", () => {
    const layers = new Set(INVERSION_MODES.map((m) => m.layer));
    expect(layers.size).toBe(4);
    expect(layers.has("surface")).toBe(true);
    expect(layers.has("deep_structural")).toBe(true);
    expect(layers.has("meta_layer")).toBe(true);
    expect(layers.has("special")).toBe(true);
  });

  it("should have correct mode counts per layer", () => {
    const counts: Record<string, number> = {};
    INVERSION_MODES.forEach((m) => { counts[m.layer] = (counts[m.layer] || 0) + 1; });
    expect(counts.surface).toBe(6);
    expect(counts.deep_structural).toBe(12);
    expect(counts.meta_layer).toBe(7);
    expect(counts.special).toBe(4);
  });

  it("should have required fields for every mode", () => {
    INVERSION_MODES.forEach((mode) => {
      expect(mode.id).toBeGreaterThan(0);
      expect(mode.key).toBeTruthy();
      expect(mode.name).toBeTruthy();
      expect(mode.layer).toBeTruthy();
      expect(mode.shortDescription).toBeTruthy();
      expect(mode.description).toBeTruthy();
      expect(mode.example).toBeDefined();
      expect(mode.example.before).toBeTruthy();
      expect(mode.example.after).toBeTruthy();
      expect(mode.axisGrid).toBeDefined();
    });
  });

  it("should have axis grid with valid axes for each mode", () => {
    const validAxes = Object.keys(INVERSION_AXES);
    const validOps = Object.keys(AXIS_OPERATIONS);
    INVERSION_MODES.forEach((mode) => {
      Object.entries(mode.axisGrid).forEach(([axis, op]) => {
        expect(validAxes).toContain(axis);
        expect(validOps).toContain(op);
      });
    });
  });
});

// ── Layer Metadata Tests ────────────────────────────────────

describe("TKS Inversion Engine — Layer Metadata", () => {
  it("should define all 4 layers with metadata", () => {
    expect(Object.keys(INVERSION_LAYERS).length).toBe(4);
    (["surface", "deep_structural", "meta_layer", "special"] as InversionLayer[]).forEach((layer) => {
      expect(INVERSION_LAYERS[layer]).toBeDefined();
      expect(INVERSION_LAYERS[layer].label).toBeTruthy();
      expect(INVERSION_LAYERS[layer].description).toBeTruthy();
      expect(INVERSION_LAYERS[layer].color).toBeTruthy();
    });
  });
});

// ── Intensity & Scope Tests ─────────────────────────────────

describe("TKS Inversion Engine — Intensity & Scope", () => {
  it("should define 3 intensity levels", () => {
    expect(Object.keys(INVERSION_INTENSITIES).length).toBe(3);
    expect(INVERSION_INTENSITIES.soft).toBeDefined();
    expect(INVERSION_INTENSITIES.medium).toBeDefined();
    expect(INVERSION_INTENSITIES.hard).toBeDefined();
  });

  it("should define 5 scope levels", () => {
    expect(Object.keys(INVERSION_SCOPES).length).toBe(5);
  });

  it("should have label and description for each intensity", () => {
    Object.values(INVERSION_INTENSITIES).forEach((info) => {
      expect(info.label).toBeTruthy();
      expect(info.description).toBeTruthy();
    });
  });

  it("should have label and description for each scope", () => {
    Object.values(INVERSION_SCOPES).forEach((info) => {
      expect(info.label).toBeTruthy();
      expect(info.description).toBeTruthy();
    });
  });
});

// ── Preset Recipes Tests ────────────────────────────────────

describe("TKS Inversion Engine — Preset Recipes", () => {
  it("should define at least 8 presets", () => {
    expect(INVERSION_PRESETS.length).toBeGreaterThanOrEqual(8);
  });

  it("should have unique IDs for all presets", () => {
    const ids = INVERSION_PRESETS.map((p) => p.id);
    expect(new Set(ids).size).toBe(INVERSION_PRESETS.length);
  });

  it("should have required fields for every preset", () => {
    INVERSION_PRESETS.forEach((preset) => {
      expect(preset.id).toBeTruthy();
      expect(preset.name).toBeTruthy();
      expect(preset.description).toBeTruthy();
      expect(preset.icon).toBeTruthy();
      expect(preset.color).toBeTruthy();
      expect(preset.modeIds.length).toBeGreaterThan(0);
      expect(["soft", "medium", "hard"]).toContain(preset.intensity);
      expect(["local", "term", "chain", "equation", "scenario"]).toContain(preset.scope);
    });
  });

  it("should reference only valid mode IDs", () => {
    const validIds = new Set(INVERSION_MODES.map((m) => m.id));
    INVERSION_PRESETS.forEach((preset) => {
      preset.modeIds.forEach((id) => {
        expect(validIds.has(id)).toBe(true);
      });
    });
  });

  it("should not exceed 5 modes per preset", () => {
    INVERSION_PRESETS.forEach((preset) => {
      expect(preset.modeIds.length).toBeLessThanOrEqual(5);
    });
  });
});

// ── Helper Function Tests ───────────────────────────────────

describe("TKS Inversion Engine — Helper Functions", () => {
  it("getInversionMode should return correct mode by ID", () => {
    const mode = getInversionMode(1);
    expect(mode).toBeDefined();
    expect(mode?.id).toBe(1);
    expect(mode?.key).toBe("opposite");
    expect(mode?.layer).toBe("surface");
  });

  it("getInversionMode should return undefined for invalid ID", () => {
    expect(getInversionMode(999)).toBeUndefined();
    expect(getInversionMode(0)).toBeUndefined();
    expect(getInversionMode(-1)).toBeUndefined();
  });

  it("getInversionModeByKey should return correct mode", () => {
    const mode = getInversionModeByKey("opposite");
    expect(mode).toBeDefined();
    expect(mode?.id).toBe(1);
    expect(mode?.name).toBe("Opposite");
  });

  it("getInversionModeByKey should return undefined for invalid key", () => {
    expect(getInversionModeByKey("nonexistent")).toBeUndefined();
  });

  it("getModesByLayer should return correct modes for each layer", () => {
    expect(getModesByLayer("surface").length).toBe(6);
    expect(getModesByLayer("deep_structural").length).toBe(12);
    expect(getModesByLayer("meta_layer").length).toBe(7);
    expect(getModesByLayer("special").length).toBe(4);
  });

  it("getPreset should return correct preset by ID", () => {
    const preset = getPreset("shadow_self");
    expect(preset).toBeDefined();
    expect(preset?.name).toContain("Shadow");
  });

  it("getPreset should return undefined for invalid ID", () => {
    expect(getPreset("nonexistent")).toBeUndefined();
  });

  it("getPresetModes should return mode objects for a preset", () => {
    const preset = INVERSION_PRESETS[0];
    const modes = getPresetModes(preset);
    expect(modes.length).toBe(preset.modeIds.length);
    modes.forEach((mode) => {
      expect(mode).toBeDefined();
      expect(preset.modeIds).toContain(mode.id);
    });
  });

  it("buildDefaultDialConfig should create valid config", () => {
    const config = buildDefaultDialConfig([1, 2, 3]);
    expect(config.modeIds).toEqual([1, 2, 3]);
    expect(config.intensity).toBe("medium");
    expect(config.scope).toBe("equation");
    expect(config.direction).toBe("forward");
  });

  it("buildDefaultDialConfig should work with empty array", () => {
    const config = buildDefaultDialConfig([]);
    expect(config.modeIds).toEqual([]);
    expect(config.intensity).toBe("medium");
  });
});

// ── Surface Layer Mode Verification ─────────────────────────

describe("TKS Inversion Engine — Surface Layer Modes", () => {
  const surfaceModes = INVERSION_MODES.filter((m) => m.layer === "surface");

  it("should include Opposite inversion", () => {
    expect(surfaceModes.find((m) => m.key === "opposite")).toBeDefined();
  });

  it("should include Dual inversion", () => {
    expect(surfaceModes.find((m) => m.key === "dual")).toBeDefined();
  });

  it("should include Counter-Pole inversion", () => {
    expect(surfaceModes.find((m) => m.key === "counter_pole")).toBeDefined();
  });

  it("should include Mirror inversion", () => {
    expect(surfaceModes.find((m) => m.key === "mirror")).toBeDefined();
  });

  it("should include Reverse-Causal inversion", () => {
    expect(surfaceModes.find((m) => m.key === "reverse_causal")).toBeDefined();
  });

  it("should include Parallel Analogue inversion", () => {
    expect(surfaceModes.find((m) => m.key === "parallel_analogue")).toBeDefined();
  });
});

// ── Deep Structural Layer Verification ──────────────────────

describe("TKS Inversion Engine — Deep Structural Layer Modes", () => {
  const deepModes = INVERSION_MODES.filter((m) => m.layer === "deep_structural");

  it("should have exactly 12 deep structural modes", () => {
    expect(deepModes.length).toBe(12);
  });

  it("should include Noetic Complement", () => {
    expect(deepModes.find((m) => m.key === "noetic_complement")).toBeDefined();
  });

  it("should include Foundation Flip", () => {
    expect(deepModes.find((m) => m.key === "foundation_flip")).toBeDefined();
  });

  it("should include Temporal inversion", () => {
    expect(deepModes.find((m) => m.key === "temporal")).toBeDefined();
  });

  it("should include Scalar inversion", () => {
    expect(deepModes.find((m) => m.key === "scalar")).toBeDefined();
  });

  it("should include all 12 deep structural keys", () => {
    const expectedKeys = [
      "noetic_complement", "acquisition_polarity", "foundation_flip",
      "sub_foundation_reversal", "domain_permutation", "temporal",
      "scalar", "structural_permutation", "context_frame",
      "motivational", "polarity", "causal_density",
    ];
    expectedKeys.forEach((key) => {
      expect(deepModes.find((m) => m.key === key)).toBeDefined();
    });
  });
});

// ── Meta-layer Verification ─────────────────────────────────

describe("TKS Inversion Engine — Meta-layer Modes", () => {
  const metaModes = INVERSION_MODES.filter((m) => m.layer === "meta_layer");

  it("should have exactly 7 meta-layer modes", () => {
    expect(metaModes.length).toBe(7);
  });

  it("should include all 7 meta-layer keys", () => {
    const expectedKeys = [
      "attention", "value", "desire_inhibition",
      "expectation", "attractor", "stability", "entropy",
    ];
    expectedKeys.forEach((key) => {
      expect(metaModes.find((m) => m.key === key)).toBeDefined();
    });
  });
});

// ── Special Transformations Verification ────────────────────

describe("TKS Inversion Engine — Special Transformations", () => {
  const specialModes = INVERSION_MODES.filter((m) => m.layer === "special");

  it("should have exactly 4 special modes", () => {
    expect(specialModes.length).toBe(4);
  });

  it("should include all 4 special keys", () => {
    const expectedKeys = ["constraint", "boundary", "semantic_parity", "agent_role"];
    expectedKeys.forEach((key) => {
      expect(specialModes.find((m) => m.key === key)).toBeDefined();
    });
  });
});

// ── Dial Config Integration Tests ───────────────────────────

describe("TKS Inversion Engine — Dial Config", () => {
  it("should allow overriding intensity after build", () => {
    const config = buildDefaultDialConfig([1]);
    config.intensity = "hard";
    expect(config.intensity).toBe("hard");
  });

  it("should allow overriding scope after build", () => {
    const config = buildDefaultDialConfig([1]);
    config.scope = "scenario";
    expect(config.scope).toBe("scenario");
  });

  it("should allow overriding direction after build", () => {
    const config = buildDefaultDialConfig([1]);
    config.direction = "bidirectional";
    expect(config.direction).toBe("bidirectional");
  });

  it("should preserve mode IDs through config", () => {
    const ids = [1, 7, 15, 22, 26];
    const config = buildDefaultDialConfig(ids);
    expect(config.modeIds).toEqual(ids);
  });
});

// ── Cross-validation Tests ──────────────────────────────────

describe("TKS Inversion Engine — Cross-validation", () => {
  it("every mode ID should be retrievable by getInversionMode", () => {
    INVERSION_MODES.forEach((mode) => {
      const found = getInversionMode(mode.id);
      expect(found).toBeDefined();
      expect(found?.key).toBe(mode.key);
    });
  });

  it("every mode key should be retrievable by getInversionModeByKey", () => {
    INVERSION_MODES.forEach((mode) => {
      const found = getInversionModeByKey(mode.key);
      expect(found).toBeDefined();
      expect(found?.id).toBe(mode.id);
    });
  });

  it("mode IDs should be sequential from 1 to 29", () => {
    const ids = INVERSION_MODES.map((m) => m.id).sort((a, b) => a - b);
    for (let i = 0; i < 29; i++) {
      expect(ids[i]).toBe(i + 1);
    }
  });

  it("every preset should resolve to valid modes via getPresetModes", () => {
    INVERSION_PRESETS.forEach((preset) => {
      const modes = getPresetModes(preset);
      expect(modes.length).toBe(preset.modeIds.length);
      modes.forEach((mode) => {
        expect(mode.id).toBeGreaterThan(0);
      });
    });
  });
});
