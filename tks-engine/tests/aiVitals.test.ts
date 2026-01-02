import { applyAiVitalsToFoundations, summarizeAiVitals, AiVitals } from "../src/aiVitals";
import { createDefaultState } from "../src/types";

describe("AI Vitals to TKS mapping", () => {
  test("basic mapping with overwrite", () => {
    const vitals: AiVitals = {
      id: "run-1",
      a_alignment: 1.0,
      b_coherence: 0.8,
      c_regulation: 0.6,
      d_grounding: 0.4,
      meta_complexity: 0.9,
      meta_reflexivity: 0.5,
    };

    const state = createDefaultState();
    applyAiVitalsToFoundations(vitals, state, { blend: false });

    expect(state.foundations["3d"]).toBeCloseTo(0.8);
    expect(state.foundations["4d"]).toBeCloseTo(0.6);
    expect(state.foundations["5d"]).toBeCloseTo(0.85);
    expect(state.foundations["6d"]).toBeCloseTo(0.4);
  });

  test("blending retains prior activation", () => {
    const vitals: AiVitals = {
      id: "run-2",
      a_alignment: 1.0,
      b_coherence: 0.8,
      c_regulation: 0.6,
      d_grounding: 0.4,
      meta_complexity: 0.9,
      meta_reflexivity: 0.5,
    };

    const state = createDefaultState();
    state.foundations["3d"] = 0.2;

    applyAiVitalsToFoundations(vitals, state, { blend: true });

    expect(state.foundations["3d"]).toBeCloseTo((0.2 + 0.8) / 2);
  });

  test("summarizeAiVitals reports targets and commentary", () => {
    const vitals: AiVitals = {
      id: "run-3",
      a_alignment: 0.7,
      b_coherence: 0.5,
      c_regulation: 0.4,
      d_grounding: 0.3,
      meta_complexity: 0.6,
      meta_reflexivity: 0.2,
    };

    const summary = summarizeAiVitals(vitals);

    expect(summary.foundations["3d"]).toBeDefined();
    expect(summary.foundations["4d"]).toBeDefined();
    expect(summary.foundations["5d"]).toBeDefined();
    expect(summary.foundations["6d"]).toBeDefined();
    expect(summary.commentary).toContain("run-3");
    expect(summary.commentary).toContain("health(3d)");
    expect(summary.commentary).toContain("material(6d)");
  });
});
