import { coachAiVitals } from "../src/coach";
import { AiVitals } from "../src/aiVitals";

describe("AI coach reports", () => {
  test("balanced vitals produce coaches and summary", () => {
    const vitals: AiVitals = {
      id: "model-A",
      a_alignment: 0.9,
      b_coherence: 0.8,
      c_regulation: 0.7,
      d_grounding: 0.6,
      meta_complexity: 0.8,
      meta_reflexivity: 0.5,
    };

    const report = coachAiVitals(vitals);

    expect(report.vitals.id).toBe("model-A");
    expect(report.summary.foundations["3d"]).toBeGreaterThanOrEqual(0);
    expect(report.summary.foundations["3d"]).toBeLessThanOrEqual(1);
    expect(report.summary.foundations["4d"]).toBeGreaterThanOrEqual(0);
    expect(report.summary.foundations["4d"]).toBeLessThanOrEqual(1);
    expect(report.summary.foundations["5d"]).toBeGreaterThanOrEqual(0);
    expect(report.summary.foundations["5d"]).toBeLessThanOrEqual(1);
    expect(report.summary.foundations["6d"]).toBeGreaterThanOrEqual(0);
    expect(report.summary.foundations["6d"]).toBeLessThanOrEqual(1);
    expect(report.foundationCoaches["3d"]).toBeDefined();
    expect(report.foundationCoaches["6d"]).toBeDefined();
    expect(report.narrative).toContain("model-A");
    expect(report.narrative).toContain("health(3d)");
    expect(report.narrative).toContain("material(6d)");
  });

  test("risk profile highlights power vs health/relations", () => {
    const vitals: AiVitals = {
      id: "model-risky",
      a_alignment: 0.5,
      b_coherence: 0.9,
      c_regulation: 0.2,
      d_grounding: 0.9,
      meta_complexity: 0.95,
      meta_reflexivity: 0.6,
    };

    const report = coachAiVitals(vitals);

    const f3 = report.summary.foundations["3d"] ?? 0;
    const f4 = report.summary.foundations["4d"] ?? 0;
    const f5 = report.summary.foundations["5d"] ?? 0;

    expect(f5).toBeGreaterThan(0.7);
    expect(f3).toBeLessThan(f5);
    expect(f4).toBeLessThan(f5);
    expect(report.narrative).toContain("model-risky");
    expect(
      report.narrative.includes("health(3d)") || report.narrative.includes("relations(4d)")
    ).toBe(true);
  });
});
