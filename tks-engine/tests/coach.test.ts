import { coachForFoundation } from "../src/coach";
import { createDefaultState } from "../src/types";

describe("coachForFoundation", () => {
  test("suggests RPM and patterns for 6d when nothing acquired", () => {
    const state = createDefaultState();
    const coach = coachForFoundation("6d", state);

    expect(coach.goalFoundation).toBe("6d");
    expect(coach.missingAcquisitions).toEqual(expect.arrayContaining([16, 17, 18]));
    expect(coach.exampleExpressions.some((expr) => expr.startsWith("<<1:4:7>>("))).toBe(true);
    expect(coach.narrative).toContain("6d");
    expect(coach.narrative).toContain("Missing acquisitions");
  });

  test("reports satisfied RPM when acquisitions are present", () => {
    const state = createDefaultState();
    state.acquisitions[16] = true;
    state.acquisitions[17] = true;
    state.acquisitions[18] = true;
    state.foundations["6d"] = 0.3;

    const coach = coachForFoundation("6d", state);

    expect(coach.missingAcquisitions).toHaveLength(0);
    expect(coach.narrative.toLowerCase()).toContain("all required acquisitions present");
  });
});
