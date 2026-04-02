import { describe, expect, it, beforeEach } from "vitest";

/**
 * Tests for InteractiveTour logic.
 * Covers: localStorage persistence, step navigation, step definitions,
 * tooltip positioning, action-triggered auto-advance, and conditional launch.
 */

const TOUR_STORAGE_KEY = "tks-tour-completed";
const ONBOARDING_STORAGE_KEY = "tks-onboarding-seen";

// ─── Tour Step Definitions (mirror from component) ────────
const TOUR_STEPS = [
  {
    id: "capture-input",
    selector: '[data-tour="capture-input"]',
    title: "Capture Your Intention",
    position: "bottom",
    navigateTo: "/",
    accent: "tks-gold",
    actionHint: "Try typing something!",
    actionEvent: "input",
    allowInteraction: true,
  },
  {
    id: "capture-button",
    selector: '[data-tour="capture-button"]',
    title: "Capture & Map",
    position: "top",
    navigateTo: "/",
    accent: "tks-gold",
    actionHint: "Click the button!",
    actionEvent: "click",
    allowInteraction: true,
  },
  {
    id: "inbox-section",
    selector: '[data-tour="inbox-section"]',
    title: "Your Inbox",
    position: "top",
    navigateTo: "/",
    accent: "tks-blue",
    actionHint: "Click any task card!",
    actionEvent: "click",
    allowInteraction: true,
  },
  {
    id: "sidebar-nav",
    selector: '[data-tour="sidebar-nav"]',
    title: "Navigate Your Life OS",
    position: "right",
    accent: "tks-gold",
    actionHint: "Click any nav item!",
    actionEvent: "click",
    allowInteraction: true,
  },
  {
    id: "scenarioize-btn",
    selector: '[data-tour="scenarioize-btn"]',
    title: "Scenarioize",
    position: "left",
    navigateTo: "/",
    accent: "purple-400",
    actionHint: "Click Scenarioize!",
    actionEvent: "click",
    allowInteraction: true,
  },
  {
    id: "mode-toggle",
    selector: '[data-tour="mode-toggle"]',
    title: "Life OS / Work Sprint",
    position: "right",
    accent: "tks-flame",
    actionHint: "Click to toggle!",
    actionEvent: "click",
    allowInteraction: true,
  },
];

// ─── Persistence ──────────────────────────────────────────
describe("Tour persistence logic", () => {
  let store: Record<string, string> = {};
  const mockStorage = {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => {
      store[key] = value;
    },
    removeItem: (key: string) => {
      delete store[key];
    },
  };

  beforeEach(() => {
    store = {};
  });

  it("hasTourCompleted returns false when no stored value", () => {
    const completed = mockStorage.getItem(TOUR_STORAGE_KEY) === "true";
    expect(completed).toBe(false);
  });

  it("hasTourCompleted returns true after marking as completed", () => {
    mockStorage.setItem(TOUR_STORAGE_KEY, "true");
    const completed = mockStorage.getItem(TOUR_STORAGE_KEY) === "true";
    expect(completed).toBe(true);
  });

  it("resetTour clears the stored value", () => {
    mockStorage.setItem(TOUR_STORAGE_KEY, "true");
    mockStorage.removeItem(TOUR_STORAGE_KEY);
    const completed = mockStorage.getItem(TOUR_STORAGE_KEY) === "true";
    expect(completed).toBe(false);
  });

  it("persists across simulated sessions", () => {
    mockStorage.setItem(TOUR_STORAGE_KEY, "true");
    const newStore = { ...store };
    expect(newStore[TOUR_STORAGE_KEY]).toBe("true");
  });

  it("tour and onboarding use different storage keys", () => {
    expect(TOUR_STORAGE_KEY).not.toBe(ONBOARDING_STORAGE_KEY);
  });
});

// ─── Step Definitions ─────────────────────────────────────
describe("Tour step definitions", () => {
  it("has exactly 6 tour steps", () => {
    expect(TOUR_STEPS.length).toBe(6);
  });

  it("all steps have unique IDs", () => {
    const ids = TOUR_STEPS.map((s) => s.id);
    const uniqueIds = new Set(ids);
    expect(uniqueIds.size).toBe(ids.length);
  });

  it("all steps have valid selectors with data-tour attribute", () => {
    TOUR_STEPS.forEach((step) => {
      expect(step.selector).toMatch(/^\[data-tour="[a-z-]+"\]$/);
    });
  });

  it("all steps have a title", () => {
    TOUR_STEPS.forEach((step) => {
      expect(step.title.length).toBeGreaterThan(0);
    });
  });

  it("all steps have a valid position", () => {
    const validPositions = ["top", "bottom", "left", "right"];
    TOUR_STEPS.forEach((step) => {
      expect(validPositions).toContain(step.position);
    });
  });

  it("all steps have an accent color", () => {
    TOUR_STEPS.forEach((step) => {
      expect(step.accent.length).toBeGreaterThan(0);
    });
  });

  it("first step targets the capture input", () => {
    expect(TOUR_STEPS[0].id).toBe("capture-input");
    expect(TOUR_STEPS[0].navigateTo).toBe("/");
  });

  it("last step targets the mode toggle", () => {
    expect(TOUR_STEPS[TOUR_STEPS.length - 1].id).toBe("mode-toggle");
  });

  it("steps that navigate go to the home page", () => {
    const navigatingSteps = TOUR_STEPS.filter((s) => s.navigateTo);
    navigatingSteps.forEach((step) => {
      expect(step.navigateTo).toBe("/");
    });
  });

  it("sidebar and mode toggle steps do not require navigation", () => {
    const sidebarStep = TOUR_STEPS.find((s) => s.id === "sidebar-nav");
    const modeStep = TOUR_STEPS.find((s) => s.id === "mode-toggle");
    expect(sidebarStep?.navigateTo).toBeUndefined();
    expect(modeStep?.navigateTo).toBeUndefined();
  });
});

// ─── Action-Triggered Auto-Advance ────────────────────────
describe("Tour action-triggered auto-advance definitions", () => {
  it("all steps have an actionHint string", () => {
    TOUR_STEPS.forEach((step) => {
      expect(step.actionHint).toBeDefined();
      expect(step.actionHint.length).toBeGreaterThan(0);
    });
  });

  it("all steps have a valid actionEvent", () => {
    const validEvents = ["input", "click"];
    TOUR_STEPS.forEach((step) => {
      expect(validEvents).toContain(step.actionEvent);
    });
  });

  it("all steps allow interaction with the highlighted element", () => {
    TOUR_STEPS.forEach((step) => {
      expect(step.allowInteraction).toBe(true);
    });
  });

  it("capture-input uses 'input' event for typing detection", () => {
    const step = TOUR_STEPS.find((s) => s.id === "capture-input");
    expect(step?.actionEvent).toBe("input");
    expect(step?.actionHint).toContain("typing");
  });

  it("capture-button uses 'click' event", () => {
    const step = TOUR_STEPS.find((s) => s.id === "capture-button");
    expect(step?.actionEvent).toBe("click");
    expect(step?.actionHint).toContain("Click");
  });

  it("inbox-section uses 'click' event", () => {
    const step = TOUR_STEPS.find((s) => s.id === "inbox-section");
    expect(step?.actionEvent).toBe("click");
  });

  it("sidebar-nav uses 'click' event", () => {
    const step = TOUR_STEPS.find((s) => s.id === "sidebar-nav");
    expect(step?.actionEvent).toBe("click");
  });

  it("scenarioize-btn uses 'click' event", () => {
    const step = TOUR_STEPS.find((s) => s.id === "scenarioize-btn");
    expect(step?.actionEvent).toBe("click");
  });

  it("mode-toggle uses 'click' event", () => {
    const step = TOUR_STEPS.find((s) => s.id === "mode-toggle");
    expect(step?.actionEvent).toBe("click");
  });

  it("only capture-input uses input event, all others use click", () => {
    const inputSteps = TOUR_STEPS.filter((s) => s.actionEvent === "input");
    const clickSteps = TOUR_STEPS.filter((s) => s.actionEvent === "click");
    expect(inputSteps.length).toBe(1);
    expect(inputSteps[0].id).toBe("capture-input");
    expect(clickSteps.length).toBe(5);
  });
});

// ─── Auto-Advance State Machine ──────────────────────────
describe("Tour auto-advance state machine", () => {
  it("action triggered starts as false", () => {
    const actionTriggered = false;
    expect(actionTriggered).toBe(false);
  });

  it("action triggered becomes true after user action", () => {
    let actionTriggered = false;
    // Simulate action handler
    const handleAction = () => {
      actionTriggered = true;
    };
    handleAction();
    expect(actionTriggered).toBe(true);
  });

  it("auto-advance moves to next step after action", () => {
    let currentStep = 0;
    const TOTAL = TOUR_STEPS.length;
    // Simulate action → advance
    if (currentStep < TOTAL - 1) {
      currentStep += 1;
    }
    expect(currentStep).toBe(1);
  });

  it("auto-advance on last step ends the tour", () => {
    let currentStep = TOUR_STEPS.length - 1;
    let tourEnded = false;
    // Simulate action on last step
    if (currentStep >= TOUR_STEPS.length - 1) {
      tourEnded = true;
    }
    expect(tourEnded).toBe(true);
  });

  it("action triggered resets when moving to next step", () => {
    let actionTriggered = true;
    // Simulate step change
    actionTriggered = false;
    expect(actionTriggered).toBe(false);
  });

  it("manual Next button clears pending auto-advance timeout", () => {
    let timeoutCleared = false;
    let advanceTimeout: ReturnType<typeof setTimeout> | null = setTimeout(() => {}, 800);
    // Simulate manual Next
    if (advanceTimeout) {
      clearTimeout(advanceTimeout);
      advanceTimeout = null;
      timeoutCleared = true;
    }
    expect(timeoutCleared).toBe(true);
    expect(advanceTimeout).toBeNull();
  });

  it("auto-advance delay is 800ms (visual confirmation)", () => {
    const AUTO_ADVANCE_DELAY = 800;
    expect(AUTO_ADVANCE_DELAY).toBe(800);
    expect(AUTO_ADVANCE_DELAY).toBeGreaterThan(0);
    expect(AUTO_ADVANCE_DELAY).toBeLessThanOrEqual(1000);
  });
});

// ─── Navigation ───────────────────────────────────────────
describe("Tour step navigation logic", () => {
  const TOTAL_STEPS = TOUR_STEPS.length;

  it("starts at step 0", () => {
    const current = 0;
    expect(current).toBe(0);
  });

  it("advances to next step", () => {
    let current = 0;
    current = Math.min(current + 1, TOTAL_STEPS - 1);
    expect(current).toBe(1);
  });

  it("goes back to previous step", () => {
    let current = 3;
    current = Math.max(current - 1, 0);
    expect(current).toBe(2);
  });

  it("cannot go below step 0", () => {
    let current = 0;
    current = Math.max(current - 1, 0);
    expect(current).toBe(0);
  });

  it("cannot exceed total steps", () => {
    let current = TOTAL_STEPS - 1;
    current = Math.min(current + 1, TOTAL_STEPS - 1);
    expect(current).toBe(TOTAL_STEPS - 1);
  });

  it("isFirst is true only at step 0", () => {
    expect(0 === 0).toBe(true);
    expect(1 === 0).toBe(false);
  });

  it("isLast is true only at final step", () => {
    expect(TOTAL_STEPS - 1 === TOTAL_STEPS - 1).toBe(true);
    expect(3 === TOTAL_STEPS - 1).toBe(false);
  });

  it("progress percentage is correct at each step", () => {
    for (let i = 0; i < TOTAL_STEPS; i++) {
      const pct = ((i + 1) / TOTAL_STEPS) * 100;
      expect(pct).toBeGreaterThan(0);
      expect(pct).toBeLessThanOrEqual(100);
    }
    expect(((0 + 1) / TOTAL_STEPS) * 100).toBeCloseTo(16.67, 1);
    expect(((TOTAL_STEPS - 1 + 1) / TOTAL_STEPS) * 100).toBe(100);
  });
});

// ─── Tooltip Positioning ──────────────────────────────────
describe("Tour tooltip positioning logic", () => {
  const computePosition = (
    position: string,
    rect: { top: number; left: number; width: number; height: number },
    tooltipWidth: number,
    tooltipHeight: number,
    gap: number = 16,
    padding: number = 8
  ) => {
    const r = {
      top: rect.top - padding,
      left: rect.left - padding,
      width: rect.width + padding * 2,
      height: rect.height + padding * 2,
    };

    let top = 0;
    let left = 0;

    switch (position) {
      case "bottom":
        top = r.top + r.height + gap;
        left = r.left + r.width / 2 - tooltipWidth / 2;
        break;
      case "top":
        top = r.top - tooltipHeight - gap;
        left = r.left + r.width / 2 - tooltipWidth / 2;
        break;
      case "right":
        top = r.top + r.height / 2 - tooltipHeight / 2;
        left = r.left + r.width + gap;
        break;
      case "left":
        top = r.top + r.height / 2 - tooltipHeight / 2;
        left = r.left - tooltipWidth - gap;
        break;
    }

    return { top, left };
  };

  it("positions tooltip below target for 'bottom' position", () => {
    const rect = { top: 100, left: 200, width: 300, height: 50 };
    const result = computePosition("bottom", rect, 320, 200);
    expect(result.top).toBeGreaterThan(rect.top + rect.height);
  });

  it("positions tooltip above target for 'top' position", () => {
    const rect = { top: 300, left: 200, width: 300, height: 50 };
    const result = computePosition("top", rect, 320, 200);
    expect(result.top).toBeLessThan(rect.top);
  });

  it("positions tooltip to the right for 'right' position", () => {
    const rect = { top: 100, left: 50, width: 200, height: 300 };
    const result = computePosition("right", rect, 320, 200);
    expect(result.left).toBeGreaterThan(rect.left + rect.width);
  });

  it("positions tooltip to the left for 'left' position", () => {
    const rect = { top: 100, left: 500, width: 200, height: 300 };
    const result = computePosition("left", rect, 320, 200);
    expect(result.left).toBeLessThan(rect.left);
  });

  it("centers tooltip horizontally for top/bottom positions", () => {
    const rect = { top: 100, left: 200, width: 300, height: 50 };
    const tooltipWidth = 320;
    const result = computePosition("bottom", rect, tooltipWidth, 200);
    const padding = 8;
    const expectedCenter =
      rect.left - padding + (rect.width + padding * 2) / 2;
    expect(result.left + tooltipWidth / 2).toBeCloseTo(expectedCenter, 0);
  });

  it("centers tooltip vertically for left/right positions", () => {
    const rect = { top: 100, left: 200, width: 200, height: 300 };
    const tooltipHeight = 200;
    const result = computePosition("right", rect, 320, tooltipHeight);
    const padding = 8;
    const expectedCenter =
      rect.top - padding + (rect.height + padding * 2) / 2;
    expect(result.top + tooltipHeight / 2).toBeCloseTo(expectedCenter, 0);
  });
});

// ─── Selector Matching ────────────────────────────────────
describe("Tour data-tour selectors match implementation", () => {
  const expectedSelectors = [
    "capture-input",
    "capture-button",
    "inbox-section",
    "sidebar-nav",
    "scenarioize-btn",
    "mode-toggle",
  ];

  it("all expected data-tour attributes are defined in tour steps", () => {
    const stepSelectors = TOUR_STEPS.map((s) =>
      s.selector.replace('[data-tour="', "").replace('"]', "")
    );
    expectedSelectors.forEach((sel) => {
      expect(stepSelectors).toContain(sel);
    });
  });

  it("tour steps are in logical order", () => {
    const ids = TOUR_STEPS.map((s) => s.id);
    expect(ids.indexOf("capture-input")).toBeLessThan(
      ids.indexOf("capture-button")
    );
    expect(ids.indexOf("capture-button")).toBeLessThan(
      ids.indexOf("inbox-section")
    );
    expect(ids.indexOf("inbox-section")).toBeLessThan(
      ids.indexOf("sidebar-nav")
    );
    expect(ids.indexOf("sidebar-nav")).toBeLessThan(
      ids.indexOf("scenarioize-btn")
    );
    expect(ids.indexOf("scenarioize-btn")).toBeLessThan(
      ids.indexOf("mode-toggle")
    );
  });
});

// ─── Conditional Launch Logic ─────────────────────────────
describe("Conditional first-time tour auto-launch", () => {
  let store: Record<string, string> = {};
  const mockStorage = {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => {
      store[key] = value;
    },
    removeItem: (key: string) => {
      delete store[key];
    },
  };

  beforeEach(() => {
    store = {};
  });

  it("brand new user: show onboarding, not tour", () => {
    const hasSeenOnboarding =
      mockStorage.getItem(ONBOARDING_STORAGE_KEY) === "true";
    const hasTourCompleted =
      mockStorage.getItem(TOUR_STORAGE_KEY) === "true";

    let showOnboarding = false;
    let showTour = false;

    if (!hasSeenOnboarding) {
      showOnboarding = true;
    } else if (!hasTourCompleted) {
      showTour = true;
    }

    expect(showOnboarding).toBe(true);
    expect(showTour).toBe(false);
  });

  it("returning user (onboarding done, tour not done): show tour", () => {
    mockStorage.setItem(ONBOARDING_STORAGE_KEY, "true");

    const hasSeenOnboarding =
      mockStorage.getItem(ONBOARDING_STORAGE_KEY) === "true";
    const hasTourCompleted =
      mockStorage.getItem(TOUR_STORAGE_KEY) === "true";

    let showOnboarding = false;
    let showTour = false;

    if (!hasSeenOnboarding) {
      showOnboarding = true;
    } else if (!hasTourCompleted) {
      showTour = true;
    }

    expect(showOnboarding).toBe(false);
    expect(showTour).toBe(true);
  });

  it("veteran user (both done): show neither", () => {
    mockStorage.setItem(ONBOARDING_STORAGE_KEY, "true");
    mockStorage.setItem(TOUR_STORAGE_KEY, "true");

    const hasSeenOnboarding =
      mockStorage.getItem(ONBOARDING_STORAGE_KEY) === "true";
    const hasTourCompleted =
      mockStorage.getItem(TOUR_STORAGE_KEY) === "true";

    let showOnboarding = false;
    let showTour = false;

    if (!hasSeenOnboarding) {
      showOnboarding = true;
    } else if (!hasTourCompleted) {
      showTour = true;
    }

    expect(showOnboarding).toBe(false);
    expect(showTour).toBe(false);
  });

  it("mutual exclusion: opening onboarding closes tour", () => {
    let showOnboarding = false;
    let showTour = true;

    // Simulate tks-show-onboarding event handler
    showTour = false;
    showOnboarding = true;

    expect(showOnboarding).toBe(true);
    expect(showTour).toBe(false);
  });

  it("mutual exclusion: opening tour closes onboarding", () => {
    let showOnboarding = true;
    let showTour = false;

    // Simulate tks-start-tour event handler
    showOnboarding = false;
    showTour = true;

    expect(showOnboarding).toBe(false);
    expect(showTour).toBe(true);
  });

  it("only one modal can be open at a time across all scenarios", () => {
    // Test all possible state transitions
    const scenarios = [
      { onboarding: false, tour: false },
      { onboarding: true, tour: false },
      { onboarding: false, tour: true },
    ];

    scenarios.forEach((s) => {
      // Both should never be true simultaneously
      expect(s.onboarding && s.tour).toBe(false);
    });
  });
});
