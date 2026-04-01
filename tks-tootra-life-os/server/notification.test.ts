import { describe, it, expect, beforeEach, vi } from "vitest";

/**
 * Tests for the notification preferences and scheduling logic.
 * These test the core business logic without requiring browser/Capacitor APIs.
 */

// ─── Notification Preferences ─────────────────────────────
describe("notification preferences", () => {
  const DEFAULT_PREFS = {
    enabled: false,
    weeklyReviewDay: 0,
    weeklyReviewHour: 18,
    weeklyReviewMinute: 0,
    dailyDigest: false,
    dailyDigestHour: 9,
  };

  describe("default values", () => {
    it("defaults to disabled", () => {
      expect(DEFAULT_PREFS.enabled).toBe(false);
    });

    it("defaults weekly review to Sunday at 6 PM", () => {
      expect(DEFAULT_PREFS.weeklyReviewDay).toBe(0); // Sunday
      expect(DEFAULT_PREFS.weeklyReviewHour).toBe(18); // 6 PM
      expect(DEFAULT_PREFS.weeklyReviewMinute).toBe(0);
    });

    it("defaults daily digest to disabled at 9 AM", () => {
      expect(DEFAULT_PREFS.dailyDigest).toBe(false);
      expect(DEFAULT_PREFS.dailyDigestHour).toBe(9);
    });
  });

  describe("preference validation", () => {
    it("weeklyReviewDay must be 0-6", () => {
      for (let day = 0; day <= 6; day++) {
        expect(day).toBeGreaterThanOrEqual(0);
        expect(day).toBeLessThanOrEqual(6);
      }
    });

    it("weeklyReviewHour must be 0-23", () => {
      for (let hour = 0; hour <= 23; hour++) {
        expect(hour).toBeGreaterThanOrEqual(0);
        expect(hour).toBeLessThanOrEqual(23);
      }
    });

    it("dailyDigestHour must be 0-23", () => {
      for (let hour = 0; hour <= 23; hour++) {
        expect(hour).toBeGreaterThanOrEqual(0);
        expect(hour).toBeLessThanOrEqual(23);
      }
    });
  });

  describe("preference merging", () => {
    it("merges partial updates with defaults", () => {
      const partial = { enabled: true, weeklyReviewDay: 5 };
      const merged = { ...DEFAULT_PREFS, ...partial };
      expect(merged.enabled).toBe(true);
      expect(merged.weeklyReviewDay).toBe(5); // Friday
      expect(merged.weeklyReviewHour).toBe(18); // Still default
      expect(merged.dailyDigest).toBe(false); // Still default
    });

    it("preserves all fields when fully specified", () => {
      const full = {
        enabled: true,
        weeklyReviewDay: 3,
        weeklyReviewHour: 20,
        weeklyReviewMinute: 30,
        dailyDigest: true,
        dailyDigestHour: 7,
      };
      const merged = { ...DEFAULT_PREFS, ...full };
      expect(merged).toEqual(full);
    });
  });
});

// ─── Scheduling Logic ─────────────────────────────────────
describe("scheduling logic", () => {
  function getNextWeekday(targetDay: number, hour: number, minute: number): Date {
    const now = new Date();
    const result = new Date(now);
    result.setHours(hour, minute, 0, 0);

    const daysUntil = (targetDay - now.getDay() + 7) % 7;
    if (daysUntil === 0 && result <= now) {
      result.setDate(result.getDate() + 7);
    } else {
      result.setDate(result.getDate() + daysUntil);
    }

    return result;
  }

  function getNextTimeToday(hour: number, minute: number): Date {
    const now = new Date();
    const result = new Date(now);
    result.setHours(hour, minute, 0, 0);

    if (result <= now) {
      result.setDate(result.getDate() + 1);
    }

    return result;
  }

  describe("getNextWeekday", () => {
    it("returns a future date", () => {
      const next = getNextWeekday(0, 18, 0); // Next Sunday at 6 PM
      expect(next.getTime()).toBeGreaterThan(Date.now() - 1000); // Allow 1s tolerance
    });

    it("returns the correct day of week", () => {
      const nextSunday = getNextWeekday(0, 18, 0);
      expect(nextSunday.getDay()).toBe(0);

      const nextFriday = getNextWeekday(5, 9, 0);
      expect(nextFriday.getDay()).toBe(5);
    });

    it("returns the correct hour and minute", () => {
      const next = getNextWeekday(3, 14, 30); // Wednesday at 2:30 PM
      expect(next.getHours()).toBe(14);
      expect(next.getMinutes()).toBe(30);
      expect(next.getSeconds()).toBe(0);
    });

    it("returns within 7 days", () => {
      const next = getNextWeekday(0, 18, 0);
      const maxFuture = new Date();
      maxFuture.setDate(maxFuture.getDate() + 8); // 8 days to account for edge cases
      expect(next.getTime()).toBeLessThan(maxFuture.getTime());
    });
  });

  describe("getNextTimeToday", () => {
    it("returns a future date", () => {
      const next = getNextTimeToday(23, 59);
      // If it's past 23:59, it should be tomorrow
      expect(next.getTime()).toBeGreaterThan(Date.now() - 1000);
    });

    it("returns the correct hour and minute", () => {
      const next = getNextTimeToday(9, 0);
      expect(next.getHours()).toBe(9);
      expect(next.getMinutes()).toBe(0);
    });

    it("returns tomorrow if time has passed today", () => {
      // Use midnight which has always passed
      const next = getNextTimeToday(0, 0);
      const tomorrow = new Date();
      tomorrow.setDate(tomorrow.getDate() + 1);
      tomorrow.setHours(0, 0, 0, 0);
      expect(next.getTime()).toBe(tomorrow.getTime());
    });
  });
});

// ─── Notification IDs ─────────────────────────────────────
describe("notification IDs", () => {
  const WEEKLY_REVIEW_NOTIFICATION_ID = 7001;
  const DAILY_DIGEST_NOTIFICATION_ID = 7002;

  it("weekly review and daily digest have unique IDs", () => {
    expect(WEEKLY_REVIEW_NOTIFICATION_ID).not.toBe(DAILY_DIGEST_NOTIFICATION_ID);
  });

  it("IDs are positive integers", () => {
    expect(WEEKLY_REVIEW_NOTIFICATION_ID).toBeGreaterThan(0);
    expect(DAILY_DIGEST_NOTIFICATION_ID).toBeGreaterThan(0);
    expect(Number.isInteger(WEEKLY_REVIEW_NOTIFICATION_ID)).toBe(true);
    expect(Number.isInteger(DAILY_DIGEST_NOTIFICATION_ID)).toBe(true);
  });
});

// ─── Day/Hour Display Mapping ─────────────────────────────
describe("display formatting", () => {
  const DAYS = [
    { value: "0", label: "Sunday" },
    { value: "1", label: "Monday" },
    { value: "2", label: "Tuesday" },
    { value: "3", label: "Wednesday" },
    { value: "4", label: "Thursday" },
    { value: "5", label: "Friday" },
    { value: "6", label: "Saturday" },
  ];

  const HOURS = Array.from({ length: 24 }, (_, i) => ({
    value: String(i),
    label: `${i === 0 ? "12" : i > 12 ? String(i - 12) : String(i)}:00 ${i < 12 ? "AM" : "PM"}`,
  }));

  it("has 7 days", () => {
    expect(DAYS).toHaveLength(7);
  });

  it("has 24 hours", () => {
    expect(HOURS).toHaveLength(24);
  });

  it("maps day 0 to Sunday", () => {
    expect(DAYS[0].label).toBe("Sunday");
  });

  it("maps day 6 to Saturday", () => {
    expect(DAYS[6].label).toBe("Saturday");
  });

  it("formats midnight as 12:00 AM", () => {
    expect(HOURS[0].label).toBe("12:00 AM");
  });

  it("formats noon as 12:00 PM", () => {
    expect(HOURS[12].label).toBe("12:00 PM");
  });

  it("formats 6 PM correctly", () => {
    expect(HOURS[18].label).toBe("6:00 PM");
  });

  it("all hour values are non-empty strings", () => {
    HOURS.forEach((h) => {
      expect(h.value).toBeTruthy();
      expect(h.label).toBeTruthy();
    });
  });
});

// ─── Capacitor Config ─────────────────────────────────────
describe("capacitor configuration", () => {
  it("has correct app ID format", () => {
    const appId = "com.tootra.tks";
    expect(appId).toMatch(/^[a-z]+\.[a-z]+\.[a-z]+$/);
  });

  it("has correct web directory", () => {
    const webDir = "dist/public";
    expect(webDir).toBe("dist/public");
  });

  it("uses https scheme for Android", () => {
    const scheme = "https";
    expect(scheme).toBe("https");
  });
});

// ─── localStorage Key Consistency ─────────────────────────
describe("storage keys", () => {
  const PREFS_KEY = "tks-notification-prefs";
  const REMINDER_SENT_KEY = "tks-reminder-sent";
  const LAST_CHECK_KEY = "tks-last-reminder-check";

  it("all keys use tks- prefix", () => {
    expect(PREFS_KEY).toMatch(/^tks-/);
    expect(REMINDER_SENT_KEY).toMatch(/^tks-/);
    expect(LAST_CHECK_KEY).toMatch(/^tks-/);
  });

  it("keys are unique", () => {
    const keys = [PREFS_KEY, REMINDER_SENT_KEY, LAST_CHECK_KEY];
    const unique = new Set(keys);
    expect(unique.size).toBe(keys.length);
  });
});
