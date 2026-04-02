/**
 * Cross-platform notification service for TKS Tootra.
 * Works on:
 * - Web (Browser Notification API)
 * - Native Android/iOS (Capacitor Local Notifications)
 *
 * Handles permission requests, scheduling weekly review reminders,
 * and managing notification preferences via localStorage.
 */

// ─── Types ────────────────────────────────────────────────
export interface NotificationPreferences {
  enabled: boolean;
  weeklyReviewDay: number; // 0=Sunday, 1=Monday, ..., 6=Saturday
  weeklyReviewHour: number; // 0-23
  weeklyReviewMinute: number; // 0-59
  dailyDigest: boolean;
  dailyDigestHour: number;
}

const PREFS_KEY = "tks-notification-prefs";
const WEEKLY_REVIEW_NOTIFICATION_ID = 7001;
const DAILY_DIGEST_NOTIFICATION_ID = 7002;

const DEFAULT_PREFS: NotificationPreferences = {
  enabled: false,
  weeklyReviewDay: 0, // Sunday
  weeklyReviewHour: 18, // 6 PM
  weeklyReviewMinute: 0,
  dailyDigest: false,
  dailyDigestHour: 9, // 9 AM
};

// ─── Platform Detection ───────────────────────────────────
function isCapacitorNative(): boolean {
  return (
    typeof window !== "undefined" &&
    !!(window as any).Capacitor?.isNativePlatform?.()
  );
}

// ─── Preferences ──────────────────────────────────────────
export function getNotificationPrefs(): NotificationPreferences {
  try {
    const stored = localStorage.getItem(PREFS_KEY);
    if (stored) {
      return { ...DEFAULT_PREFS, ...JSON.parse(stored) };
    }
  } catch {
    // ignore parse errors
  }
  return { ...DEFAULT_PREFS };
}

export function saveNotificationPrefs(prefs: NotificationPreferences): void {
  localStorage.setItem(PREFS_KEY, JSON.stringify(prefs));
}

// ─── Permission ───────────────────────────────────────────
export async function requestNotificationPermission(): Promise<boolean> {
  if (isCapacitorNative()) {
    try {
      const { LocalNotifications } = await import(
        "@capacitor/local-notifications"
      );
      const result = await LocalNotifications.requestPermissions();
      return result.display === "granted";
    } catch {
      return false;
    }
  }

  // Web fallback
  if (!("Notification" in window)) return false;
  if (Notification.permission === "granted") return true;
  if (Notification.permission === "denied") return false;

  const result = await Notification.requestPermission();
  return result === "granted";
}

export function getNotificationPermissionStatus(): string {
  if (isCapacitorNative()) {
    return "unknown"; // Must be checked async on native
  }
  if (!("Notification" in window)) return "unsupported";
  return Notification.permission;
}

// ─── Schedule Weekly Review Reminder ──────────────────────
export async function scheduleWeeklyReviewReminder(
  prefs: NotificationPreferences
): Promise<boolean> {
  if (!prefs.enabled) {
    await cancelWeeklyReviewReminder();
    return false;
  }

  if (isCapacitorNative()) {
    try {
      const { LocalNotifications } = await import(
        "@capacitor/local-notifications"
      );

      // Cancel existing
      await LocalNotifications.cancel({
        notifications: [{ id: WEEKLY_REVIEW_NOTIFICATION_ID }],
      });

      // Calculate next occurrence
      const now = new Date();
      const nextDate = getNextWeekday(
        prefs.weeklyReviewDay,
        prefs.weeklyReviewHour,
        prefs.weeklyReviewMinute
      );

      await LocalNotifications.schedule({
        notifications: [
          {
            id: WEEKLY_REVIEW_NOTIFICATION_ID,
            title: "Weekly Review Time",
            body: "Take 10 minutes to review your week. Check your Foundation balance, celebrate wins, and plan ahead.",
            schedule: {
              at: nextDate,
              repeats: true,
              every: "week",
            },
            actionTypeId: "WEEKLY_REVIEW",
            extra: { route: "/review" },
          },
        ],
      });
      return true;
    } catch (err) {
      console.error("Failed to schedule native notification:", err);
      return false;
    }
  }

  // Web: Use service worker scheduled notification (if supported)
  // For now, store the preference and check on app load
  return true;
}

export async function scheduleDailyDigest(
  prefs: NotificationPreferences
): Promise<boolean> {
  if (!prefs.dailyDigest) {
    await cancelDailyDigest();
    return false;
  }

  if (isCapacitorNative()) {
    try {
      const { LocalNotifications } = await import(
        "@capacitor/local-notifications"
      );

      await LocalNotifications.cancel({
        notifications: [{ id: DAILY_DIGEST_NOTIFICATION_ID }],
      });

      const nextDate = getNextTimeToday(prefs.dailyDigestHour, 0);

      await LocalNotifications.schedule({
        notifications: [
          {
            id: DAILY_DIGEST_NOTIFICATION_ID,
            title: "Your Daily TKS Digest",
            body: "You have tasks waiting. Check your inbox and keep building momentum.",
            schedule: {
              at: nextDate,
              repeats: true,
              every: "day",
            },
            actionTypeId: "DAILY_DIGEST",
            extra: { route: "/" },
          },
        ],
      });
      return true;
    } catch (err) {
      console.error("Failed to schedule daily digest:", err);
      return false;
    }
  }

  return true;
}

// ─── Cancel ───────────────────────────────────────────────
export async function cancelWeeklyReviewReminder(): Promise<void> {
  if (isCapacitorNative()) {
    try {
      const { LocalNotifications } = await import(
        "@capacitor/local-notifications"
      );
      await LocalNotifications.cancel({
        notifications: [{ id: WEEKLY_REVIEW_NOTIFICATION_ID }],
      });
    } catch {
      // ignore
    }
  }
}

export async function cancelDailyDigest(): Promise<void> {
  if (isCapacitorNative()) {
    try {
      const { LocalNotifications } = await import(
        "@capacitor/local-notifications"
      );
      await LocalNotifications.cancel({
        notifications: [{ id: DAILY_DIGEST_NOTIFICATION_ID }],
      });
    } catch {
      // ignore
    }
  }
}

// ─── Web Notification (immediate) ─────────────────────────
export function showWebNotification(
  title: string,
  body: string,
  route?: string
): void {
  if (!("Notification" in window) || Notification.permission !== "granted")
    return;

  const notification = new Notification(title, {
    body,
    icon: "/favicon.ico",
    badge: "/favicon.ico",
    tag: "tks-tootra",
  });

  if (route) {
    notification.onclick = () => {
      window.focus();
      window.location.hash = route;
      notification.close();
    };
  }
}

// ─── Check & Trigger Web Reminders on App Load ────────────
export function checkWebReminders(): void {
  const prefs = getNotificationPrefs();
  if (!prefs.enabled) return;

  const now = new Date();
  const lastCheck = localStorage.getItem("tks-last-reminder-check");
  const lastCheckDate = lastCheck ? new Date(lastCheck) : null;

  // Weekly review check
  if (
    now.getDay() === prefs.weeklyReviewDay &&
    now.getHours() >= prefs.weeklyReviewHour
  ) {
    const todayKey = `${now.getFullYear()}-${now.getMonth()}-${now.getDate()}-weekly`;
    const alreadySent = localStorage.getItem("tks-reminder-sent") === todayKey;

    if (!alreadySent) {
      showWebNotification(
        "Weekly Review Time",
        "Take 10 minutes to review your week. Check your Foundation balance and plan ahead.",
        "/review"
      );
      localStorage.setItem("tks-reminder-sent", todayKey);
    }
  }

  localStorage.setItem("tks-last-reminder-check", now.toISOString());
}

// ─── Helpers ──────────────────────────────────────────────
function getNextWeekday(
  targetDay: number,
  hour: number,
  minute: number
): Date {
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
