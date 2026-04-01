import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import {
  Bell,
  BellOff,
  Calendar,
  Clock,
  CheckCircle2,
  AlertTriangle,
  Smartphone,
  Globe,
  ArrowLeft,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { toast } from "sonner";
import { useLocation } from "wouter";
import {
  getNotificationPrefs,
  saveNotificationPrefs,
  requestNotificationPermission,
  getNotificationPermissionStatus,
  scheduleWeeklyReviewReminder,
  scheduleDailyDigest,
  type NotificationPreferences,
} from "@/lib/notifications";

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

export default function NotificationSettings() {
  const [, navigate] = useLocation();
  const [prefs, setPrefs] = useState<NotificationPreferences>(getNotificationPrefs());
  const [permissionStatus, setPermissionStatus] = useState<string>("unknown");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setPermissionStatus(getNotificationPermissionStatus());
  }, []);

  const handleToggleNotifications = async (enabled: boolean) => {
    if (enabled) {
      const granted = await requestNotificationPermission();
      if (!granted) {
        toast.error("Notification permission denied. Please enable notifications in your browser/device settings.");
        return;
      }
      setPermissionStatus("granted");
    }

    const newPrefs = { ...prefs, enabled };
    setPrefs(newPrefs);
    saveNotificationPrefs(newPrefs);

    if (enabled) {
      await scheduleWeeklyReviewReminder(newPrefs);
      if (newPrefs.dailyDigest) {
        await scheduleDailyDigest(newPrefs);
      }
      toast.success("Notifications enabled! You'll receive weekly review reminders.");
    } else {
      await scheduleWeeklyReviewReminder(newPrefs); // This cancels when disabled
      await scheduleDailyDigest(newPrefs);
      toast.info("Notifications disabled.");
    }
  };

  const handlePrefsChange = async (updates: Partial<NotificationPreferences>) => {
    const newPrefs = { ...prefs, ...updates };
    setPrefs(newPrefs);
    saveNotificationPrefs(newPrefs);

    if (newPrefs.enabled) {
      setSaving(true);
      await scheduleWeeklyReviewReminder(newPrefs);
      if (newPrefs.dailyDigest) {
        await scheduleDailyDigest(newPrefs);
      }
      setSaving(false);
      toast.success("Notification schedule updated.");
    }
  };

  const isNative = typeof window !== "undefined" && !!(window as any).Capacitor?.isNativePlatform?.();

  return (
    <div className="max-w-2xl mx-auto px-4 py-8 space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Button
          variant="ghost"
          size="icon"
          onClick={() => navigate("/")}
          className="shrink-0"
        >
          <ArrowLeft className="h-4 w-4" />
        </Button>
        <div>
          <h1 className="text-xl font-bold text-foreground">Notification Settings</h1>
          <p className="text-sm text-muted-foreground">
            Configure reminders to stay on track with your TKS practice
          </p>
        </div>
      </div>

      {/* Platform Info */}
      <Card className="border-border/50">
        <CardContent className="pt-6">
          <div className="flex items-center gap-3 text-sm">
            {isNative ? (
              <>
                <Smartphone className="h-4 w-4 text-tks-gold" />
                <span className="text-muted-foreground">
                  Native notifications — reminders will fire even when the app is closed
                </span>
              </>
            ) : (
              <>
                <Globe className="h-4 w-4 text-tks-blue" />
                <span className="text-muted-foreground">
                  Web notifications — reminders work when the browser is open
                </span>
              </>
            )}
          </div>
          {permissionStatus === "denied" && (
            <div className="mt-3 flex items-center gap-2 text-xs text-destructive">
              <AlertTriangle className="h-3.5 w-3.5" />
              <span>
                Notifications are blocked. Please enable them in your browser settings.
              </span>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Master Toggle */}
      <Card className="border-border/50">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {prefs.enabled ? (
                <Bell className="h-5 w-5 text-tks-gold" />
              ) : (
                <BellOff className="h-5 w-5 text-muted-foreground" />
              )}
              <div>
                <CardTitle className="text-base">Enable Notifications</CardTitle>
                <CardDescription>
                  Receive reminders for weekly reviews and daily check-ins
                </CardDescription>
              </div>
            </div>
            <Switch
              checked={prefs.enabled}
              onCheckedChange={handleToggleNotifications}
            />
          </div>
        </CardHeader>
      </Card>

      {/* Weekly Review Reminder */}
      <motion.div
        initial={false}
        animate={{ opacity: prefs.enabled ? 1 : 0.4 }}
      >
        <Card className="border-border/50">
          <CardHeader>
            <div className="flex items-center gap-3">
              <Calendar className="h-5 w-5 text-tks-gold" />
              <div>
                <CardTitle className="text-base">Weekly Review Reminder</CardTitle>
                <CardDescription>
                  A nudge to reflect on your week and rebalance your Foundations
                </CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-xs font-medium text-muted-foreground">Day</label>
                <Select
                  value={String(prefs.weeklyReviewDay)}
                  onValueChange={(v) =>
                    handlePrefsChange({ weeklyReviewDay: Number(v) })
                  }
                  disabled={!prefs.enabled}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {DAYS.map((d) => (
                      <SelectItem key={d.value} value={d.value}>
                        {d.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <label className="text-xs font-medium text-muted-foreground">Time</label>
                <Select
                  value={String(prefs.weeklyReviewHour)}
                  onValueChange={(v) =>
                    handlePrefsChange({ weeklyReviewHour: Number(v) })
                  }
                  disabled={!prefs.enabled}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {HOURS.map((h) => (
                      <SelectItem key={h.value} value={h.value}>
                        {h.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
            {prefs.enabled && (
              <motion.p
                initial={{ opacity: 0, y: -4 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-xs text-muted-foreground flex items-center gap-1.5"
              >
                <CheckCircle2 className="h-3 w-3 text-green-400" />
                Reminder set for every{" "}
                {DAYS.find((d) => d.value === String(prefs.weeklyReviewDay))?.label} at{" "}
                {HOURS.find((h) => h.value === String(prefs.weeklyReviewHour))?.label}
              </motion.p>
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* Daily Digest */}
      <motion.div
        initial={false}
        animate={{ opacity: prefs.enabled ? 1 : 0.4 }}
      >
        <Card className="border-border/50">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Clock className="h-5 w-5 text-tks-blue" />
                <div>
                  <CardTitle className="text-base">Daily Digest</CardTitle>
                  <CardDescription>
                    A morning reminder to check your inbox and keep momentum
                  </CardDescription>
                </div>
              </div>
              <Switch
                checked={prefs.dailyDigest}
                onCheckedChange={(v) => handlePrefsChange({ dailyDigest: v })}
                disabled={!prefs.enabled}
              />
            </div>
          </CardHeader>
          {prefs.dailyDigest && prefs.enabled && (
            <CardContent>
              <div className="space-y-2">
                <label className="text-xs font-medium text-muted-foreground">Time</label>
                <Select
                  value={String(prefs.dailyDigestHour)}
                  onValueChange={(v) =>
                    handlePrefsChange({ dailyDigestHour: Number(v) })
                  }
                  disabled={!prefs.enabled}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {HOURS.map((h) => (
                      <SelectItem key={h.value} value={h.value}>
                        {h.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              {prefs.enabled && prefs.dailyDigest && (
                <motion.p
                  initial={{ opacity: 0, y: -4 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-3 text-xs text-muted-foreground flex items-center gap-1.5"
                >
                  <CheckCircle2 className="h-3 w-3 text-green-400" />
                  Daily digest at{" "}
                  {HOURS.find((h) => h.value === String(prefs.dailyDigestHour))?.label}
                </motion.p>
              )}
            </CardContent>
          )}
        </Card>
      </motion.div>

      {/* Saving indicator */}
      {saving && (
        <p className="text-xs text-center text-muted-foreground animate-pulse">
          Updating notification schedule...
        </p>
      )}
    </div>
  );
}
