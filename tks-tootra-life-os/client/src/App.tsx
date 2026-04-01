import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import NotFound from "@/pages/NotFound";
import { Route, Switch } from "wouter";
import ErrorBoundary from "./components/ErrorBoundary";
import { ThemeProvider } from "./contexts/ThemeContext";
import DashboardLayout from "./components/DashboardLayout";
import OnboardingSlideshow, { hasSeenOnboarding } from "./components/OnboardingSlideshow";
import InteractiveTour, { hasTourCompleted } from "./components/InteractiveTour";
import PWAInstallPrompt from "./components/PWAInstallPrompt";
import Home from "./pages/Home";
import Tasks from "./pages/Tasks";
import Equations from "./pages/Equations";
import Heatmap from "./pages/Heatmap";
import PowerGaps from "./pages/PowerGaps";
import WeeklyReview from "./pages/WeeklyReview";
import DWPTracker from "./pages/DWPTracker";
import TaskDetail from "./pages/TaskDetail";
import NotificationSettings from "./pages/NotificationSettings";
import { useState, useEffect } from "react";
import { checkWebReminders } from "@/lib/notifications";

function Router() {
  return (
    <DashboardLayout>
      <Switch>
        <Route path="/" component={Home} />
        <Route path="/tasks" component={Tasks} />
        <Route path="/tasks/:id" component={TaskDetail} />
        <Route path="/equations" component={Equations} />
        <Route path="/heatmap" component={Heatmap} />
        <Route path="/power" component={PowerGaps} />
        <Route path="/review" component={WeeklyReview} />
        <Route path="/dwp" component={DWPTracker} />
        <Route path="/notifications" component={NotificationSettings} />
        <Route path="/404" component={NotFound} />
        <Route component={NotFound} />
      </Switch>
    </DashboardLayout>
  );
}

function App() {
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [showTour, setShowTour] = useState(false);

  // Check web reminders on app load
  useEffect(() => {
    checkWebReminders();
  }, []);

  // Auto-show on first visit: onboarding for brand-new users, tour for returning users
  useEffect(() => {
    if (!hasSeenOnboarding()) {
      // Brand new user — show the onboarding slideshow
      setShowOnboarding(true);
    } else if (!hasTourCompleted()) {
      // Returning user who finished onboarding but hasn't done the tour yet
      // Small delay so the dashboard renders first
      const timer = setTimeout(() => setShowTour(true), 600);
      return () => clearTimeout(timer);
    }
  }, []);

  // Listen for sidebar "How to Use" relaunch from any page
  // Mutual exclusion: close tour when opening onboarding
  useEffect(() => {
    const handler = () => {
      setShowTour(false);
      setShowOnboarding(true);
    };
    window.addEventListener("tks-show-onboarding", handler);
    return () => window.removeEventListener("tks-show-onboarding", handler);
  }, []);

  // Listen for "Take a Tour" from sidebar or onboarding
  // Mutual exclusion: close onboarding when opening tour
  useEffect(() => {
    const handler = () => {
      setShowOnboarding(false);
      setShowTour(true);
    };
    window.addEventListener("tks-start-tour", handler);
    return () => window.removeEventListener("tks-start-tour", handler);
  }, []);

  return (
    <ErrorBoundary>
      <ThemeProvider defaultTheme="dark">
        <TooltipProvider>
          <Toaster />
          <OnboardingSlideshow
            open={showOnboarding}
            onClose={() => setShowOnboarding(false)}
          />
          <InteractiveTour
            open={showTour}
            onClose={() => setShowTour(false)}
          />
          <Router />
          <PWAInstallPrompt />
        </TooltipProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;
