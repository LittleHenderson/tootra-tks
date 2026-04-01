import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import NotFound from "@/pages/NotFound";
import { Route, Switch } from "wouter";
import ErrorBoundary from "./components/ErrorBoundary";
import { ThemeProvider } from "./contexts/ThemeContext";
import DashboardLayout from "./components/DashboardLayout";
import OnboardingSlideshow, { hasSeenOnboarding } from "./components/OnboardingSlideshow";
import PWAInstallPrompt from "./components/PWAInstallPrompt";
import Home from "./pages/Home";
import Tasks from "./pages/Tasks";
import Equations from "./pages/Equations";
import Heatmap from "./pages/Heatmap";
import PowerGaps from "./pages/PowerGaps";
import WeeklyReview from "./pages/WeeklyReview";
import DWPTracker from "./pages/DWPTracker";
import TaskDetail from "./pages/TaskDetail";
import { useState, useEffect } from "react";

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
        <Route path="/404" component={NotFound} />
        <Route component={NotFound} />
      </Switch>
    </DashboardLayout>
  );
}

function App() {
  const [showOnboarding, setShowOnboarding] = useState(false);

  // Auto-show on first visit
  useEffect(() => {
    if (!hasSeenOnboarding()) {
      setShowOnboarding(true);
    }
  }, []);

  // Listen for sidebar "How to Use" relaunch from any page
  useEffect(() => {
    const handler = () => setShowOnboarding(true);
    window.addEventListener("tks-show-onboarding", handler);
    return () => window.removeEventListener("tks-show-onboarding", handler);
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
          <Router />
          <PWAInstallPrompt />
        </TooltipProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;
