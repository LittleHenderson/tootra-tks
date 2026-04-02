import { useState, useEffect, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  X,
  ArrowRight,
  ArrowLeft,
  Send,
  Inbox,
  Grid3X3,
  ImagePlay,
  Flame,
  Triangle,
  Sparkles,
  CheckCircle2,
  MousePointerClick,
  Type,
} from "lucide-react";
import { Button } from "@/components/ui/button";

// ─── Storage ──────────────────────────────────────────────
const TOUR_STORAGE_KEY = "tks-tour-completed";

export function hasTourCompleted(): boolean {
  return localStorage.getItem(TOUR_STORAGE_KEY) === "true";
}

export function resetTour(): void {
  localStorage.removeItem(TOUR_STORAGE_KEY);
}

// ─── Tour Step Definition ─────────────────────────────────
interface TourStep {
  id: string;
  selector: string;
  title: string;
  description: string;
  icon: React.ReactNode;
  position: "top" | "bottom" | "left" | "right";
  navigateTo?: string;
  accent: string;
  /** Describes the action the user should take to auto-advance */
  actionHint: string;
  /** The DOM event to listen for on the target element */
  actionEvent: string;
  /** Whether to allow interaction with the highlighted element */
  allowInteraction: boolean;
}

const TOUR_STEPS: TourStep[] = [
  {
    id: "capture-input",
    selector: '[data-tour="capture-input"]',
    title: "Capture Your Intention",
    description:
      "Type any task, goal, or thought here. Be natural — say it like you would to a friend. The AI handles the rest.",
    icon: <Send className="h-5 w-5" />,
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
    description:
      "Click this button (or press Ctrl+Enter) and the AI will instantly map your task to a Foundation, generate a TKS Equation, and diagnose the D/W/P lifecycle.",
    icon: <Sparkles className="h-5 w-5" />,
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
    description:
      "Captured tasks land here first. Each card shows the Foundation, archetype, and D/W/P status at a glance. Click any task to see its full analysis.",
    icon: <Inbox className="h-5 w-5" />,
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
    description:
      "The sidebar gives you access to all TKS features — Tasks, Equations, Heatmap, Power Gaps, Weekly Review, and D/W/P Tracker.",
    icon: <Grid3X3 className="h-5 w-5" />,
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
    description:
      "Upload a photo and the AI generates a realistic vision of what happens if you follow through — or what happens if you don't. Available on any task.",
    icon: <ImagePlay className="h-5 w-5" />,
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
    description:
      "Toggle between full Life OS mode (all 7 Foundations) and focused Work Sprint mode (just Inbox + Tasks). Use Sprint when you need to lock in.",
    icon: <Flame className="h-5 w-5" />,
    position: "right",
    accent: "tks-flame",
    actionHint: "Click to toggle!",
    actionEvent: "click",
    allowInteraction: true,
  },
];

// ─── Spotlight Overlay ────────────────────────────────────
interface SpotlightRect {
  top: number;
  left: number;
  width: number;
  height: number;
}

function SpotlightOverlay({
  rect,
  onClickOutside,
  allowInteraction,
}: {
  rect: SpotlightRect | null;
  onClickOutside: () => void;
  allowInteraction: boolean;
}) {
  if (!rect) return null;

  const padding = 8;
  const r = {
    top: rect.top - padding,
    left: rect.left - padding,
    width: rect.width + padding * 2,
    height: rect.height + padding * 2,
  };

  return (
    <>
      {/* Dark overlay with cutout */}
      <div
        className="fixed inset-0 z-[9997]"
        onClick={onClickOutside}
        style={{ pointerEvents: "auto" }}
      >
        <svg
          className="absolute inset-0 w-full h-full"
          xmlns="http://www.w3.org/2000/svg"
        >
          <defs>
            <mask id="tour-spotlight-mask">
              <rect x="0" y="0" width="100%" height="100%" fill="white" />
              <rect
                x={r.left}
                y={r.top}
                width={r.width}
                height={r.height}
                rx="12"
                ry="12"
                fill="black"
              />
            </mask>
          </defs>
          <rect
            x="0"
            y="0"
            width="100%"
            height="100%"
            fill="rgba(0,0,0,0.65)"
            mask="url(#tour-spotlight-mask)"
          />
          {/* Animated glow ring */}
          <rect
            x={r.left - 2}
            y={r.top - 2}
            width={r.width + 4}
            height={r.height + 4}
            rx="14"
            ry="14"
            fill="none"
            stroke="oklch(0.795 0.184 86.05)"
            strokeWidth="2"
            opacity="0.6"
          >
            <animate
              attributeName="opacity"
              values="0.6;0.2;0.6"
              dur="2s"
              repeatCount="indefinite"
            />
          </rect>
        </svg>
      </div>

      {/* Transparent interaction window over the highlighted element */}
      {allowInteraction && (
        <div
          className="fixed z-[9998]"
          style={{
            top: r.top,
            left: r.left,
            width: r.width,
            height: r.height,
            pointerEvents: "none",
          }}
        />
      )}
    </>
  );
}

// ─── Tooltip ──────────────────────────────────────────────
interface TooltipProps {
  step: TourStep;
  rect: SpotlightRect;
  currentIndex: number;
  totalSteps: number;
  onNext: () => void;
  onPrev: () => void;
  onSkip: () => void;
  actionTriggered: boolean;
}

function TourTooltip({
  step,
  rect,
  currentIndex,
  totalSteps,
  onNext,
  onPrev,
  onSkip,
  actionTriggered,
}: TooltipProps) {
  const tooltipRef = useRef<HTMLDivElement>(null);
  const [tooltipStyle, setTooltipStyle] = useState<React.CSSProperties>({});

  useEffect(() => {
    const el = tooltipRef.current;
    if (!el) return;

    const padding = 8;
    const gap = 16;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const tw = el.offsetWidth;
    const th = el.offsetHeight;

    let top = 0;
    let left = 0;

    const r = {
      top: rect.top - padding,
      left: rect.left - padding,
      width: rect.width + padding * 2,
      height: rect.height + padding * 2,
    };

    switch (step.position) {
      case "bottom":
        top = r.top + r.height + gap;
        left = r.left + r.width / 2 - tw / 2;
        break;
      case "top":
        top = r.top - th - gap;
        left = r.left + r.width / 2 - tw / 2;
        break;
      case "right":
        top = r.top + r.height / 2 - th / 2;
        left = r.left + r.width + gap;
        break;
      case "left":
        top = r.top + r.height / 2 - th / 2;
        left = r.left - tw - gap;
        break;
    }

    // Clamp to viewport
    if (left < 12) left = 12;
    if (left + tw > vw - 12) left = vw - tw - 12;
    if (top < 12) top = 12;
    if (top + th > vh - 12) top = vh - th - 12;

    setTooltipStyle({ top, left });
  }, [rect, step.position]);

  const isFirst = currentIndex === 0;
  const isLast = currentIndex === totalSteps - 1;

  return (
    <motion.div
      ref={tooltipRef}
      initial={{ opacity: 0, scale: 0.9, y: 8 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.9, y: 8 }}
      transition={{ duration: 0.25, ease: "easeOut" }}
      className="fixed z-[9999] w-80 bg-card border border-border rounded-xl shadow-2xl overflow-hidden"
      style={{ ...tooltipStyle, pointerEvents: "auto" }}
      onClick={(e) => e.stopPropagation()}
    >
      {/* Header */}
      <div className="px-4 pt-4 pb-2 flex items-start gap-3">
        <div className={`p-2 rounded-lg bg-${step.accent}/10 text-${step.accent} shrink-0`}>
          {step.icon}
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-bold text-foreground">{step.title}</h3>
          <p className="text-xs text-muted-foreground mt-0.5">
            Step {currentIndex + 1} of {totalSteps}
          </p>
        </div>
        <button
          onClick={onSkip}
          className="p-1 rounded-full hover:bg-secondary/80 text-muted-foreground hover:text-foreground transition-colors shrink-0"
          aria-label="End tour"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>

      {/* Body */}
      <div className="px-4 pb-2">
        <p className="text-xs text-muted-foreground leading-relaxed">
          {step.description}
        </p>
      </div>

      {/* Action Hint */}
      <div className="px-4 pb-3">
        <AnimatePresence mode="wait">
          {actionTriggered ? (
            <motion.div
              key="done"
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="flex items-center gap-1.5 text-xs font-medium text-green-400"
            >
              <CheckCircle2 className="h-3.5 w-3.5" />
              <span>Nice! Moving on...</span>
            </motion.div>
          ) : (
            <motion.div
              key="hint"
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="flex items-center gap-1.5"
            >
              {step.actionEvent === "input" ? (
                <Type className="h-3.5 w-3.5 text-tks-gold animate-pulse" />
              ) : (
                <MousePointerClick className="h-3.5 w-3.5 text-tks-gold animate-pulse" />
              )}
              <span className="text-xs font-medium text-tks-gold">
                {step.actionHint}
              </span>
              <span className="text-[10px] text-muted-foreground ml-1">
                or use Next →
              </span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Progress bar */}
      <div className="h-0.5 bg-secondary/30 mx-4">
        <div
          className="h-full bg-tks-gold transition-all duration-300"
          style={{ width: `${((currentIndex + 1) / totalSteps) * 100}%` }}
        />
      </div>

      {/* Footer */}
      <div className="px-4 py-3 flex items-center justify-between">
        <button
          onClick={onSkip}
          className="text-[11px] text-muted-foreground hover:text-foreground transition-colors"
        >
          End Tour
        </button>
        <div className="flex items-center gap-2">
          {!isFirst && (
            <Button
              size="sm"
              variant="ghost"
              onClick={onPrev}
              className="h-7 text-xs text-muted-foreground"
            >
              <ArrowLeft className="h-3 w-3 mr-1" />
              Back
            </Button>
          )}
          <Button
            size="sm"
            onClick={onNext}
            className="h-7 bg-tks-gold hover:bg-tks-gold-light text-background text-xs"
          >
            {isLast ? (
              <>
                Finish
                <CheckCircle2 className="h-3 w-3 ml-1" />
              </>
            ) : (
              <>
                Next
                <ArrowRight className="h-3 w-3 ml-1" />
              </>
            )}
          </Button>
        </div>
      </div>
    </motion.div>
  );
}

// ─── Main Tour Component ──────────────────────────────────
interface InteractiveTourProps {
  open: boolean;
  onClose: () => void;
}

export default function InteractiveTour({ open, onClose }: InteractiveTourProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [targetRect, setTargetRect] = useState<SpotlightRect | null>(null);
  const [navigating, setNavigating] = useState(false);
  const [actionTriggered, setActionTriggered] = useState(false);
  const rafRef = useRef<number>(0);
  const advanceTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const step = TOUR_STEPS[currentStep];

  // Find and track the target element
  const updateTargetRect = useCallback(() => {
    if (!open || !step) return;

    const el = document.querySelector(step.selector);
    if (el) {
      const rect = el.getBoundingClientRect();
      setTargetRect({
        top: rect.top,
        left: rect.left,
        width: rect.width,
        height: rect.height,
      });
      setNavigating(false);

      // Scroll element into view if needed
      const isVisible =
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= window.innerHeight &&
        rect.right <= window.innerWidth;

      if (!isVisible) {
        el.scrollIntoView({ behavior: "smooth", block: "center" });
        setTimeout(() => {
          const newRect = el.getBoundingClientRect();
          setTargetRect({
            top: newRect.top,
            left: newRect.left,
            width: newRect.width,
            height: newRect.height,
          });
        }, 400);
      }
    } else {
      setTargetRect(null);
    }
  }, [open, step]);

  // Poll for element position
  useEffect(() => {
    if (!open) return;

    const tick = () => {
      updateTargetRect();
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);

    return () => cancelAnimationFrame(rafRef.current);
  }, [open, updateTargetRect]);

  // Navigate to the correct page for each step
  useEffect(() => {
    if (!open || !step?.navigateTo) return;

    const currentPath = window.location.pathname;
    if (currentPath !== step.navigateTo) {
      setNavigating(true);
      window.dispatchEvent(
        new CustomEvent("tks-tour-navigate", { detail: { path: step.navigateTo } })
      );
      setTimeout(updateTargetRect, 300);
    }
  }, [open, currentStep, step, updateTargetRect]);

  // ─── Action-triggered auto-advance ──────────────────────
  useEffect(() => {
    if (!open || !step) return;

    setActionTriggered(false);

    // Small delay to ensure the element is rendered
    const setupTimeout = setTimeout(() => {
      const el = document.querySelector(step.selector);
      if (!el) return;

      const handleAction = () => {
        setActionTriggered(true);

        // Auto-advance after a brief visual confirmation
        advanceTimeoutRef.current = setTimeout(() => {
          setCurrentStep((c) => {
            if (c >= TOUR_STEPS.length - 1) {
              // Last step — end the tour
              localStorage.setItem(TOUR_STORAGE_KEY, "true");
              onClose();
              return c;
            }
            return c + 1;
          });
        }, 800);
      };

      // Listen for the step's specific action event
      if (step.actionEvent === "input") {
        // For input events, listen on child textarea/input elements
        const inputs = el.querySelectorAll("textarea, input");
        inputs.forEach((input) => {
          input.addEventListener("input", handleAction, { once: true });
        });
        // Also listen on the element itself
        el.addEventListener("input", handleAction, { once: true });
      } else {
        // For click events, use capture phase to catch clicks on children
        el.addEventListener("click", handleAction, { once: true, capture: true });
      }

      return () => {
        if (step.actionEvent === "input") {
          const inputs = el.querySelectorAll("textarea, input");
          inputs.forEach((input) => {
            input.removeEventListener("input", handleAction);
          });
          el.removeEventListener("input", handleAction);
        } else {
          el.removeEventListener("click", handleAction, { capture: true });
        }
      };
    }, 200);

    return () => {
      clearTimeout(setupTimeout);
      if (advanceTimeoutRef.current) {
        clearTimeout(advanceTimeoutRef.current);
        advanceTimeoutRef.current = null;
      }
    };
  }, [open, currentStep, step, onClose]);

  // Keyboard navigation
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight" || e.key === "Enter") goNext();
      else if (e.key === "ArrowLeft") goPrev();
      else if (e.key === "Escape") handleEnd();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, currentStep]);

  // Reset on open
  useEffect(() => {
    if (open) {
      setCurrentStep(0);
      setTargetRect(null);
      setActionTriggered(false);
    }
  }, [open]);

  const goNext = useCallback(() => {
    if (advanceTimeoutRef.current) {
      clearTimeout(advanceTimeoutRef.current);
      advanceTimeoutRef.current = null;
    }
    if (currentStep >= TOUR_STEPS.length - 1) {
      handleEnd();
      return;
    }
    setActionTriggered(false);
    setCurrentStep((c) => c + 1);
  }, [currentStep]);

  const goPrev = useCallback(() => {
    if (currentStep <= 0) return;
    if (advanceTimeoutRef.current) {
      clearTimeout(advanceTimeoutRef.current);
      advanceTimeoutRef.current = null;
    }
    setActionTriggered(false);
    setCurrentStep((c) => c - 1);
  }, [currentStep]);

  const handleEnd = useCallback(() => {
    if (advanceTimeoutRef.current) {
      clearTimeout(advanceTimeoutRef.current);
      advanceTimeoutRef.current = null;
    }
    localStorage.setItem(TOUR_STORAGE_KEY, "true");
    setTargetRect(null);
    setActionTriggered(false);
    onClose();
  }, [onClose]);

  if (!open) return null;

  return (
    <AnimatePresence>
      {targetRect && !navigating && (
        <>
          <SpotlightOverlay
            rect={targetRect}
            onClickOutside={goNext}
            allowInteraction={step.allowInteraction}
          />
          <TourTooltip
            step={step}
            rect={targetRect}
            currentIndex={currentStep}
            totalSteps={TOUR_STEPS.length}
            onNext={goNext}
            onPrev={goPrev}
            onSkip={handleEnd}
            actionTriggered={actionTriggered}
          />
        </>
      )}
      {navigating && (
        <div className="fixed inset-0 z-[9998] bg-background/60 backdrop-blur-sm flex items-center justify-center">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex items-center gap-3 text-muted-foreground"
          >
            <Triangle className="h-5 w-5 text-tks-gold animate-pulse" />
            <span className="text-sm">Navigating...</span>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}
