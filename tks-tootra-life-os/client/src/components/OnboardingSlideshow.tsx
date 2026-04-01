import { useState, useCallback, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { motion, AnimatePresence } from "framer-motion";
import {
  Triangle, Mic, Volume2, Send, ArrowRight, ArrowLeft, X,
  Inbox, CalendarCheck, Play, CheckCircle2, Grid3X3,
  BarChart3, Zap, Flame, BookOpen, Eye, Sparkles, ImagePlay, Camera,
  MonitorPlay,
  Compass,
} from "lucide-react";

const STORAGE_KEY = "tks-onboarding-seen";

interface OnboardingSlideshowProps {
  open: boolean;
  onClose: () => void;
}

// ─── CDN URLs for walkthrough screenshots ──────────────────
const WALKTHROUGH = {
  inbox: "https://d2xsxph8kpxj0f.cloudfront.net/94944829/f5haNuBW7Ar9emg5jDjqxD/walkthrough-02-inbox-clean_52dcfb49.webp",
  typing: "https://d2xsxph8kpxj0f.cloudfront.net/94944829/f5haNuBW7Ar9emg5jDjqxD/walkthrough-03-typing_a895407e.webp",
  interpreted: "https://d2xsxph8kpxj0f.cloudfront.net/94944829/f5haNuBW7Ar9emg5jDjqxD/walkthrough-05-interpreted_ae4bb5a6.webp",
  taskDetail: "https://d2xsxph8kpxj0f.cloudfront.net/94944829/f5haNuBW7Ar9emg5jDjqxD/walkthrough-06-taskdetail_754d1807.webp",
  dwpPower: "https://d2xsxph8kpxj0f.cloudfront.net/94944829/f5haNuBW7Ar9emg5jDjqxD/walkthrough-07-dwp-power_4af9cb31.webp",
  scenarioizeReady: "https://d2xsxph8kpxj0f.cloudfront.net/94944829/f5haNuBW7Ar9emg5jDjqxD/walkthrough-10-scenarioize-ready_eb207ca2.webp",
  visionFull: "https://d2xsxph8kpxj0f.cloudfront.net/94944829/f5haNuBW7Ar9emg5jDjqxD/walkthrough-13-vision-full_6570cd19.webp",
  heatmap: "https://d2xsxph8kpxj0f.cloudfront.net/94944829/f5haNuBW7Ar9emg5jDjqxD/walkthrough-14-heatmap_b1471a38.webp",
  equations: "https://d2xsxph8kpxj0f.cloudfront.net/94944829/f5haNuBW7Ar9emg5jDjqxD/walkthrough-15-equations_73e8c51e.webp",
  powerGaps: "https://d2xsxph8kpxj0f.cloudfront.net/94944829/f5haNuBW7Ar9emg5jDjqxD/walkthrough-16-powergaps_b4435fef.webp",
  weeklyReview: "https://d2xsxph8kpxj0f.cloudfront.net/94944829/f5haNuBW7Ar9emg5jDjqxD/walkthrough-17-weeklyreview_343b977b.webp",
  dwpTracker: "https://d2xsxph8kpxj0f.cloudfront.net/94944829/f5haNuBW7Ar9emg5jDjqxD/walkthrough-18-dwptracker_236770d9.webp",
};

// ─── Slide Data ──────────────────────────────────────────────
interface Slide {
  id: string;
  icon: React.ReactNode;
  title: string;
  subtitle: string;
  body: React.ReactNode;
  accent: string; // tailwind color class for accent elements
}

const FOUNDATION_COLORS = [
  { name: "Selfhood", color: "#C8962E", desc: "Identity, purpose, self-knowledge" },
  { name: "Couplehood", color: "#E03C1F", desc: "Intimate relationships, partnership" },
  { name: "Familyhood", color: "#2B4E9B", desc: "Family bonds, parenting, heritage" },
  { name: "Socialhood", color: "#4CAF50", desc: "Friendships, community, belonging" },
  { name: "Professionalhood", color: "#9C27B0", desc: "Career, craft, professional growth" },
  { name: "Societalhood", color: "#00BCD4", desc: "Civic duty, society, contribution" },
  { name: "Creatorhood", color: "#FF9800", desc: "Creation, art, legacy building" },
];

const DWP_VISUAL = [
  { phase: "Desire", color: "#E03C1F", stages: ["Spark", "Craving", "Commitment", "Burning"], icon: "🔥" },
  { phase: "Wisdom", color: "#2B4E9B", stages: ["Curiosity", "Research", "Planning", "Mastery"], icon: "📘" },
  { phase: "Power", color: "#C8962E", stages: ["Awareness", "Gathering", "Wielding", "Overflow"], icon: "⚡" },
];

const TOKEN_EXAMPLES = [
  { type: "ACT", label: "study", color: "#C8962E", desc: "The action" },
  { type: "NOE", label: "technical", color: "#9C27B0", desc: "Knowledge type" },
  { type: "TIME", label: "evening", color: "#2B4E9B", desc: "When" },
  { type: "DUR", label: "2h", color: "#00BCD4", desc: "How long" },
  { type: "D", label: "craving", color: "#E03C1F", desc: "Desire stage" },
  { type: "MAP", label: "home", color: "#4CAF50", desc: "Where" },
];

const STATUS_FLOW = [
  { status: "Inbox", color: "#C8962E", icon: Inbox, desc: "Captured, awaiting triage" },
  { status: "Scheduled", color: "#2B4E9B", icon: CalendarCheck, desc: "Planned for a specific time" },
  { status: "Active", color: "#4CAF50", icon: Play, desc: "Currently being worked on" },
  { status: "Done", color: "#388E3C", icon: CheckCircle2, desc: "Completed successfully" },
];

/** Reusable screenshot thumbnail component */
function ScreenshotPreview({ src, alt, caption }: { src: string; alt: string; caption?: string }) {
  return (
    <div className="mt-3 space-y-1.5">
      <div className="rounded-lg overflow-hidden border border-border/50 shadow-lg">
        <img
          src={src}
          alt={alt}
          className="w-full h-auto"
          loading="lazy"
        />
      </div>
      {caption && (
        <p className="text-[10px] text-muted-foreground/70 text-center italic">{caption}</p>
      )}
    </div>
  );
}

const slides: Slide[] = [
  // ── 1. Welcome ──
  {
    id: "welcome",
    icon: <Triangle className="h-12 w-12 text-tks-gold fill-tks-gold/20" />,
    title: "Welcome to TKS Tootra",
    subtitle: "Your Life Execution System",
    accent: "tks-gold",
    body: (
      <div className="space-y-4 text-sm text-muted-foreground">
        <p className="text-base text-foreground leading-relaxed">
          TKS Tootra is not just a to-do app. It is a <strong className="text-tks-gold">life operating system</strong> built
          on the Total Knowledge System — a framework that maps every task to the deeper
          dimensions of your life.
        </p>
        <p className="leading-relaxed">
          Instead of a flat list of chores, you get a living map of your intentions,
          knowledge, and resources — so you can act with clarity and purpose.
        </p>
        <div className="mt-4 p-3 rounded-lg bg-tks-gold/5 border border-tks-gold/20 text-xs italic text-tks-gold/80">
          "Est Ars Celare Artem" — It is art to conceal art. The system does the heavy
          lifting so you can focus on living.
        </div>
      </div>
    ),
  },

  // ── 2. Capture ──
  {
    id: "capture",
    icon: <Send className="h-12 w-12 text-tks-gold" />,
    title: "Capture a Task",
    subtitle: "Type it or speak it — the AI does the rest",
    accent: "tks-gold",
    body: (
      <div className="space-y-4 text-sm text-muted-foreground">
        <p className="text-base text-foreground leading-relaxed">
          Just describe what you need to do in <strong>plain language</strong>. The AI will
          automatically figure out which life area it belongs to, how urgent it is, and
          what resources you need.
        </p>
        <ScreenshotPreview
          src={WALKTHROUGH.typing}
          alt="Typing a task into the capture input"
          caption="Type your task and click Capture & Map"
        />
        <div className="flex items-center gap-3 p-3 rounded-lg bg-tks-flame/5 border border-tks-flame/20">
          <Mic className="h-5 w-5 text-tks-flame shrink-0" />
          <p className="text-xs">
            <strong className="text-tks-flame">Voice capture:</strong> Click the <strong>Speak</strong> button
            to dictate your task instead of typing.
          </p>
        </div>
      </div>
    ),
  },

  // ── 3. AI Interpretation ──
  {
    id: "interpretation",
    icon: <Sparkles className="h-12 w-12 text-tks-gold" />,
    title: "AI Interprets Your Task",
    subtitle: "Foundation mapping, confidence scoring, and equation generation",
    accent: "tks-gold",
    body: (
      <div className="space-y-4 text-sm text-muted-foreground">
        <p className="text-base text-foreground leading-relaxed">
          After you capture a task, the AI instantly maps it to a <strong className="text-tks-gold">Foundation</strong>,
          assigns an archetype, generates a <strong className="text-tks-gold">TKS Equation</strong>, and
          scores its confidence — all in seconds.
        </p>
        <ScreenshotPreview
          src={WALKTHROUGH.interpreted}
          alt="AI interpretation result showing TKS equation and D/W/P bars"
          caption="AI maps to Knowledge Foundation with 90% confidence, generates full TKS equation"
        />
      </div>
    ),
  },

  // ── 4. Foundations ──
  {
    id: "foundations",
    icon: <Grid3X3 className="h-12 w-12 text-tks-gold" />,
    title: "The 7 Foundations",
    subtitle: "Every task maps to a dimension of your life",
    accent: "tks-gold",
    body: (
      <div className="space-y-3 text-sm">
        <p className="text-muted-foreground leading-relaxed">
          TKS organizes life into <strong className="text-foreground">7 Foundations</strong>. When you capture a task,
          the AI determines which Foundation it serves — so you can see if you are
          neglecting any area.
        </p>
        <div className="grid grid-cols-1 gap-1.5">
          {FOUNDATION_COLORS.map((f) => (
            <div key={f.name} className="flex items-center gap-2 p-2 rounded-md bg-secondary/30">
              <span
                className="w-3 h-3 rounded-full shrink-0"
                style={{ backgroundColor: f.color }}
              />
              <span className="text-xs font-medium text-foreground w-32">{f.name}</span>
              <span className="text-xs text-muted-foreground">{f.desc}</span>
            </div>
          ))}
        </div>
      </div>
    ),
  },

  // ── 5. D/W/P Lifecycle ──
  {
    id: "dwp",
    icon: <Flame className="h-12 w-12 text-tks-flame" />,
    title: "Desire / Wisdom / Power",
    subtitle: "The lifecycle of every intention",
    accent: "tks-flame",
    body: (
      <div className="space-y-4 text-sm">
        <p className="text-muted-foreground leading-relaxed">
          Every task moves through three phases. <strong className="text-foreground">Desire</strong> is your motivation.{" "}
          <strong className="text-foreground">Wisdom</strong> is your understanding. <strong className="text-foreground">Power</strong> is
          your ability to act.
        </p>
        <div className="space-y-3">
          {DWP_VISUAL.map((phase) => (
            <div key={phase.phase} className="space-y-1">
              <div className="flex items-center gap-2">
                <span className="text-base">{phase.icon}</span>
                <span className="text-xs font-semibold" style={{ color: phase.color }}>
                  {phase.phase}
                </span>
              </div>
              <div className="flex gap-1 ml-7">
                {phase.stages.map((stage, i) => (
                  <div
                    key={stage}
                    className="flex-1 text-center py-1.5 rounded text-[10px] font-medium"
                    style={{
                      backgroundColor: phase.color + (i === 0 ? "15" : i === 1 ? "25" : i === 2 ? "35" : "50"),
                      color: phase.color,
                    }}
                  >
                    {stage}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
        <ScreenshotPreview
          src={WALKTHROUGH.dwpTracker}
          alt="D/W/P Lifecycle Tracker showing Desire, Wisdom, Power distributions and per-task lifecycle"
          caption="The D/W/P Tracker monitors lifecycle stages across all your tasks with distribution charts"
        />
      </div>
    ),
  },

  // ── 6. TKS Equations ──
  {
    id: "equations",
    icon: <Sparkles className="h-12 w-12 text-purple-400" />,
    title: "TKS Equations",
    subtitle: "A visual formula for every task",
    accent: "purple-400",
    body: (
      <div className="space-y-4 text-sm">
        <p className="text-muted-foreground leading-relaxed">
          Each task gets a <strong className="text-foreground">TKS Equation</strong> — a color-coded formula
          that captures the action, timing, knowledge type, desire level, and more in a single glance.
        </p>
        <div className="flex flex-wrap gap-1.5 p-3 rounded-lg bg-secondary/30 border border-border">
          {TOKEN_EXAMPLES.map((token) => (
            <span
              key={token.type}
              className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-[11px] font-medium"
              style={{
                backgroundColor: token.color + "20",
                color: token.color,
                border: `1px solid ${token.color}40`,
              }}
            >
              <span className="font-bold">{token.type}</span>
              <span className="opacity-70">({token.label})</span>
            </span>
          ))}
        </div>
        <ScreenshotPreview
          src={WALKTHROUGH.equations}
          alt="TKS Equations page showing color-coded token types"
          caption="The Equations page shows all your tasks as color-coded TKS formulas"
        />
      </div>
    ),
  },

  // ── 7. Task Management ──
  {
    id: "status",
    icon: <CalendarCheck className="h-12 w-12 text-tks-blue" />,
    title: "Task Status Flow",
    subtitle: "From capture to completion",
    accent: "tks-blue",
    body: (
      <div className="space-y-4 text-sm">
        <p className="text-muted-foreground leading-relaxed">
          Tasks move through a clear pipeline. You control the flow — schedule when ready,
          start when focused, complete when done.
        </p>
        <div className="flex items-center gap-1">
          {STATUS_FLOW.map((s, i) => (
            <div key={s.status} className="flex items-center gap-1 flex-1">
              <div
                className="flex flex-col items-center gap-1 p-2 rounded-lg flex-1 text-center"
                style={{ backgroundColor: s.color + "15", border: `1px solid ${s.color}30` }}
              >
                <s.icon className="h-4 w-4" style={{ color: s.color }} />
                <span className="text-[10px] font-semibold" style={{ color: s.color }}>{s.status}</span>
                <span className="text-[9px] text-muted-foreground leading-tight">{s.desc}</span>
              </div>
              {i < STATUS_FLOW.length - 1 && (
                <ArrowRight className="h-3 w-3 text-muted-foreground/40 shrink-0" />
              )}
            </div>
          ))}
        </div>
        <ScreenshotPreview
          src={WALKTHROUGH.taskDetail}
          alt="Task detail page showing full TKS analysis"
          caption="Click any task to see its full TKS analysis, equation, and D/W/P diagnosis"
        />
      </div>
    ),
  },

  // ── 8. Heatmap & Power Gaps ──
  {
    id: "heatmap",
    icon: <BarChart3 className="h-12 w-12 text-green-400" />,
    title: "Heatmap & Power Gaps",
    subtitle: "Spot imbalances before they become problems",
    accent: "green-400",
    body: (
      <div className="space-y-4 text-sm">
        <p className="text-muted-foreground leading-relaxed">
          The <strong className="text-foreground">Foundation Heatmap</strong> shows how your tasks are distributed
          across the 7 life areas. Hot zones mean overload; cold zones mean neglect.
        </p>
        <ScreenshotPreview
          src={WALKTHROUGH.heatmap}
          alt="Foundation Heatmap showing 7 life areas with balance analytics"
          caption="The Heatmap shows all 7 Foundations with overload/neglect alerts and balance analytics"
        />
        <p className="text-muted-foreground leading-relaxed">
          <strong className="text-foreground">Power Gap Analysis</strong> checks 8 resource categories for each
          task — time, money, tools, access, body, skill, connections, and environment — and
          generates subtasks to close any gaps.
        </p>
        <ScreenshotPreview
          src={WALKTHROUGH.powerGaps}
          alt="Power Gap Analysis showing 8 resource categories with Have/Partial/Lack status"
          caption="Power Gaps assess all 8 resource categories and generate subtasks to close gaps"
        />
      </div>
    ),
  },

  // ── 9. Weekly Review ──
  {
    id: "review",
    icon: <BookOpen className="h-12 w-12 text-tks-blue" />,
    title: "Weekly Review",
    subtitle: "Reflect, recalibrate, and export",
    accent: "tks-blue",
    body: (
      <div className="space-y-4 text-sm">
        <p className="text-muted-foreground leading-relaxed">
          Each week, the <strong className="text-foreground">Weekly Review</strong> dashboard shows your completion
          rate, foundation balance trends, and drift analysis — whether you are moving toward
          or away from balance.
        </p>
        <div className="grid grid-cols-3 gap-2">
          {[
            { label: "Completion Rate", value: "73%", color: "#4CAF50" },
            { label: "Active Tasks", value: "12", color: "#2B4E9B" },
            { label: "Foundations Hit", value: "5/7", color: "#C8962E" },
          ].map((stat) => (
            <div
              key={stat.label}
              className="p-2 rounded-lg text-center"
              style={{ backgroundColor: stat.color + "10", border: `1px solid ${stat.color}20` }}
            >
              <div className="text-lg font-bold" style={{ color: stat.color }}>{stat.value}</div>
              <div className="text-[10px] text-muted-foreground">{stat.label}</div>
            </div>
          ))}
        </div>
        <p className="text-xs text-muted-foreground">
          Export your data anytime as <strong>CSV</strong> or <strong>JSON</strong> for external analysis
          or backup.
        </p>
        <ScreenshotPreview
          src={WALKTHROUGH.weeklyReview}
          alt="Weekly Review dashboard showing completion rate, foundation balance, and drift analysis"
          caption="Weekly Review tracks completion rates, foundation balance, and drift analysis"
        />
      </div>
    ),
  },

  // ── 10. Modes ──
  {
    id: "modes",
    icon: <Eye className="h-12 w-12 text-tks-gold" />,
    title: "Two Modes",
    subtitle: "Deep thinking or pure execution — your choice",
    accent: "tks-gold",
    body: (
      <div className="space-y-4 text-sm">
        <p className="text-muted-foreground leading-relaxed">
          Switch between two operational modes depending on what you need right now.
        </p>
        <div className="grid grid-cols-2 gap-3">
          <div className="p-3 rounded-lg bg-tks-gold/5 border border-tks-gold/20 space-y-2">
            <div className="flex items-center gap-2">
              <Triangle className="h-4 w-4 text-tks-gold fill-tks-gold/20" />
              <span className="text-xs font-semibold text-tks-gold">Life OS</span>
            </div>
            <p className="text-[11px] text-muted-foreground leading-relaxed">
              Full TKS visibility. See foundations, equations, D/W/P stages, heatmaps,
              power gaps, and weekly reviews. For intentional, reflective work.
            </p>
          </div>
          <div className="p-3 rounded-lg bg-tks-flame/5 border border-tks-flame/20 space-y-2">
            <div className="flex items-center gap-2">
              <Flame className="h-4 w-4 text-tks-flame" />
              <span className="text-xs font-semibold text-tks-flame">Work Sprint</span>
            </div>
            <p className="text-[11px] text-muted-foreground leading-relaxed">
              Minimal overhead. Just Inbox and Tasks with quick-capture and one-click
              actions. For pure execution when you need to get things done fast.
            </p>
          </div>
        </div>
        <p className="text-xs text-muted-foreground">
          Toggle between modes using the switch at the bottom of the sidebar.
        </p>
      </div>
    ),
  },

  // ── 11. Voice & Read Aloud ──
  {
    id: "voice",
    icon: <Mic className="h-12 w-12 text-tks-flame" />,
    title: "Voice & Read Aloud",
    subtitle: "Speak your tasks, hear your summaries",
    accent: "tks-flame",
    body: (
      <div className="space-y-4 text-sm">
        <div className="flex gap-3">
          <div className="flex-1 p-3 rounded-lg bg-tks-flame/5 border border-tks-flame/20 space-y-2">
            <div className="flex items-center gap-2">
              <Mic className="h-4 w-4 text-tks-flame" />
              <span className="text-xs font-semibold text-tks-flame">Voice Capture</span>
            </div>
            <p className="text-[11px] text-muted-foreground leading-relaxed">
              Click <strong>Speak</strong> or the mic icon, talk naturally, and your words are
              transcribed into the task input.
            </p>
          </div>
          <div className="flex-1 p-3 rounded-lg bg-tks-blue/5 border border-tks-blue/20 space-y-2">
            <div className="flex items-center gap-2">
              <Volume2 className="h-4 w-4 text-tks-blue" />
              <span className="text-xs font-semibold text-tks-blue">Read Aloud</span>
            </div>
            <p className="text-[11px] text-muted-foreground leading-relaxed">
              Click the speaker icon on any task to hear its full summary — title, description,
              D/W/P diagnosis, and power inventory.
            </p>
          </div>
        </div>
        <p className="text-xs text-muted-foreground">
          Your browser will ask for microphone permission the first time. Voice and speed
          preferences are saved automatically.
        </p>
      </div>
    ),
  },

  // ── 12. Scenarioize ──
  {
    id: "scenarioize",
    icon: <ImagePlay className="h-12 w-12 text-purple-400" />,
    title: "Scenarioize Your Vision",
    subtitle: "See the future consequences of your choices",
    accent: "purple-400",
    body: (
      <div className="space-y-4 text-sm">
        <p className="text-muted-foreground leading-relaxed">
          Upload a photo of yourself and the AI generates a <strong className="text-foreground">realistic vision</strong> of
          what happens if you follow through — or what happens if you don't.
        </p>
        <ScreenshotPreview
          src={WALKTHROUGH.scenarioizeReady}
          alt="Scenarioize panel with photo uploaded and ready to generate"
          caption="Upload your photo, choose a mode, and click Generate"
        />
        <ScreenshotPreview
          src={WALKTHROUGH.visionFull}
          alt="AI-generated scenario image showing positive outcome"
          caption="AI generates a realistic vision of your future if you follow through"
        />
      </div>
    ),
  },

  // ── 13. Live Walkthrough ──
  {
    id: "walkthrough",
    icon: <MonitorPlay className="h-12 w-12 text-tks-gold" />,
    title: "See It In Action",
    subtitle: "A real walkthrough of capturing and analyzing a task",
    accent: "tks-gold",
    body: (
      <div className="space-y-3 text-sm text-muted-foreground">
        <p className="text-base text-foreground leading-relaxed">
          Here is a real walkthrough showing the complete flow from task capture to AI analysis:
        </p>
        <div className="space-y-3">
          <div className="flex items-start gap-3 p-2.5 rounded-lg bg-secondary/30">
            <span className="text-tks-gold font-bold text-xs mt-0.5 shrink-0">1.</span>
            <div className="space-y-1.5 flex-1">
              <span className="text-xs text-foreground font-medium">Type your task in the Inbox</span>
              <div className="rounded-md overflow-hidden border border-border/40">
                <img src={WALKTHROUGH.inbox} alt="Clean inbox" className="w-full h-auto" loading="lazy" />
              </div>
            </div>
          </div>
          <div className="flex items-start gap-3 p-2.5 rounded-lg bg-secondary/30">
            <span className="text-tks-gold font-bold text-xs mt-0.5 shrink-0">2.</span>
            <div className="space-y-1.5 flex-1">
              <span className="text-xs text-foreground font-medium">AI maps it to a Foundation and generates the equation</span>
              <div className="rounded-md overflow-hidden border border-border/40">
                <img src={WALKTHROUGH.interpreted} alt="AI interpretation" className="w-full h-auto" loading="lazy" />
              </div>
            </div>
          </div>
          <div className="flex items-start gap-3 p-2.5 rounded-lg bg-secondary/30">
            <span className="text-tks-gold font-bold text-xs mt-0.5 shrink-0">3.</span>
            <div className="space-y-1.5 flex-1">
              <span className="text-xs text-foreground font-medium">Scenarioize generates your future vision</span>
              <div className="rounded-md overflow-hidden border border-border/40">
                <img src={WALKTHROUGH.visionFull} alt="Generated vision" className="w-full h-auto" loading="lazy" />
              </div>
            </div>
          </div>
        </div>
      </div>
    ),
  },

  // ── 14. Get Started ──
  {
    id: "getstarted",
    icon: <Zap className="h-12 w-12 text-tks-gold" />,
    title: "You're Ready",
    subtitle: "Capture your first task and let the system map your life",
    accent: "tks-gold",
    body: (
      <div className="space-y-4 text-sm text-muted-foreground">
        <p className="text-base text-foreground leading-relaxed">
          That is everything you need to know. The system handles the complexity —
          you just need to <strong className="text-tks-gold">speak your intention</strong>.
        </p>
        <div className="space-y-2 p-3 rounded-lg bg-secondary/30 border border-border">
          <p className="text-xs font-medium text-foreground">Quick start checklist:</p>
          <div className="space-y-1.5">
            {[
              "Type or speak a task in the Inbox",
              "Watch the AI map it to a Foundation and generate an Equation",
              "Review the D/W/P stages and Power inventory",
              "Scenarioize it — see what happens if you do (or don't)",
              "Schedule it, start it, complete it",
              "Check your Heatmap to see your life balance",
            ].map((step, i) => (
              <div key={i} className="flex items-start gap-2">
                <span className="text-tks-gold font-bold text-xs mt-0.5">{i + 1}.</span>
                <span className="text-xs">{step}</span>
              </div>
            ))}
          </div>
        </div>
        <p className="text-xs italic text-tks-gold/70">
          You can reopen this guide anytime from the sidebar — look for "How to Use" or "Take a Tour."
        </p>
      </div>
    ),
  },
];

// ─── Component ───────────────────────────────────────────────
export default function OnboardingSlideshow({ open, onClose }: OnboardingSlideshowProps) {
  const [current, setCurrent] = useState(0);
  const [direction, setDirection] = useState(0); // -1 = prev, 1 = next

  const slide = slides[current];
  const isFirst = current === 0;
  const isLast = current === slides.length - 1;

  const goNext = useCallback(() => {
    if (isLast) {
      localStorage.setItem(STORAGE_KEY, "true");
      onClose();
      return;
    }
    setDirection(1);
    setCurrent((c) => c + 1);
  }, [isLast, onClose]);

  const handleStartTour = useCallback(() => {
    localStorage.setItem(STORAGE_KEY, "true");
    onClose();
    // Small delay to let the onboarding close before tour opens
    setTimeout(() => {
      window.dispatchEvent(new CustomEvent("tks-start-tour"));
    }, 300);
  }, [onClose]);

  const goPrev = useCallback(() => {
    if (isFirst) return;
    setDirection(-1);
    setCurrent((c) => c - 1);
  }, [isFirst]);

  const handleSkip = useCallback(() => {
    localStorage.setItem(STORAGE_KEY, "true");
    onClose();
  }, [onClose]);

  // Reset to first slide when opened
  useEffect(() => {
    if (open) setCurrent(0);
  }, [open]);

  // Keyboard navigation
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight" || e.key === "Enter") goNext();
      else if (e.key === "ArrowLeft") goPrev();
      else if (e.key === "Escape") handleSkip();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, goNext, goPrev, handleSkip]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="absolute inset-0 bg-background/80 backdrop-blur-sm"
        onClick={handleSkip}
      />

      {/* Modal */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: 20 }}
        transition={{ duration: 0.3, ease: "easeOut" }}
        className="relative z-10 w-full max-w-lg mx-4 bg-card border border-border rounded-2xl shadow-2xl overflow-hidden max-h-[90vh] flex flex-col"
      >
        {/* Close & Skip controls - always visible */}
        <div className="absolute top-3 right-3 z-20 flex items-center gap-2">
          {!isLast && (
            <button
              onClick={handleSkip}
              className="text-[11px] text-muted-foreground hover:text-tks-gold transition-colors flex items-center gap-1 px-2 py-1 rounded-md hover:bg-secondary/60"
            >
              Skip to App
              <ArrowRight className="h-3 w-3" />
            </button>
          )}
          <button
            onClick={handleSkip}
            className="p-1.5 rounded-full hover:bg-secondary/80 text-muted-foreground hover:text-foreground transition-colors"
            aria-label="Close"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Progress bar */}
        <div className="h-1 bg-secondary/30 shrink-0">
          <motion.div
            className="h-full bg-tks-gold"
            initial={false}
            animate={{ width: `${((current + 1) / slides.length) * 100}%` }}
            transition={{ duration: 0.3 }}
          />
        </div>

        {/* Slide content */}
        <div className="px-6 pt-6 pb-4 min-h-[420px] flex flex-col flex-1 overflow-hidden">
          <AnimatePresence mode="wait" custom={direction}>
            <motion.div
              key={slide.id}
              custom={direction}
              initial={{ opacity: 0, x: direction >= 0 ? 40 : -40 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: direction >= 0 ? -40 : 40 }}
              transition={{ duration: 0.25, ease: "easeInOut" }}
              className="flex-1 flex flex-col overflow-hidden"
            >
              {/* Icon + Title */}
              <div className="flex items-start gap-4 mb-4 shrink-0">
                <div className="shrink-0 p-2 rounded-xl bg-secondary/30">
                  {slide.icon}
                </div>
                <div>
                  <h2 className="text-lg font-bold tracking-tight text-foreground">{slide.title}</h2>
                  <p className="text-xs text-muted-foreground mt-0.5">{slide.subtitle}</p>
                </div>
              </div>

              {/* Body - scrollable */}
              <div className="flex-1 overflow-y-auto pr-1 scrollbar-thin">{slide.body}</div>
            </motion.div>
          </AnimatePresence>
        </div>

        {/* Footer: dots + navigation */}
        <div className="px-6 pb-5 pt-2 flex items-center justify-between shrink-0">
          {/* Progress dots */}
          <div className="flex gap-1.5">
            {slides.map((_, i) => (
              <button
                key={i}
                onClick={() => {
                  setDirection(i > current ? 1 : -1);
                  setCurrent(i);
                }}
                className={`h-1.5 rounded-full transition-all duration-300 ${
                  i === current
                    ? "w-6 bg-tks-gold"
                    : i < current
                    ? "w-1.5 bg-tks-gold/40"
                    : "w-1.5 bg-secondary"
                }`}
                aria-label={`Go to slide ${i + 1}`}
              />
            ))}
          </div>

          {/* Buttons */}
          <div className="flex items-center gap-2">
            {!isFirst && (
              <Button
                size="sm"
                variant="ghost"
                onClick={goPrev}
                className="text-xs text-muted-foreground"
              >
                <ArrowLeft className="h-3 w-3 mr-1" />
                Back
              </Button>
            )}
            {!isLast && (
              <Button
                size="sm"
                variant="outline"
                onClick={handleSkip}
                className="text-xs border-border/50 hover:border-tks-gold/30 hover:text-tks-gold"
              >
                Skip to App
                <ArrowRight className="h-3 w-3 ml-1" />
              </Button>
            )}
            {isLast && (
              <Button
                size="sm"
                variant="outline"
                onClick={handleStartTour}
                className="text-xs border-tks-gold/30 text-tks-gold hover:bg-tks-gold/10"
              >
                <Compass className="h-3 w-3 mr-1" />
                Take a Tour
              </Button>
            )}
            <Button
              size="sm"
              onClick={goNext}
              className="bg-tks-gold hover:bg-tks-gold-light text-background text-xs"
            >
              {isLast ? (
                <>
                  Get Started
                  <Zap className="h-3 w-3 ml-1" />
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
    </div>
  );
}

/** Check if onboarding has been completed */
export function hasSeenOnboarding(): boolean {
  return localStorage.getItem(STORAGE_KEY) === "true";
}

/** Reset onboarding state (for "How to Use" relaunch) */
export function resetOnboarding(): void {
  localStorage.removeItem(STORAGE_KEY);
}
