// ═══════════════════════════════════════════════════════════════
// TKS (Total Knowledge System) — Shared Types & Constants
// ═══════════════════════════════════════════════════════════════

// ── Foundations ──────────────────────────────────────────────
export const FOUNDATIONS = [
  { id: 1, name: "Self", symbol: "♈", world: "Fire", color: "#E03C1F", description: "Identity, ego, self-awareness, personal truth" },
  { id: 2, name: "Knowledge", symbol: "♉", world: "Earth", color: "#2B4E9B", description: "Learning, education, skill acquisition, wisdom pursuit" },
  { id: 3, name: "Body", symbol: "♊", world: "Air", color: "#4CAF50", description: "Physical health, fitness, nutrition, bodily care" },
  { id: 4, name: "Relationships", symbol: "♋", world: "Water", color: "#E91E63", description: "Family, friends, romantic, social connections" },
  { id: 5, name: "Work", symbol: "♌", world: "Fire", color: "#C8962E", description: "Career, profession, vocation, livelihood" },
  { id: 6, name: "Environment", symbol: "♍", world: "Earth", color: "#795548", description: "Home, possessions, finances, material world" },
  { id: 7, name: "Spirit", symbol: "♎", world: "Air", color: "#9C27B0", description: "Purpose, meaning, creativity, transcendence" },
] as const;

export const SUB_FOUNDATIONS = [
  // Foundation 1: Self
  { id: "1a", name: "Identity", foundationId: 1, description: "Who you are at your core" },
  { id: "1b", name: "Values", foundationId: 1, description: "What you stand for" },
  { id: "1c", name: "Boundaries", foundationId: 1, description: "What you protect" },
  { id: "1d", name: "Expression", foundationId: 1, description: "How you show up" },
  // Foundation 2: Knowledge
  { id: "2a", name: "Formal Education", foundationId: 2, description: "Structured learning" },
  { id: "2b", name: "Self-Education", foundationId: 2, description: "Autodidactic pursuit" },
  { id: "2c", name: "Applied Knowledge", foundationId: 2, description: "Practical application" },
  { id: "2d", name: "Teaching", foundationId: 2, description: "Knowledge transmission" },
  // Foundation 3: Body
  { id: "3a", name: "Exercise", foundationId: 3, description: "Physical training" },
  { id: "3b", name: "Nutrition", foundationId: 3, description: "Fuel and diet" },
  { id: "3c", name: "Rest", foundationId: 3, description: "Sleep and recovery" },
  { id: "3d", name: "Appearance", foundationId: 3, description: "Grooming and presentation" },
  // Foundation 4: Relationships
  { id: "4a", name: "Family", foundationId: 4, description: "Blood and chosen family" },
  { id: "4b", name: "Romantic", foundationId: 4, description: "Intimate partnership" },
  { id: "4c", name: "Friends", foundationId: 4, description: "Social bonds" },
  { id: "4d", name: "Community", foundationId: 4, description: "Broader social circles" },
  // Foundation 5: Work
  { id: "5a", name: "Career", foundationId: 5, description: "Professional trajectory" },
  { id: "5b", name: "Projects", foundationId: 5, description: "Active work initiatives" },
  { id: "5c", name: "Skills", foundationId: 5, description: "Professional capabilities" },
  { id: "5d", name: "Network", foundationId: 5, description: "Professional connections" },
  // Foundation 6: Environment
  { id: "6a", name: "Home", foundationId: 6, description: "Living space" },
  { id: "6b", name: "Finances", foundationId: 6, description: "Money management" },
  { id: "6c", name: "Possessions", foundationId: 6, description: "Material goods" },
  { id: "6d", name: "Errands", foundationId: 6, description: "Maintenance tasks" },
  // Foundation 7: Spirit
  { id: "7a", name: "Purpose", foundationId: 7, description: "Life mission" },
  { id: "7b", name: "Creativity", foundationId: 7, description: "Creative expression" },
  { id: "7c", name: "Mindfulness", foundationId: 7, description: "Presence and meditation" },
  { id: "7d", name: "Transcendence", foundationId: 7, description: "Beyond the self" },
] as const;

export const WORLDS = [
  { id: "fire", name: "Fire", symbol: "🔥", foundations: [1, 5] },
  { id: "earth", name: "Earth", symbol: "🌍", foundations: [2, 6] },
  { id: "air", name: "Air", symbol: "💨", foundations: [3, 7] },
  { id: "water", name: "Water", symbol: "💧", foundations: [4] },
] as const;

// ── Noetics (10) ────────────────────────────────────────────
export const NOETICS = [
  { id: 0, symbol: "^0", name: "Idea", meaning: "Seed of new thought or intention" },
  { id: 1, symbol: "^1", name: "Mind", meaning: "Focused attention and concentration" },
  { id: 2, symbol: "^2", name: "Positive", meaning: "Constructive, affirming energy" },
  { id: 3, symbol: "^3", name: "Negative", meaning: "Destructive, problem-indicating energy" },
  { id: 4, symbol: "^4", name: "Vibration", meaning: "Energy frequency and resonance" },
  { id: 5, symbol: "^5", name: "Female", meaning: "Receptive, nurturing quality" },
  { id: 6, symbol: "^6", name: "Male", meaning: "Active, projective quality" },
  { id: 7, symbol: "^7", name: "Rhythm", meaning: "Cyclical pattern and cadence" },
  { id: 8, symbol: "^8", name: "Above/Cause", meaning: "Causal force, what drives" },
  { id: 9, symbol: "^9", name: "Below/Effect", meaning: "Resultant force, what follows" },
] as const;

// ── Operators (11) ──────────────────────────────────────────
export const OPERATORS = [
  { id: "o", name: "Sequence", meaning: "Then / and then" },
  { id: "->", name: "Cause", meaning: "Leads to / enables" },
  { id: "<-", name: "Effect", meaning: "Caused by / result of" },
  { id: "+", name: "Combine", meaning: "Together with" },
  { id: "-", name: "Without", meaning: "Excluding" },
  { id: "*", name: "Amplify", meaning: "Intensified" },
  { id: "/", name: "Divide", meaning: "Split between" },
  { id: "||", name: "Parallel", meaning: "Simultaneously" },
  { id: "?", name: "Conditional", meaning: "If / depends on" },
  { id: "!", name: "Urgent", meaning: "Critical priority" },
  { id: "~", name: "Approximate", meaning: "Roughly / about" },
] as const;

// ── Token Types (13) ────────────────────────────────────────
export const TOKEN_TYPES = {
  ACT:  { label: "Action",      color: "#C8962E", bgColor: "#C8962E20", icon: "⚡" },
  TIME: { label: "Time",        color: "#2B4E9B", bgColor: "#2B4E9B20", icon: "🕐" },
  DUR:  { label: "Duration",    color: "#0288D1", bgColor: "#0288D120", icon: "⏱" },
  RHY:  { label: "Rhythm",      color: "#7B1FA2", bgColor: "#7B1FA220", icon: "🔄" },
  MAP:  { label: "Mapping",     color: "#388E3C", bgColor: "#388E3C20", icon: "🗺" },
  NOE:  { label: "Noetic",      color: "#F57C00", bgColor: "#F57C0020", icon: "✧" },
  LINK: { label: "Link",        color: "#5C6BC0", bgColor: "#5C6BC020", icon: "🔗" },
  D:    { label: "Desire",      color: "#E03C1F", bgColor: "#E03C1F20", icon: "🔥" },
  W:    { label: "Wisdom",      color: "#2B4E9B", bgColor: "#2B4E9B20", icon: "📖" },
  P:    { label: "Power",       color: "#C8962E", bgColor: "#C8962E20", icon: "⚡" },
  REQ:  { label: "Requires",    color: "#D32F2F", bgColor: "#D32F2F20", icon: "❗" },
  HAVE: { label: "Has",         color: "#2E7D32", bgColor: "#2E7D3220", icon: "✓" },
  LACK: { label: "Lacks",       color: "#C62828", bgColor: "#C6282820", icon: "✗" },
} as const;

export type TokenType = keyof typeof TOKEN_TYPES;

// ── D/W/P Lifecycle (12 Stages) ─────────────────────────────
export const DWP_STAGES = {
  desire: [
    { id: "D_MIDNIGHT", name: "Midnight", phase: "desire", order: 0, meaning: "No desire yet — dormant seed", card: "Ace", zodiac: "Aries" },
    { id: "D_SUNRISE",  name: "Sunrise",  phase: "desire", order: 1, meaning: "Desire awakening — first spark", card: "2", zodiac: "Taurus" },
    { id: "D_NOON",     name: "Noon",     phase: "desire", order: 2, meaning: "Desire at peak — full motivation", card: "3", zodiac: "Gemini" },
    { id: "D_SUNSET",   name: "Sunset",   phase: "desire", order: 3, meaning: "Desire fading — waning motivation", card: "4", zodiac: "Cancer" },
  ],
  wisdom: [
    { id: "W_IGNORANCE",    name: "Ignorance",    phase: "wisdom", order: 0, meaning: "Don't know what you don't know", card: "5", zodiac: "Leo" },
    { id: "W_LEARNING",     name: "Learning",     phase: "wisdom", order: 1, meaning: "Actively acquiring knowledge", card: "6", zodiac: "Virgo" },
    { id: "W_KNOWING",      name: "Knowing",      phase: "wisdom", order: 2, meaning: "Sufficient knowledge to act", card: "7", zodiac: "Libra" },
    { id: "W_MASTERING",    name: "Mastering",    phase: "wisdom", order: 3, meaning: "Deep expertise, automatic", card: "8", zodiac: "Scorpio" },
  ],
  power: [
    { id: "P_BIRTH",        name: "Birth",        phase: "power", order: 0, meaning: "No resources gathered yet", card: "9", zodiac: "Sagittarius" },
    { id: "P_ADOLESCENCE",  name: "Adolescence",  phase: "power", order: 1, meaning: "Some resources, gaps remain", card: "10", zodiac: "Capricorn" },
    { id: "P_MATURITY",     name: "Maturity",     phase: "power", order: 2, meaning: "All resources available", card: "Jack", zodiac: "Aquarius" },
    { id: "P_MASTERY",      name: "Mastery",      phase: "power", order: 3, meaning: "Surplus resources, can share", card: "Queen", zodiac: "Pisces" },
  ],
} as const;

export type DWPPhase = "desire" | "wisdom" | "power";
export type DesireStage = "D_MIDNIGHT" | "D_SUNRISE" | "D_NOON" | "D_SUNSET";
export type WisdomStage = "W_IGNORANCE" | "W_LEARNING" | "W_KNOWING" | "W_MASTERING";
export type PowerStage = "P_BIRTH" | "P_ADOLESCENCE" | "P_MATURITY" | "P_MASTERY";

// ── Task Archetypes (6) ─────────────────────────────────────
export const TASK_ARCHETYPES = [
  { id: "execution",   name: "Execution",   icon: "⚡", color: "#C8962E", description: "Do the thing — concrete action producing a result" },
  { id: "prep",        name: "Preparation", icon: "📋", color: "#2B4E9B", description: "Get ready — gathering resources and prerequisites" },
  { id: "maintenance", name: "Maintenance", icon: "🔄", color: "#388E3C", description: "Keep doing — recurring rhythm-driven tasks" },
  { id: "repair",      name: "Repair",      icon: "🔧", color: "#D32F2F", description: "Fix what broke — reactive to failure or disruption" },
  { id: "study",       name: "Study",       icon: "📖", color: "#7B1FA2", description: "Learn before doing — knowledge acquisition" },
  { id: "audit",       name: "Audit",       icon: "🔍", color: "#795548", description: "Check how it went — review and retrospective" },
] as const;

export type TaskArchetype = "execution" | "prep" | "maintenance" | "repair" | "study" | "audit";

// ── Task Statuses ───────────────────────────────────────────
export const TASK_STATUSES = ["inbox", "scheduled", "active", "done", "dropped"] as const;
export type TaskStatus = (typeof TASK_STATUSES)[number];

// ── Power Categories (8) ────────────────────────────────────
export const POWER_CATEGORIES = [
  { id: "time",        name: "Time",         icon: "⏰", description: "Available hours and scheduling capacity" },
  { id: "money",       name: "Money",        icon: "💰", description: "Financial resources needed" },
  { id: "tools",       name: "Tools",        icon: "🛠", description: "Equipment, software, instruments" },
  { id: "access",      name: "Access",       icon: "🔑", description: "Permissions, credentials, entry" },
  { id: "body",        name: "Body",         icon: "💪", description: "Physical capacity and energy" },
  { id: "skill",       name: "Skill",        icon: "🎯", description: "Competence and ability level" },
  { id: "connections", name: "Connections",  icon: "🤝", description: "People who can help" },
  { id: "environment", name: "Environment",  icon: "🏠", description: "Physical space and conditions" },
] as const;

export type PowerCategory = "time" | "money" | "tools" | "access" | "body" | "skill" | "connections" | "environment";

// ── Priority Scoring Weights ────────────────────────────────
export const PRIORITY_WEIGHTS = {
  deadlineProximity: 0.25,
  foundationNeglect: 0.20,
  desireStage: 0.15,
  powerCompleteness: 0.15,
  urgency: 0.15,
  rhythm: 0.10,
} as const;

// ── Operational Modes ───────────────────────────────────────
export const MODES = ["life_os", "work_sprint"] as const;
export type OperationalMode = (typeof MODES)[number];
