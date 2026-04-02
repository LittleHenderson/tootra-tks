// ═══════════════════════════════════════════════════════════════
// TKS Inversion Engine — 29 Canonical Inversion Modes
// ═══════════════════════════════════════════════════════════════

// ── Inversion Layers ───────────────────────────────────────────
export type InversionLayer = "surface" | "deep_structural" | "meta_layer" | "special";

export const INVERSION_LAYERS: Record<InversionLayer, { label: string; color: string; description: string }> = {
  surface: {
    label: "Surface",
    color: "#C8962E",
    description: "Story-level inversions — easy to understand, transform the narrative meaning",
  },
  deep_structural: {
    label: "Deep Structural",
    color: "#2B4E9B",
    description: "Operate directly on the TKS math engine — transform internal symbolic structure",
  },
  meta_layer: {
    label: "Meta-Layer",
    color: "#9C27B0",
    description: "Neurocognitive inversions — transform how the subject perceives and relates to the scenario",
  },
  special: {
    label: "Special",
    color: "#E03C1F",
    description: "Advanced transformations — modify system-level properties like constraints and boundaries",
  },
};

// ── Axis Definitions ───────────────────────────────────────────
export type InversionAxis =
  | "noetic"
  | "element"
  | "world"
  | "foundation"
  | "subFoundation"
  | "acquisition"
  | "causal"
  | "narrativeRole"
  | "scalarValence";

export const INVERSION_AXES: Record<InversionAxis, { label: string; shortCode: string; description: string }> = {
  noetic: { label: "Noetic", shortCode: "AX_NOE", description: "The 10 noetics (ν₀–ν₉) — internal motivation and thought quality" },
  element: { label: "Element", shortCode: "AX_ELM", description: "The 40 elements (A0–F9) — symbolic building blocks" },
  world: { label: "World", shortCode: "AX_WRLD", description: "The 4 worlds (A/B/C/D) — spiritual, mental, emotional, physical" },
  foundation: { label: "Foundation", shortCode: "AX_FND", description: "The 7 foundations (F1–F7) — life domains" },
  subFoundation: { label: "Sub-Foundation", shortCode: "AX_SFND", description: "The 28 sub-foundations — micro life areas" },
  acquisition: { label: "Acquisition", shortCode: "AX_ACQ", description: "The 22 acquisitions — how resources are obtained" },
  causal: { label: "Causal", shortCode: "AX_CAUS", description: "Causal chain direction and structure" },
  narrativeRole: { label: "Narrative Role", shortCode: "AX_ROLE", description: "Agent/patient/subject/object roles in the story" },
  scalarValence: { label: "Scalar Valence", shortCode: "AX_SVAL", description: "Intensity and polarity magnitude" },
};

// ── Axis Operation Types ───────────────────────────────────────
export type AxisOperation = "OPP" | "DUAL" | "CP" | "ID" | "PERM" | "MIR" | "RC" | "COMP" | "FLIP" | "SCALE" | "SWAP" | "REMAP";

export const AXIS_OPERATIONS: Record<AxisOperation, string> = {
  OPP: "Opposite-table lookup",
  DUAL: "Dual-table lookup (cross-world)",
  CP: "Counter-pole flip (±)",
  ID: "Identity (no-op)",
  PERM: "Permutation / structural reshuffle",
  MIR: "Mirror / reverse order",
  RC: "Reverse causal recomputation",
  COMP: "Complement lookup",
  FLIP: "Binary flip (up↔down, in↔out)",
  SCALE: "Scalar inversion (low↔high)",
  SWAP: "Role/agent swap",
  REMAP: "Remap to parallel domain",
};

// ── Intensity Levels ───────────────────────────────────────────
export type InversionIntensity = "soft" | "medium" | "hard";

export const INVERSION_INTENSITIES: Record<InversionIntensity, { label: string; description: string; icon: string }> = {
  soft: { label: "Soft", description: "Minimal symbol switches — preserve as much structure as possible", icon: "🌊" },
  medium: { label: "Medium", description: "Balanced transformation — meaningful change while keeping coherence", icon: "⚡" },
  hard: { label: "Hard", description: "Full opposite universe — apply all applicable transform rules", icon: "🔥" },
};

// ── Scope Levels ───────────────────────────────────────────────
export type InversionScope = "local" | "term" | "chain" | "equation" | "scenario";

export const INVERSION_SCOPES: Record<InversionScope, { label: string; description: string }> = {
  local: { label: "Local", description: "Invert a single symbol" },
  term: { label: "Term", description: "Invert one term (e.g., a sum)" },
  chain: { label: "Chain", description: "Invert a full causal chain" },
  equation: { label: "Equation", description: "Invert the entire equation" },
  scenario: { label: "Scenario", description: "Invert the whole story/scenario semantic layer" },
};

// ── Inversion Mode Definition ──────────────────────────────────
export interface InversionMode {
  id: number;
  key: string;
  name: string;
  layer: InversionLayer;
  description: string;
  shortDescription: string;
  axisGrid: Partial<Record<InversionAxis, AxisOperation>>;
  example: {
    before: string;
    after: string;
  };
}

// ── All 29 Inversion Modes ─────────────────────────────────────
export const INVERSION_MODES: InversionMode[] = [
  // ═══ LAYER 1: Surface Inversions (6) ═══
  {
    id: 1,
    key: "opposite",
    name: "Opposite",
    layer: "surface",
    description: "Maps each symbol to its semantic opposite. Resources become losses, joy becomes pain, presence becomes absence. The causal chain is preserved — only the meaning of each node flips.",
    shortDescription: "Flip every symbol to its semantic opposite",
    axisGrid: { noetic: "OPP", element: "OPP", world: "OPP", foundation: "OPP", subFoundation: "OPP", acquisition: "OPP", causal: "ID", narrativeRole: "ID", scalarValence: "OPP" },
    example: { before: "Resources + strategic move → cognitive alignment", after: "Resource loss + confusion → cognitive dissonance" },
  },
  {
    id: 2,
    key: "dual",
    name: "Dual",
    layer: "surface",
    description: "Maps each symbol to its dual element in the cross-world duality table. Spiritual (A) ↔ Physical (D), Mental (B) ↔ Emotional (C). The entire world-context of the equation shifts to its complementary domain.",
    shortDescription: "Swap to the complementary world domain",
    axisGrid: { noetic: "ID", element: "DUAL", world: "DUAL", foundation: "DUAL", subFoundation: "DUAL", acquisition: "ID", causal: "ID", narrativeRole: "ID", scalarValence: "ID" },
    example: { before: "Spiritual insight → mental clarity", after: "Physical sensation → emotional response" },
  },
  {
    id: 3,
    key: "counter_pole",
    name: "Counter-Pole",
    layer: "surface",
    description: "Flips the hierarchical pole inside each element's own sense-layer. Every constructive form becomes destructive, and vice versa. The element identity is preserved — only its direction flips.",
    shortDescription: "Flip constructive ↔ destructive within each element",
    axisGrid: { noetic: "CP", element: "CP", world: "ID", foundation: "ID", subFoundation: "CP", acquisition: "CP", causal: "ID", narrativeRole: "ID", scalarValence: "CP" },
    example: { before: "Nurturing feminine energy", after: "Withholding feminine energy" },
  },
  {
    id: 4,
    key: "mirror",
    name: "Mirror",
    layer: "surface",
    description: "Mirrors the structural pattern of the equation across the causal axis. The sequence reverses but symbol identity is preserved. Effects become causes, conclusions become premises.",
    shortDescription: "Reverse the causal sequence, keep symbols",
    axisGrid: { noetic: "ID", element: "ID", world: "ID", foundation: "ID", subFoundation: "ID", acquisition: "ID", causal: "MIR", narrativeRole: "ID", scalarValence: "ID" },
    example: { before: "She reacts because of him", after: "He reacts because of her" },
  },
  {
    id: 5,
    key: "reverse_causal",
    name: "Reverse-Causal",
    layer: "surface",
    description: "Reverses the causal order while re-computing dependencies using Acquisition rules. Unlike Mirror, this changes the semantic functions — not just the order. Causes become effects with new meaning.",
    shortDescription: "Reverse causal order and recompute dependencies",
    axisGrid: { noetic: "RC", element: "RC", world: "ID", foundation: "ID", subFoundation: "ID", acquisition: "RC", causal: "RC", narrativeRole: "SWAP", scalarValence: "ID" },
    example: { before: "My fear leads to withdrawal", after: "Withdrawal produces fear" },
  },
  {
    id: 6,
    key: "parallel_analogue",
    name: "Parallel Analogue",
    layer: "surface",
    description: "Replaces all symbols with their analogous structure in a parallel world or domain. The structural pattern and causal flow are preserved, but the element families change according to the analogue map (Love → Money, Health → Power, etc.).",
    shortDescription: "Same skeleton, different life domain",
    axisGrid: { noetic: "ID", element: "REMAP", world: "REMAP", foundation: "REMAP", subFoundation: "REMAP", acquisition: "REMAP", causal: "ID", narrativeRole: "ID", scalarValence: "ID" },
    example: { before: "Love/affection scenario", after: "Resource/wealth abundance analogue" },
  },

  // ═══ LAYER 2: Deep Structural Inversions (12) ═══
  {
    id: 7,
    key: "noetic_complement",
    name: "Noetic Complement",
    layer: "deep_structural",
    description: "Each noetic is replaced by its complement: ν₁↔ν₉, ν₂↔ν₈, ν₃↔ν₇, ν₄↔ν₆. ν₅ (center) is self-inverse. This inverts the internal motivation driving each symbol in the equation.",
    shortDescription: "Swap each noetic with its complement pair",
    axisGrid: { noetic: "COMP", element: "ID", world: "ID", foundation: "ID", subFoundation: "ID", acquisition: "ID", causal: "ID", narrativeRole: "ID", scalarValence: "ID" },
    example: { before: "Driven by Mind (ν₁) and Positive (ν₂)", after: "Driven by Effect (ν₉) and Rhythm (ν₈)" },
  },
  {
    id: 8,
    key: "acquisition_polarity",
    name: "Acquisition Polarity",
    layer: "deep_structural",
    description: "Every Acquisition flips between its constructive and destructive form. Cross-plane acquisitions become in-plane and vice versa. How you obtain resources completely reverses.",
    shortDescription: "Flip constructive ↔ destructive acquisition modes",
    axisGrid: { noetic: "ID", element: "ID", world: "ID", foundation: "ID", subFoundation: "ID", acquisition: "CP", causal: "ID", narrativeRole: "ID", scalarValence: "CP" },
    example: { before: "Constructive resource gathering", after: "Destructive resource depletion" },
  },
  {
    id: 9,
    key: "foundation_flip",
    name: "Foundation Flip",
    layer: "deep_structural",
    description: "Each Foundation flips to its opposing pair: F1 (Self/Stability) ↔ F4 (Relationships/Freedom), F2 (Knowledge) ↔ F6 (Environment), F3 (Body) ↔ F7 (Spirit). F5 (Work) is the pivot.",
    shortDescription: "Swap each foundation with its opposing pair",
    axisGrid: { noetic: "ID", element: "ID", world: "ID", foundation: "FLIP", subFoundation: "FLIP", acquisition: "ID", causal: "ID", narrativeRole: "ID", scalarValence: "ID" },
    example: { before: "Self-identity task (F1)", after: "Relationship/freedom task (F4)" },
  },
  {
    id: 10,
    key: "sub_foundation_reversal",
    name: "Sub-Foundation Reversal",
    layer: "deep_structural",
    description: "Each sub-foundation shifts to its micro-pole opposite within the same foundation. The 2-4 micro-poles within each sub-foundation reverse their directional flow.",
    shortDescription: "Reverse micro-poles within sub-foundations",
    axisGrid: { noetic: "ID", element: "ID", world: "ID", foundation: "ID", subFoundation: "CP", acquisition: "ID", causal: "ID", narrativeRole: "ID", scalarValence: "ID" },
    example: { before: "Sub-foundation 6d (Errands/maintenance)", after: "Sub-foundation 6a (Home/sanctuary)" },
  },
  {
    id: 11,
    key: "domain_permutation",
    name: "Domain Permutation",
    layer: "deep_structural",
    description: "Goes beyond simple duality — permutes the 4 worlds (A/B/C/D) in any valid combination. Not just A↔D and B↔C, but A↔B, A↔C, B↔D, etc. Creates 12 possible domain inversions.",
    shortDescription: "Permute worlds beyond simple duality",
    axisGrid: { noetic: "ID", element: "PERM", world: "PERM", foundation: "PERM", subFoundation: "PERM", acquisition: "ID", causal: "ID", narrativeRole: "ID", scalarValence: "ID" },
    example: { before: "Spiritual (A) + Physical (D) scenario", after: "Mental (B) + Emotional (C) scenario" },
  },
  {
    id: 12,
    key: "temporal",
    name: "Temporal Inversion",
    layer: "deep_structural",
    description: "Two types: order-reversal (backwards sequence) and future-to-past collapse (attractor-first calibration). Useful for reverse foreshadowing logic — seeing the end before the beginning.",
    shortDescription: "Reverse temporal sequence or collapse future to past",
    axisGrid: { noetic: "ID", element: "ID", world: "ID", foundation: "ID", subFoundation: "ID", acquisition: "ID", causal: "MIR", narrativeRole: "ID", scalarValence: "ID" },
    example: { before: "Plan → Execute → Review", after: "Review → Execute → Plan" },
  },
  {
    id: 13,
    key: "scalar",
    name: "Scalar Inversion",
    layer: "deep_structural",
    description: "If a sense has magnitude (pain intensity, suppression depth, resource abundance), invert the scale: low → high, high → low. Creates emotional anti-mirror effects.",
    shortDescription: "Invert intensity: low ↔ high magnitude",
    axisGrid: { noetic: "ID", element: "ID", world: "ID", foundation: "ID", subFoundation: "ID", acquisition: "ID", causal: "ID", narrativeRole: "ID", scalarValence: "SCALE" },
    example: { before: "Mild discomfort, low urgency", after: "Intense agony, critical urgency" },
  },
  {
    id: 14,
    key: "structural_permutation",
    name: "Structural Permutation",
    layer: "deep_structural",
    description: "Full permutation group (S₃) acting on the equation chain. A 3-element chain has 6 possible orderings (XYZ, XZY, YXZ, YZX, ZXY, ZYX). Used for symbolic recomposition of scenarios.",
    shortDescription: "Permute all elements in the equation chain",
    axisGrid: { noetic: "ID", element: "PERM", world: "ID", foundation: "ID", subFoundation: "ID", acquisition: "ID", causal: "PERM", narrativeRole: "PERM", scalarValence: "ID" },
    example: { before: "Resources → Cognition → Identity", after: "Identity → Resources → Cognition" },
  },
  {
    id: 15,
    key: "context_frame",
    name: "Contextual Frame",
    layer: "deep_structural",
    description: "Inverts the contextual domain: love-frame ↔ survival-frame, money-frame ↔ respect-frame, health-frame ↔ identity-frame. The story remains the same but the meaning becomes radically different.",
    shortDescription: "Same story, radically different meaning frame",
    axisGrid: { noetic: "ID", element: "REMAP", world: "REMAP", foundation: "REMAP", subFoundation: "REMAP", acquisition: "ID", causal: "ID", narrativeRole: "ID", scalarValence: "ID" },
    example: { before: "Love-frame: giving to partner", after: "Survival-frame: protecting resources" },
  },
  {
    id: 16,
    key: "motivational",
    name: "Motivational Inversion",
    layer: "deep_structural",
    description: "Swaps the agent of the action. Who does what to whom reverses. Same symbols, but roles are swapped — the subject becomes the object and vice versa.",
    shortDescription: "Swap who acts on whom",
    axisGrid: { noetic: "ID", element: "ID", world: "ID", foundation: "ID", subFoundation: "ID", acquisition: "ID", causal: "ID", narrativeRole: "SWAP", scalarValence: "ID" },
    example: { before: "Woman hides money from partner", after: "Partner hides money from woman" },
  },
  {
    id: 17,
    key: "polarity",
    name: "Polarity Inversion",
    layer: "deep_structural",
    description: "Global sign flip — everything positive becomes negative, and vice versa. Joy→Pain, Stability→Chaos, Nurture→Neglect. This is the most dramatic single-axis inversion.",
    shortDescription: "Global positive ↔ negative sign flip",
    axisGrid: { noetic: "OPP", element: "OPP", world: "ID", foundation: "ID", subFoundation: "ID", acquisition: "OPP", causal: "ID", narrativeRole: "ID", scalarValence: "OPP" },
    example: { before: "Joyful abundance, stable growth", after: "Painful scarcity, unstable decline" },
  },
  {
    id: 18,
    key: "causal_density",
    name: "Causal Density",
    layer: "deep_structural",
    description: "Dense causal chains become sparse, and sparse become dense. Used for predicting or undoing attractor collapse. A tightly coupled chain loosens; a loose chain tightens.",
    shortDescription: "Dense ↔ sparse causal coupling",
    axisGrid: { noetic: "ID", element: "ID", world: "ID", foundation: "ID", subFoundation: "ID", acquisition: "ID", causal: "SCALE", narrativeRole: "ID", scalarValence: "SCALE" },
    example: { before: "Tightly coupled chain: A→B→C→D", after: "Loosely coupled: A...B...C...D" },
  },

  // ═══ LAYER 3: Meta-Layer Inversions (7) ═══
  {
    id: 19,
    key: "attention",
    name: "Attention Inversion",
    layer: "meta_layer",
    description: "What the subject focuses on becomes background, and what was background becomes the focus. The spotlight shifts — the hidden becomes visible, the obvious becomes invisible.",
    shortDescription: "Foreground ↔ background focus swap",
    axisGrid: { noetic: "FLIP", element: "ID", world: "ID", foundation: "ID", subFoundation: "ID", acquisition: "ID", causal: "ID", narrativeRole: "SWAP", scalarValence: "FLIP" },
    example: { before: "Focus on career success, ignore health", after: "Focus on health, career fades to background" },
  },
  {
    id: 20,
    key: "value",
    name: "Value Inversion",
    layer: "meta_layer",
    description: "Inverts the value weight of each element: what is valued becomes feared, what is feared becomes valued. The emotional charge attached to each symbol reverses completely.",
    shortDescription: "Valued ↔ feared emotional charge flip",
    axisGrid: { noetic: "OPP", element: "ID", world: "ID", foundation: "ID", subFoundation: "ID", acquisition: "ID", causal: "ID", narrativeRole: "ID", scalarValence: "OPP" },
    example: { before: "Woman (valued, desired)", after: "Woman (feared, avoided)" },
  },
  {
    id: 21,
    key: "desire_inhibition",
    name: "Desire-Inhibition",
    layer: "meta_layer",
    description: "Desire (C2) converts to Inhibition (C1) and vice versa. What you want becomes what you suppress; what you suppress becomes what you crave. The wanting/blocking axis flips.",
    shortDescription: "Desire ↔ inhibition conversion",
    axisGrid: { noetic: "OPP", element: "CP", world: "ID", foundation: "ID", subFoundation: "ID", acquisition: "CP", causal: "ID", narrativeRole: "ID", scalarValence: "ID" },
    example: { before: "Strong desire to connect", after: "Strong inhibition against connection" },
  },
  {
    id: 22,
    key: "expectation",
    name: "Expectation Inversion",
    layer: "meta_layer",
    description: "Expected outcomes flip: expect good → expect bad, expect bad → expect good. Used in scenario prediction to explore what happens when assumptions reverse.",
    shortDescription: "Expected success ↔ expected failure",
    axisGrid: { noetic: "OPP", element: "ID", world: "ID", foundation: "ID", subFoundation: "ID", acquisition: "ID", causal: "ID", narrativeRole: "ID", scalarValence: "OPP" },
    example: { before: "Expecting promotion and recognition", after: "Expecting demotion and obscurity" },
  },
  {
    id: 23,
    key: "attractor",
    name: "Attractor Inversion",
    layer: "meta_layer",
    description: "Inverts the attractor vector: pull → push, push → pull. What draws you in now repels you; what repelled you now attracts. The gravitational field of the scenario reverses.",
    shortDescription: "Pull ↔ push attractor reversal",
    axisGrid: { noetic: "FLIP", element: "ID", world: "ID", foundation: "ID", subFoundation: "ID", acquisition: "FLIP", causal: "ID", narrativeRole: "ID", scalarValence: "FLIP" },
    example: { before: "Drawn toward comfort and security", after: "Repelled by comfort, drawn to risk" },
  },
  {
    id: 24,
    key: "stability",
    name: "Stability Inversion",
    layer: "meta_layer",
    description: "Stable structural patterns become unstable, and unstable become stable. Used in scenario collapse mapping — what was solid ground becomes quicksand, and vice versa.",
    shortDescription: "Stable ↔ unstable pattern flip",
    axisGrid: { noetic: "ID", element: "ID", world: "ID", foundation: "FLIP", subFoundation: "FLIP", acquisition: "ID", causal: "ID", narrativeRole: "ID", scalarValence: "FLIP" },
    example: { before: "Stable career, predictable routine", after: "Volatile career, chaotic schedule" },
  },
  {
    id: 25,
    key: "entropy",
    name: "Entropy Inversion",
    layer: "meta_layer",
    description: "Structured becomes chaotic, chaotic becomes structured. Order dissolves into randomness, or randomness crystallizes into order. The information density of the scenario flips.",
    shortDescription: "Order ↔ chaos transformation",
    axisGrid: { noetic: "ID", element: "PERM", world: "ID", foundation: "ID", subFoundation: "ID", acquisition: "PERM", causal: "PERM", narrativeRole: "ID", scalarValence: "ID" },
    example: { before: "Organized plan with clear steps", after: "Chaotic situation with no clear path" },
  },

  // ═══ LAYER 4: Special Transformations (4) ═══
  {
    id: 26,
    key: "constraint",
    name: "Constraint Inversion",
    layer: "special",
    description: "Hard constraints become soft, soft become hard. What was non-negotiable becomes flexible; what was optional becomes mandatory. The rigidity of the system inverts.",
    shortDescription: "Hard ↔ soft constraint boundaries",
    axisGrid: { noetic: "ID", element: "ID", world: "ID", foundation: "ID", subFoundation: "ID", acquisition: "FLIP", causal: "ID", narrativeRole: "ID", scalarValence: "FLIP" },
    example: { before: "Must finish by Friday (hard deadline)", after: "Flexible timeline, no fixed deadline" },
  },
  {
    id: 27,
    key: "boundary",
    name: "Boundary Inversion",
    layer: "special",
    description: "Internal boundaries become external, and external become internal. Personal limits become social expectations; social norms become personal convictions.",
    shortDescription: "Internal ↔ external boundary swap",
    axisGrid: { noetic: "FLIP", element: "ID", world: "DUAL", foundation: "ID", subFoundation: "ID", acquisition: "ID", causal: "ID", narrativeRole: "SWAP", scalarValence: "ID" },
    example: { before: "Personal boundary: I won't overwork", after: "Social expectation: the team won't overwork" },
  },
  {
    id: 28,
    key: "semantic_parity",
    name: "Semantic Parity",
    layer: "special",
    description: "Replaces each element with its parity-preserving analog — same type, different meaning. Generates 'same skeleton, new story' transformations. The structure is identical but the content shifts.",
    shortDescription: "Same skeleton, completely new story",
    axisGrid: { noetic: "ID", element: "REMAP", world: "ID", foundation: "REMAP", subFoundation: "REMAP", acquisition: "ID", causal: "ID", narrativeRole: "ID", scalarValence: "ID" },
    example: { before: "Love story with partner", after: "Mentorship story with teacher" },
  },
  {
    id: 29,
    key: "agent_role",
    name: "Agent-Role",
    layer: "special",
    description: "Derived from Motivational + Boundary/Context. The agent, patient, and observer roles all rotate. Who acts, who receives, and who watches all shift positions in the narrative.",
    shortDescription: "Rotate agent/patient/observer roles",
    axisGrid: { noetic: "ID", element: "ID", world: "ID", foundation: "ID", subFoundation: "ID", acquisition: "ID", causal: "ID", narrativeRole: "PERM", scalarValence: "ID" },
    example: { before: "I teach my student a lesson", after: "My student teaches me a lesson" },
  },
];

// ── Preset Recipes ─────────────────────────────────────────────
export interface InversionPreset {
  id: string;
  name: string;
  icon: string;
  description: string;
  modeIds: number[];
  intensity: InversionIntensity;
  scope: InversionScope;
  color: string;
}

export const INVERSION_PRESETS: InversionPreset[] = [
  {
    id: "shadow_self",
    name: "Shadow Self",
    icon: "🌑",
    description: "See your dark mirror — every positive becomes negative, every strength becomes weakness. The version of you that chose differently at every turn.",
    modeIds: [1, 17, 3],
    intensity: "hard",
    scope: "scenario",
    color: "#1a1a2e",
  },
  {
    id: "material_mirror",
    name: "Material Mirror",
    icon: "💎",
    description: "Transpose your spiritual/emotional scenario into the material/physical world. Love becomes wealth, insight becomes muscle, faith becomes real estate.",
    modeIds: [2, 6],
    intensity: "medium",
    scope: "equation",
    color: "#C8962E",
  },
  {
    id: "emotional_reversal",
    name: "Emotional Reversal",
    icon: "🔄",
    description: "Every desire becomes an inhibition, every attraction becomes repulsion. What you want becomes what you avoid. The emotional landscape inverts completely.",
    modeIds: [21, 23, 20],
    intensity: "hard",
    scope: "scenario",
    color: "#E91E63",
  },
  {
    id: "role_swap",
    name: "Role Swap",
    icon: "🎭",
    description: "You become them, they become you. The agent and patient switch places. See the scenario from the other side of every interaction.",
    modeIds: [16, 29, 4],
    intensity: "medium",
    scope: "scenario",
    color: "#7B1FA2",
  },
  {
    id: "time_reversal",
    name: "Time Reversal",
    icon: "⏪",
    description: "Run the scenario backwards. Effects become causes, conclusions become premises. See what the end looks like as a beginning.",
    modeIds: [5, 12, 4],
    intensity: "medium",
    scope: "chain",
    color: "#2B4E9B",
  },
  {
    id: "chaos_engine",
    name: "Chaos Engine",
    icon: "🌪️",
    description: "Maximum entropy. Structured plans dissolve, stable patterns shatter, dense connections scatter. See what emerges from the rubble.",
    modeIds: [25, 24, 18],
    intensity: "hard",
    scope: "scenario",
    color: "#D32F2F",
  },
  {
    id: "foundation_shift",
    name: "Foundation Shift",
    icon: "🏛️",
    description: "Move the entire scenario to its opposing life domain. Self ↔ Relationships, Knowledge ↔ Environment, Body ↔ Spirit. Same energy, different arena.",
    modeIds: [9, 10, 15],
    intensity: "medium",
    scope: "equation",
    color: "#388E3C",
  },
  {
    id: "inner_outer",
    name: "Inner ↔ Outer",
    icon: "🔮",
    description: "What's internal becomes external, what's external becomes internal. Personal boundaries become social norms, private fears become public challenges.",
    modeIds: [27, 19, 15],
    intensity: "soft",
    scope: "scenario",
    color: "#0288D1",
  },
  {
    id: "intensity_flip",
    name: "Intensity Flip",
    icon: "📊",
    description: "Mild becomes extreme, extreme becomes mild. Whispers become screams, crises become inconveniences. The volume knob of every element reverses.",
    modeIds: [13, 22, 17],
    intensity: "medium",
    scope: "equation",
    color: "#F57C00",
  },
  {
    id: "new_story",
    name: "New Story",
    icon: "📖",
    description: "Same structural skeleton, completely different narrative. The architecture of your scenario is preserved but filled with entirely new content.",
    modeIds: [28, 6, 11],
    intensity: "soft",
    scope: "scenario",
    color: "#795548",
  },
];

// ── Helper Functions ───────────────────────────────────────────
export function getInversionMode(id: number): InversionMode | undefined {
  return INVERSION_MODES.find(m => m.id === id);
}

export function getInversionModeByKey(key: string): InversionMode | undefined {
  return INVERSION_MODES.find(m => m.key === key);
}

export function getModesByLayer(layer: InversionLayer): InversionMode[] {
  return INVERSION_MODES.filter(m => m.layer === layer);
}

export function getPreset(id: string): InversionPreset | undefined {
  return INVERSION_PRESETS.find(p => p.id === id);
}

export function getPresetModes(preset: InversionPreset): InversionMode[] {
  return preset.modeIds.map(id => getInversionMode(id)).filter((m): m is InversionMode => m !== undefined);
}

// ── Inversion Dial Config (for LLM) ───────────────────────────
export interface InversionDialConfig {
  modeIds: number[];
  axes: Partial<Record<InversionAxis, boolean>>;
  intensity: InversionIntensity;
  scope: InversionScope;
  direction: "forward" | "backward" | "bidirectional";
  targetProfile?: {
    enable: boolean;
    fromFoundation?: string;
    toFoundation?: string;
    fromWorld?: string;
    toWorld?: string;
  };
}

export function buildDefaultDialConfig(modeIds: number[]): InversionDialConfig {
  // Merge axis grids from all selected modes
  const axes: Partial<Record<InversionAxis, boolean>> = {};
  for (const id of modeIds) {
    const mode = getInversionMode(id);
    if (mode) {
      for (const [axis, op] of Object.entries(mode.axisGrid)) {
        if (op !== "ID") {
          axes[axis as InversionAxis] = true;
        }
      }
    }
  }
  return {
    modeIds,
    axes,
    intensity: "medium",
    scope: "equation",
    direction: "forward",
  };
}
