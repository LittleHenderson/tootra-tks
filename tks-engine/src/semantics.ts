import { AcquisitionId, FoundationId, IdeaId, NoeticIndex } from "./types";

// ============================================================================
// FOUNDATION SEMANTICS
// ============================================================================

export type WorldId = "A" | "B" | "C" | "D";
export type AxisId = "a" | "b" | "c" | "d";

export interface FoundationMeta {
  id: FoundationId;
  label: string;
  world: WorldId;
  axis: AxisId;
  shortDescription: string;
  acquisitionTriad: AcquisitionId[];
  defaultFractals: NoeticIndex[][];
}

const STANDARD_PATTERN: NoeticIndex[] = [1, 4, 7];
const AWARE_RHYTHM: NoeticIndex[] = [1, 7];
const VIBRATE_RHYTHM: NoeticIndex[] = [4, 7];

export const FOUNDATIONS: Record<FoundationId, FoundationMeta> = {
  "1a": { id: "1a", label: "Spirit – Origin spark", world: "A", axis: "a", shortDescription: "Root intent issuing from source.", acquisitionTriad: [], defaultFractals: [STANDARD_PATTERN] },
  "2a": { id: "2a", label: "Soul – Desire vector", world: "A", axis: "a", shortDescription: "Personal longing shaped toward a path.", acquisitionTriad: [], defaultFractals: [STANDARD_PATTERN] },
  "3a": { id: "3a", label: "Mind – Clarity beam", world: "B", axis: "a", shortDescription: "Clean perception and focused thought.", acquisitionTriad: [], defaultFractals: [STANDARD_PATTERN] },
  "4a": { id: "4a", label: "Heart – Devotional fire", world: "B", axis: "a", shortDescription: "Love-backed intent that warms action.", acquisitionTriad: [], defaultFractals: [STANDARD_PATTERN] },
  "5a": { id: "5a", label: "Voice – Expression line", world: "C", axis: "a", shortDescription: "Clear broadcast of inner signal.", acquisitionTriad: [], defaultFractals: [STANDARD_PATTERN] },
  "6a": { id: "6a", label: "Vision – Pattern sight", world: "C", axis: "a", shortDescription: "Seeing structure and flow before form.", acquisitionTriad: [], defaultFractals: [STANDARD_PATTERN] },
  "7a": { id: "7a", label: "Crown – Coherence field", world: "D", axis: "a", shortDescription: "Unifying all layers into one rhythm.", acquisitionTriad: [], defaultFractals: [STANDARD_PATTERN] },

  "1b": { id: "1b", label: "Spirit – Trust in current", world: "A", axis: "b", shortDescription: "Faith that source flow sustains.", acquisitionTriad: [], defaultFractals: [STANDARD_PATTERN] },
  "2b": { id: "2b", label: "Soul – Surrender", world: "A", axis: "b", shortDescription: "Yielding will to wiser motion.", acquisitionTriad: [], defaultFractals: [STANDARD_PATTERN] },
  "3b": { id: "3b", label: "Mind – Discernment", world: "B", axis: "b", shortDescription: "Choosing signal over noise.", acquisitionTriad: [], defaultFractals: [STANDARD_PATTERN] },
  "4b": { id: "4b", label: "Heart – Compassion", world: "B", axis: "b", shortDescription: "Holding others within the field of care.", acquisitionTriad: [], defaultFractals: [STANDARD_PATTERN] },
  "5b": { id: "5b", label: "Voice – Counsel", world: "C", axis: "b", shortDescription: "Speaking guidance without coercion.", acquisitionTriad: [], defaultFractals: [STANDARD_PATTERN] },
  "6b": { id: "6b", label: "Vision – Perspective", world: "C", axis: "b", shortDescription: "Seeing multiple frames to avoid bias.", acquisitionTriad: [], defaultFractals: [STANDARD_PATTERN] },
  "7b": { id: "7b", label: "Crown – Silence", world: "D", axis: "b", shortDescription: "Quiet mind that receives timing.", acquisitionTriad: [], defaultFractals: [AWARE_RHYTHM] },

  "1c": { id: "1c", label: "Spirit – Courage", world: "A", axis: "c", shortDescription: "Bold entrance into the work.", acquisitionTriad: [], defaultFractals: [STANDARD_PATTERN] },
  "2c": { id: "2c", label: "Soul – Direction", world: "A", axis: "c", shortDescription: "Choosing a line and holding it.", acquisitionTriad: [], defaultFractals: [STANDARD_PATTERN] },
  "3c": { id: "3c", label: "Mind – Strategy", world: "B", axis: "c", shortDescription: "Sequencing moves that respect reality.", acquisitionTriad: [], defaultFractals: [STANDARD_PATTERN] },
  "4c": { id: "4c", label: "Heart – Resolve", world: "B", axis: "c", shortDescription: "Staying with love when it is hard.", acquisitionTriad: [], defaultFractals: [STANDARD_PATTERN] },
  "5c": { id: "5c", label: "Voice – Command", world: "C", axis: "c", shortDescription: "Stating boundaries and charges cleanly.", acquisitionTriad: [], defaultFractals: [STANDARD_PATTERN] },
  "6c": { id: "6c", label: "Vision – Stewardship", world: "C", axis: "c", shortDescription: "Managing flows with integrity.", acquisitionTriad: [], defaultFractals: [VIBRATE_RHYTHM] },
  "7c": { id: "7c", label: "Crown – Sovereignty", world: "D", axis: "c", shortDescription: "Maintaining rightful rule of self.", acquisitionTriad: [], defaultFractals: [AWARE_RHYTHM] },

  "1d": { id: "1d", label: "Body – Vital spark", world: "A", axis: "d", shortDescription: "Baseline life-force ignition.", acquisitionTriad: [], defaultFractals: [STANDARD_PATTERN] },
  "2d": { id: "2d", label: "Body – Immunity", world: "A", axis: "d", shortDescription: "Defensive resilience to stressors.", acquisitionTriad: [], defaultFractals: [STANDARD_PATTERN] },
  "3d": { id: "3d", label: "Body – Health", world: "B", axis: "d", shortDescription: "Physical vitality and recovery.", acquisitionTriad: [], defaultFractals: [STANDARD_PATTERN] },
  "4d": { id: "4d", label: "Body – Movement", world: "B", axis: "d", shortDescription: "Graceful coordinated motion.", acquisitionTriad: [], defaultFractals: [STANDARD_PATTERN] },
  "5d": { id: "5d", label: "World – Influence", world: "C", axis: "d", shortDescription: "Power to steer systems and people.", acquisitionTriad: [13, 14, 15], defaultFractals: [STANDARD_PATTERN] },
  "6d": { id: "6d", label: "World – Material flow", world: "C", axis: "d", shortDescription: "Money, goods, sustenance flows.", acquisitionTriad: [16, 17, 18], defaultFractals: [STANDARD_PATTERN] },
  "7d": { id: "7d", label: "World – Legacy", world: "D", axis: "d", shortDescription: "Imprint that remains after you.", acquisitionTriad: [], defaultFractals: [AWARE_RHYTHM] },
};

// ============================================================================
// ACQUISITION SEMANTICS
// ============================================================================

export type AcquisitionKind = "Desire" | "Wisdom" | "Power" | "Meta";

export interface AcquisitionMeta {
  id: AcquisitionId;
  label: string;
  kind: AcquisitionKind;
  linkedFoundations: FoundationId[];
  shortDescription: string;
}

export const ACQUISITIONS: Record<AcquisitionId, AcquisitionMeta> = {
  1: { id: 1, label: "DESIRE – spark", kind: "Desire", linkedFoundations: ["1a", "1d"], shortDescription: "Initial wanting to enter the work." },
  2: { id: 2, label: "WISDOM – orientation", kind: "Wisdom", linkedFoundations: ["2a"], shortDescription: "Knowing where to face and why." },
  3: { id: 3, label: "POWER – capacity", kind: "Power", linkedFoundations: ["3a"], shortDescription: "Energy to act without collapse." },
  4: { id: 4, label: "META – endurance", kind: "Meta", linkedFoundations: ["7a"], shortDescription: "Keeping rhythm over long arcs." },
  5: { id: 5, label: "DESIRE – purity", kind: "Desire", linkedFoundations: ["4a"], shortDescription: "Unmixed motive aligned to source." },
  6: { id: 6, label: "WISDOM – balance", kind: "Wisdom", linkedFoundations: ["6b"], shortDescription: "Choosing equilibrium across pulls." },
  7: { id: 7, label: "POWER – assertion", kind: "Power", linkedFoundations: ["5c"], shortDescription: "Healthy force to state and hold." },
  8: { id: 8, label: "META – timing", kind: "Meta", linkedFoundations: ["7b"], shortDescription: "Sensing right moment and pacing." },
  9: { id: 9, label: "DESIRE – devotion", kind: "Desire", linkedFoundations: ["4b"], shortDescription: "Heart-backed willingness to stay." },
  10: { id: 10, label: "WISDOM – mapping", kind: "Wisdom", linkedFoundations: ["3c"], shortDescription: "Plotting sequences that fit reality." },
  11: { id: 11, label: "POWER – protection", kind: "Power", linkedFoundations: ["2d"], shortDescription: "Shielding the work and body." },
  12: { id: 12, label: "META – reflection", kind: "Meta", linkedFoundations: ["1b"], shortDescription: "Learning loop that refines aim." },
  13: { id: 13, label: "DESIRE – worldly power", kind: "Desire", linkedFoundations: ["5d"], shortDescription: "Wanting influence and control ethically." },
  14: { id: 14, label: "WISDOM – worldly power", kind: "Wisdom", linkedFoundations: ["5d"], shortDescription: "Knowing how systems and hierarchy move." },
  15: { id: 15, label: "POWER – worldly power", kind: "Power", linkedFoundations: ["5d"], shortDescription: "Force to steer outcomes without tyranny." },
  16: { id: 16, label: "DESIRE – material flow", kind: "Desire", linkedFoundations: ["6d"], shortDescription: "Wanting resources, food, money cleanly." },
  17: { id: 17, label: "WISDOM – material flow", kind: "Wisdom", linkedFoundations: ["6d"], shortDescription: "Understanding markets and provisioning." },
  18: { id: 18, label: "POWER – material flow", kind: "Power", linkedFoundations: ["6d"], shortDescription: "Ability to generate and hold assets." },
  19: { id: 19, label: "META – gratitude", kind: "Meta", linkedFoundations: ["7d"], shortDescription: "Stabilizing appreciation that opens gates." },
  20: { id: 20, label: "DESIRE – service", kind: "Desire", linkedFoundations: ["4b", "5b"], shortDescription: "Drive to uplift community." },
  21: { id: 21, label: "WISDOM – integration", kind: "Wisdom", linkedFoundations: ["7c"], shortDescription: "Weaving lessons back into self-rule." },
  22: { id: 22, label: "META – transcend", kind: "Meta", linkedFoundations: ["7d"], shortDescription: "Stepping beyond the current cycle." },
};

// ============================================================================
// IDEA / ELEMENT SEMANTICS
// ============================================================================

export interface IdeaMeta {
  id: IdeaId;
  world: WorldId;
  label: string;
  shortDescription: string;
  relatedFoundations: FoundationId[];
}

export const IDEAS: Record<IdeaId, IdeaMeta> = {
  A1: { id: "A1", world: "A", label: "A1 – Spirit seed", shortDescription: "First impulse from source.", relatedFoundations: ["1a", "1d"] },
  A2: { id: "A2", world: "A", label: "A2 – Inner flame", shortDescription: "Desire ignition and warmth.", relatedFoundations: ["2a", "4a"] },
  A3: { id: "A3", world: "A", label: "A3 – Breath of intent", shortDescription: "Sustaining the charge of will.", relatedFoundations: ["1b", "3a"] },
  A4: { id: "A4", world: "A", label: "A4 – Alignment", shortDescription: "Bringing spirit and soul into line.", relatedFoundations: ["2b", "7a"] },
  A5: { id: "A5", world: "A", label: "A5 – Purity", shortDescription: "Cleansing motives and channels.", relatedFoundations: ["5a"] },
  A6: { id: "A6", world: "A", label: "A6 – Call", shortDescription: "Hearing the summons of path.", relatedFoundations: ["6a", "6b"] },
  A7: { id: "A7", world: "A", label: "A7 – Devotion arc", shortDescription: "Long vow to the work.", relatedFoundations: ["4b", "7b"] },
  A8: { id: "A8", world: "A", label: "A8 – Trust field", shortDescription: "Resting in unseen support.", relatedFoundations: ["1b"] },
  A9: { id: "A9", world: "A", label: "A9 – Silence", shortDescription: "Stillness that reveals signal.", relatedFoundations: ["7b"] },
  A10: { id: "A10", world: "A", label: "A10 – Benediction", shortDescription: "Blessing current over the path.", relatedFoundations: ["7a", "7d"] },

  B1: { id: "B1", world: "B", label: "B1 – Mental calm", shortDescription: "Settled mind to perceive clearly.", relatedFoundations: ["3a", "7b"] },
  B2: { id: "B2", world: "B", label: "B2 – Mind uplift", shortDescription: "Positive mental charge.", relatedFoundations: ["3a", "3b"] },
  B3: { id: "B3", world: "B", label: "B3 – Discern filter", shortDescription: "Separating truth from noise.", relatedFoundations: ["3b"] },
  B4: { id: "B4", world: "B", label: "B4 – Empathy", shortDescription: "Feeling others without drowning.", relatedFoundations: ["4b"] },
  B5: { id: "B5", world: "B", label: "B5 – Moral axis", shortDescription: "Ethical compass under choice.", relatedFoundations: ["5b"] },
  B6: { id: "B6", world: "B", label: "B6 – Creative imagining", shortDescription: "Seeing possible shapes.", relatedFoundations: ["6a", "6b"] },
  B7: { id: "B7", world: "B", label: "B7 – Emotional steadiness", shortDescription: "Regulating affect in motion.", relatedFoundations: ["2b", "4c"] },
  B8: { id: "B8", world: "B", label: "B8 – Reframing", shortDescription: "Changing lens to unlock choice.", relatedFoundations: ["6b"] },
  B9: { id: "B9", world: "B", label: "B9 – Communion", shortDescription: "Shared mindspace and rapport.", relatedFoundations: ["4b", "5b"] },
  B10: { id: "B10", world: "B", label: "B10 – Beacon", shortDescription: "Holding a mental lighthouse.", relatedFoundations: ["7a", "7c"] },

  C1: { id: "C1", world: "C", label: "C1 – Spark of action", shortDescription: "First move into the world.", relatedFoundations: ["1c", "1d"] },
  C2: { id: "C2", world: "C", label: "C2 – Direction of force", shortDescription: "Aimed effort with feedback.", relatedFoundations: ["2c"] },
  C3: { id: "C3", world: "C", label: "C3 – Creative pulse", shortDescription: "Inventive flow shaping form.", relatedFoundations: ["5c", "5d"] },
  C4: { id: "C4", world: "C", label: "C4 – Rhythm", shortDescription: "Repeated cycles that build momentum.", relatedFoundations: ["7c"] },
  C5: { id: "C5", world: "C", label: "C5 – Collaboration", shortDescription: "Working with others cleanly.", relatedFoundations: ["5b", "5c"] },
  C6: { id: "C6", world: "C", label: "C6 – Stewardship", shortDescription: "Careful tending of resources.", relatedFoundations: ["6c", "6d"] },
  C7: { id: "C7", world: "C", label: "C7 – Boundary", shortDescription: "Defining edges to protect flow.", relatedFoundations: ["5c", "2d"] },
  C8: { id: "C8", world: "C", label: "C8 – Integration", shortDescription: "Weaving inputs into one effort.", relatedFoundations: ["7c"] },
  C9: { id: "C9", world: "C", label: "C9 – Outreach", shortDescription: "Extending signal to networks.", relatedFoundations: ["5a", "5d"] },
  C10: { id: "C10", world: "C", label: "C10 – Harvest", shortDescription: "Gathering results of cycles.", relatedFoundations: ["6d", "7d"] },

  D1: { id: "D1", world: "D", label: "D1 – Grounding", shortDescription: "Stability and embodiment.", relatedFoundations: ["1d"] },
  D2: { id: "D2", world: "D", label: "D2 – Protection", shortDescription: "Shields for body and work.", relatedFoundations: ["2d"] },
  D3: { id: "D3", world: "D", label: "D3 – Embodied force", shortDescription: "Grounded drive through the body.", relatedFoundations: ["3d"] },
  D4: { id: "D4", world: "D", label: "D4 – Flow state", shortDescription: "Effortless action in motion.", relatedFoundations: ["4d"] },
  D5: { id: "D5", world: "D", label: "D5 – Influence channel", shortDescription: "Command projected into structures.", relatedFoundations: ["5d"] },
  D6: { id: "D6", world: "D", label: "D6 – Wealth current", shortDescription: "Channels of material flow.", relatedFoundations: ["6d"] },
  D7: { id: "D7", world: "D", label: "D7 – Legacy trace", shortDescription: "Imprint that remains beyond you.", relatedFoundations: ["7d"] },
  D8: { id: "D8", world: "D", label: "D8 – Continuity", shortDescription: "Keeping cycles alive across time.", relatedFoundations: ["7a", "7d"] },
  D9: { id: "D9", world: "D", label: "D9 – Renewal", shortDescription: "Resetting forms without losing essence.", relatedFoundations: ["3d", "7d"] },
  D10: { id: "D10", world: "D", label: "D10 – Completion", shortDescription: "Sealing the work and closing loops.", relatedFoundations: ["7d"] },
};

// ============================================================================
// NOETIC PATTERN SUGGESTIONS
// ============================================================================

export interface NoeticPattern {
  id: string;
  sequence: NoeticIndex[];
  label: string;
  shortDescription: string;
}

export const NOETIC_PATTERNS: NoeticPattern[] = [
  {
    id: "avr",
    sequence: [1, 4, 7],
    label: "Aware-Vibrate-Rhythm",
    shortDescription: "Standard pattern: bring awareness, energize, then entrain into rhythm.",
  },
  {
    id: "ar",
    sequence: [1, 7],
    label: "Aware-Rhythm",
    shortDescription: "Simple settle: tag with awareness then lock rhythm.",
  },
  {
    id: "vr",
    sequence: [4, 7],
    label: "Vibrate-Rhythm",
    shortDescription: "Charge the value then stabilize into cadence.",
  },
  {
    id: "id",
    sequence: [0],
    label: "Identity",
    shortDescription: "Noetic 0: hold value without additional modulation.",
  },
];

// ============================================================================
// HELPERS
// ============================================================================

function sequencesEqual(a: NoeticIndex[], b: NoeticIndex[]): boolean {
  if (a.length !== b.length) return false;
  return a.every((v, i) => v === b[i]);
}

export function getDefaultPatternsForFoundation(id: FoundationId): NoeticPattern[] {
  const meta = FOUNDATIONS[id];
  if (!meta || !meta.defaultFractals) return [];
  const matches: NoeticPattern[] = [];
  for (const seq of meta.defaultFractals) {
    const pattern = NOETIC_PATTERNS.find((p) => sequencesEqual(p.sequence, seq));
    if (pattern && !matches.includes(pattern)) {
      matches.push(pattern);
    }
  }
  return matches;
}
