# TKS INVERSION DIAL SPECIFICATION v1.0

## Multi-Agent Controlled Inversion System

**Document:** TKS_Inversion_Dial_Spec_v1.0.md
**Version:** 1.0
**Date:** 2025-12-11
**System:** TKS-Inversion-Dial (Multi-Agent)
**Canonical Source:** TKS v7.4+

---

# SECTION 1: INVERSION DIAL CONFIGURATION

## 1.1 Dial Config Object Schema

```typescript
interface TKS_InversionDialConfig {
  // Primary inversion mode (required)
  mode: InversionMode;

  // Axis flags - which layers are affected
  axes: {
    noetic: boolean;        // AX_NOE
    element: boolean;       // AX_ELM
    world: boolean;         // AX_WRLD
    foundation: boolean;    // AX_FND
    subFoundation: boolean; // AX_SFND
    acquisition: boolean;   // AX_ACQ
    causal: boolean;        // AX_CAUS
    narrativeRole: boolean; // AX_ROLE
    scalarValence: boolean; // AX_SVAL
  };

  // Transform intensity
  intensity: "soft" | "medium" | "hard";

  // Scope of application
  scope: "local" | "term" | "chain" | "equation" | "scenario";

  // Direction of inversion
  direction: "forward" | "backward" | "bidirectional";

  // Target profile for domain mapping modes
  targetProfile: {
    enable: boolean;
    fromFoundation?: Foundation;
    toFoundation?: Foundation;
    fromWorld?: World;
    toWorld?: World;
    customMap?: Record<string, string>;
  };
}
```

## 1.2 Parameter Definitions

### 1.2.1 Mode

The primary inversion mode must be one of the canonical 29 names:

```typescript
type InversionMode =
  // Surface layer (1-6)
  | "Opposite"           // 1
  | "Dual"               // 2
  | "CounterPole"        // 3
  | "Mirror"             // 4
  | "ReverseCausal"      // 5
  | "ParallelAnalogue"   // 6

  // Deep structure (7-18)
  | "NoeticComplement"   // 7
  | "AcquisitionPolarity"// 8
  | "FoundationFlip"     // 9
  | "SubFoundationReversal" // 10
  | "DomainPermutation"  // 11
  | "TemporalInversion"  // 12
  | "ScalarInversion"    // 13
  | "StructuralPermutation" // 14
  | "ContextFrame"       // 15
  | "Motivational"       // 16
  | "Polarity"           // 17
  | "CausalDensity"      // 18

  // Meta-layer (19-25)
  | "Attention"          // 19
  | "Value"              // 20
  | "DesireInhibition"   // 21
  | "Expectation"        // 22
  | "Attractor"          // 23
  | "Stability"          // 24
  | "Entropy"            // 25

  // Special (26-29)
  | "Constraint"         // 26
  | "Boundary"           // 27
  | "SemanticParity"     // 28
  | "AgentRole";         // 29 (derived: Motivational + Boundary/Context)
```

### 1.2.2 Axes

Each axis flag determines whether that layer participates in the inversion:

| Axis | Short Code | Description | Affected Symbols |
|------|------------|-------------|------------------|
| `noetic` | AX_NOE | Noetic indices (0-9) | ν₀-ν₉ superscripts |
| `element` | AX_ELM | Element identities | A0-D9 base symbols |
| `world` | AX_WRLD | World domains | A↔D, B↔C mappings |
| `foundation` | AX_FND | Foundation layer | F₁-F₇ |
| `subFoundation` | AX_SFND | Sub-Foundation layer | SF_{m,w} |
| `acquisition` | AX_ACQ | Acquisition layer | A₀, D_m, W_m, P_m |
| `causal` | AX_CAUS | Causal structure | → arrows, ∘ sequences |
| `narrativeRole` | AX_ROLE | Narrative semantics | Agent/Patient, Subject/Object |
| `scalarValence` | AX_SVAL | Intensity/polarity | +/-, high/low markers |

**Behavior:** If `axes[X] = false`, that axis is treated as `ID` (identity/no-op) regardless of grid lookup.

### 1.2.3 Intensity

Controls how many sub-rules fire for the selected mode:

| Level | Description | Behavior |
|-------|-------------|----------|
| `soft` | Minimal transform | Apply only primary rule; preserve maximum structure |
| `medium` | Balanced transform | Apply primary + secondary rules |
| `hard` | Full transform | Apply all applicable rules for mode |

**Example for Opposite mode:**
- `soft`: Only flip 2↔3 (Positive↔Negative)
- `medium`: Flip 2↔3, 5↔6 (add Female↔Male)
- `hard`: Flip 2↔3, 5↔6, 8↔9 (full Noetic opposite)

### 1.2.4 Scope

Determines what portion of the equation is inverted:

| Scope | Target | Description |
|-------|--------|-------------|
| `local` | Single symbol | One Xn element only |
| `term` | Single term | A compound like `(X +_T Y)` |
| `chain` | Causal chain | `X → Y → Z` sequence |
| `equation` | Full equation | Entire TKS expression |
| `scenario` | Story layer | Equation + META/narrative semantics |

### 1.2.5 Direction

| Direction | Behavior |
|-----------|----------|
| `forward` | Apply Inv(E) |
| `backward` | Apply Inv⁻¹(E) (same as Inv for involutions) |
| `bidirectional` | Return both E and Inv(E) |

### 1.2.6 Target Profile

Used only for modes that perform domain mapping:
- `ParallelAnalogue`
- `ContextFrame`
- `DomainPermutation`

```typescript
interface TargetProfile {
  enable: boolean;
  fromFoundation?: "F1" | "F2" | "F3" | "F4" | "F5" | "F6" | "F7";
  toFoundation?: "F1" | "F2" | "F3" | "F4" | "F5" | "F6" | "F7";
  fromWorld?: "A" | "B" | "C" | "D";
  toWorld?: "A" | "B" | "C" | "D";
  customMap?: Record<string, string>;  // e.g., {"C2": "D0", "D5": "D6"}
}
```

**Standard Mappings:**

| Name | From | To | Description |
|------|------|----|-------------|
| Love→Money | F₄ | F₆ | Relationship → Material |
| Health→Power | F₃ | F₅ | Vitality → Control |
| Wisdom→Material | F₂ | F₆ | Knowledge → Resources |
| Unity→Lust | F₁ | F₇ | Spiritual → Physical creation |
| Emotion→Intellect | C | B | Emotional → Mental world |
| Spirit→Physical | A | D | Spiritual → Physical world |

---

# SECTION 2: CANONICAL MODE ENUMERATION

## 2.1 Fixed Mode Names

**IMMUTABLE** — These names must never be changed in internal code:

```
╔════════════════════════════════════════════════════════════════╗
║  #  │ MODE NAME              │ CLASS         │ SYMBOL          ║
╠════════════════════════════════════════════════════════════════╣
║  1  │ Opposite               │ Surface       │ ⊖               ║
║  2  │ Dual                   │ Surface       │ ⊗               ║
║  3  │ CounterPole            │ Surface       │ ⊕               ║
║  4  │ Mirror                 │ Surface       │ ⟷               ║
║  5  │ ReverseCausal          │ Surface       │ ⟲               ║
║  6  │ ParallelAnalogue       │ Surface       │ ∥               ║
╠════════════════════════════════════════════════════════════════╣
║  7  │ NoeticComplement       │ Deep          │ ν̄               ║
║  8  │ AcquisitionPolarity    │ Deep          │ 𝔄±              ║
║  9  │ FoundationFlip         │ Deep          │ F↔              ║
║ 10  │ SubFoundationReversal  │ Deep          │ SF⟲             ║
║ 11  │ DomainPermutation      │ Deep          │ D⟳              ║
║ 12  │ TemporalInversion      │ Deep          │ T⁻¹             ║
║ 13  │ ScalarInversion        │ Deep          │ S±              ║
║ 14  │ StructuralPermutation  │ Deep          │ Σ⟳              ║
║ 15  │ ContextFrame           │ Deep          │ CF⁻¹            ║
║ 16  │ Motivational           │ Deep          │ M⁻¹             ║
║ 17  │ Polarity               │ Deep          │ P±              ║
║ 18  │ CausalDensity          │ Deep          │ CD±             ║
╠════════════════════════════════════════════════════════════════╣
║ 19  │ Attention              │ Meta          │ Att⁻¹           ║
║ 20  │ Value                  │ Meta          │ Val⁻¹           ║
║ 21  │ DesireInhibition       │ Meta          │ DI⁻¹            ║
║ 22  │ Expectation            │ Meta          │ Exp⁻¹           ║
║ 23  │ Attractor              │ Meta          │ Attr⁻¹          ║
║ 24  │ Stability              │ Meta          │ Stab⁻¹          ║
║ 25  │ Entropy                │ Meta          │ Ent⁻¹           ║
╠════════════════════════════════════════════════════════════════╣
║ 26  │ Constraint             │ Special       │ Con⁻¹           ║
║ 27  │ Boundary               │ Special       │ Bnd⁻¹           ║
║ 28  │ SemanticParity         │ Special       │ SP±             ║
║ 29  │ AgentRole              │ Special       │ AR⁻¹            ║
╚════════════════════════════════════════════════════════════════╝
```

## 2.2 Mode Validation Function

```typescript
const CANONICAL_MODES = [
  "Opposite", "Dual", "CounterPole", "Mirror", "ReverseCausal", "ParallelAnalogue",
  "NoeticComplement", "AcquisitionPolarity", "FoundationFlip", "SubFoundationReversal",
  "DomainPermutation", "TemporalInversion", "ScalarInversion", "StructuralPermutation",
  "ContextFrame", "Motivational", "Polarity", "CausalDensity",
  "Attention", "Value", "DesireInhibition", "Expectation", "Attractor", "Stability", "Entropy",
  "Constraint", "Boundary", "SemanticParity", "AgentRole"
] as const;

function validateMode(mode: string): boolean {
  return CANONICAL_MODES.includes(mode as InversionMode);
}

function getModeIndex(mode: InversionMode): number {
  return CANONICAL_MODES.indexOf(mode) + 1;
}
```

---

# SECTION 3: TOTAL INVERSION GRID

## 3.1 Axis Short Codes

```typescript
enum AxisCode {
  AX_NOE  = "noetic",
  AX_ELM  = "element",
  AX_WRLD = "world",
  AX_FND  = "foundation",
  AX_SFND = "subFoundation",
  AX_ACQ  = "acquisition",
  AX_CAUS = "causal",
  AX_ROLE = "narrativeRole",
  AX_SVAL = "scalarValence"
}
```

## 3.2 Operation Codes

| Code | Operation | Description |
|------|-----------|-------------|
| `OPP` | Opposite-table | Use polarity/complement inversion table |
| `DUAL` | Dual-table | Use world-swap dual table (A↔D, B↔C) |
| `CP` | Counter-pole table | Use +/- pole flip table |
| `ID` | Identity | No operation (pass through) |
| `PERM` | Permutation | Structural/order permutation |
| `MIR` | Mirror | Reverse order left↔right |
| `RC` | Reverse-causal | Reverse + transform roles |
| `PAR` | Parallel-analogue | Domain translation table |
| `META` | Meta-layer | Narrative/weight adjustment (no symbol change) |
| `DER` | Derived | Computed from other layers |

## 3.3 Mode × Axis Grid (Routing Table)

```
╔═══════════════════════╦═══════╦═══════╦═══════╦═══════╦═══════╦═══════╦═══════╦═══════╦═══════╗
║ MODE                  ║AX_NOE ║AX_ELM ║AX_WRLD║AX_FND ║AX_SFND║AX_ACQ ║AX_CAUS║AX_ROLE║AX_SVAL║
╠═══════════════════════╬═══════╬═══════╬═══════╬═══════╬═══════╬═══════╬═══════╬═══════╬═══════╣
║ 1  Opposite           ║  OPP  ║  OPP  ║  OPP  ║  OPP  ║  OPP  ║  OPP  ║  ID   ║  ID   ║  OPP  ║
║ 2  Dual               ║  ID   ║  DUAL ║  DUAL ║  ID   ║  ID   ║  ID   ║  ID   ║  ID   ║  ID   ║
║ 3  CounterPole        ║  ID   ║  CP   ║  ID   ║  CP   ║  CP   ║  CP   ║  ID   ║  ID   ║  CP   ║
║ 4  Mirror             ║  ID   ║  ID   ║  ID   ║  ID   ║  ID   ║  ID   ║  MIR  ║  META ║  ID   ║
║ 5  ReverseCausal      ║  ID   ║  ID   ║  ID   ║  ID   ║  ID   ║  ID   ║  RC   ║  META ║  ID   ║
║ 6  ParallelAnalogue   ║  ID   ║  PAR  ║  PAR  ║  PAR  ║  ID   ║  ID   ║  ID   ║  META ║  ID   ║
╠═══════════════════════╬═══════╬═══════╬═══════╬═══════╬═══════╬═══════╬═══════╬═══════╬═══════╣
║ 7  NoeticComplement   ║  OPP  ║  ID   ║  ID   ║  ID   ║  ID   ║  ID   ║  ID   ║  ID   ║  ID   ║
║ 8  AcquisitionPolarity║  ID   ║  ID   ║  ID   ║  ID   ║  ID   ║  OPP  ║  ID   ║  ID   ║  ID   ║
║ 9  FoundationFlip     ║  ID   ║  ID   ║  ID   ║  OPP  ║  OPP  ║  ID   ║  ID   ║  ID   ║  ID   ║
║ 10 SubFoundReversal   ║  ID   ║  ID   ║  ID   ║  ID   ║  OPP  ║  ID   ║  ID   ║  ID   ║  ID   ║
║ 11 DomainPermutation  ║  ID   ║  ID   ║  PERM ║  ID   ║  ID   ║  ID   ║  ID   ║  ID   ║  ID   ║
║ 12 TemporalInversion  ║  ID   ║  ID   ║  ID   ║  ID   ║  ID   ║  ID   ║  PERM ║  META ║  ID   ║
║ 13 ScalarInversion    ║  ID   ║  ID   ║  ID   ║  ID   ║  ID   ║  ID   ║  ID   ║  ID   ║  OPP  ║
║ 14 StructPermutation  ║  ID   ║  ID   ║  ID   ║  ID   ║  ID   ║  ID   ║  PERM ║  META ║  ID   ║
║ 15 ContextFrame       ║  ID   ║  PAR  ║  PAR  ║  PAR  ║  ID   ║  ID   ║  ID   ║  META ║  ID   ║
║ 16 Motivational       ║  ID   ║  ID   ║  ID   ║  ID   ║  ID   ║  ID   ║  ID   ║  OPP  ║  ID   ║
║ 17 Polarity           ║  ID   ║  OPP  ║  ID   ║  OPP  ║  OPP  ║  OPP  ║  ID   ║  ID   ║  OPP  ║
║ 18 CausalDensity      ║  ID   ║  ID   ║  ID   ║  ID   ║  ID   ║  ID   ║  META ║  META ║  ID   ║
╠═══════════════════════╬═══════╬═══════╬═══════╬═══════╬═══════╬═══════╬═══════╬═══════╬═══════╣
║ 19 Attention          ║  META ║  META ║  META ║  META ║  META ║  META ║  META ║  META ║  META ║
║ 20 Value              ║  META ║  META ║  META ║  META ║  META ║  META ║  META ║  META ║  META ║
║ 21 DesireInhibition   ║  OPP  ║  ID   ║  ID   ║  ID   ║  ID   ║  ID   ║  META ║  META ║  OPP  ║
║ 22 Expectation        ║  META ║  META ║  META ║  META ║  META ║  META ║  META ║  META ║  META ║
║ 23 Attractor          ║  META ║  ID   ║  ID   ║  ID   ║  ID   ║  ID   ║  META ║  META ║  META ║
║ 24 Stability          ║  META ║  META ║  META ║  META ║  META ║  META ║  META ║  META ║  META ║
║ 25 Entropy            ║  META ║  META ║  META ║  META ║  META ║  META ║  META ║  META ║  META ║
╠═══════════════════════╬═══════╬═══════╬═══════╬═══════╬═══════╬═══════╬═══════╬═══════╬═══════╣
║ 26 Constraint         ║  META ║  META ║  META ║  META ║  META ║  META ║  META ║  META ║  META ║
║ 27 Boundary           ║  META ║  META ║  META ║  META ║  META ║  META ║  META ║  OPP  ║  META ║
║ 28 SemanticParity     ║  ID   ║  PAR  ║  ID   ║  PAR  ║  PAR  ║  PAR  ║  ID   ║  META ║  ID   ║
║ 29 AgentRole          ║  ID   ║  ID   ║  ID   ║  ID   ║  ID   ║  ID   ║  ID   ║  OPP  ║  META ║
╚═══════════════════════╩═══════╩═══════╩═══════╩═══════╩═══════╩═══════╩═══════╩═══════╩═══════╝
```

## 3.4 Grid Lookup Function

```typescript
type OpCode = "OPP" | "DUAL" | "CP" | "ID" | "PERM" | "MIR" | "RC" | "PAR" | "META" | "DER";

const INVERSION_GRID: Record<InversionMode, Record<AxisCode, OpCode>> = {
  // Surface
  "Opposite":           { noetic: "OPP",  element: "OPP",  world: "OPP",  foundation: "OPP",  subFoundation: "OPP",  acquisition: "OPP",  causal: "ID",   narrativeRole: "ID",   scalarValence: "OPP"  },
  "Dual":               { noetic: "ID",   element: "DUAL", world: "DUAL", foundation: "ID",   subFoundation: "ID",   acquisition: "ID",   causal: "ID",   narrativeRole: "ID",   scalarValence: "ID"   },
  "CounterPole":        { noetic: "ID",   element: "CP",   world: "ID",   foundation: "CP",   subFoundation: "CP",   acquisition: "CP",   causal: "ID",   narrativeRole: "ID",   scalarValence: "CP"   },
  "Mirror":             { noetic: "ID",   element: "ID",   world: "ID",   foundation: "ID",   subFoundation: "ID",   acquisition: "ID",   causal: "MIR",  narrativeRole: "META", scalarValence: "ID"   },
  "ReverseCausal":      { noetic: "ID",   element: "ID",   world: "ID",   foundation: "ID",   subFoundation: "ID",   acquisition: "ID",   causal: "RC",   narrativeRole: "META", scalarValence: "ID"   },
  "ParallelAnalogue":   { noetic: "ID",   element: "PAR",  world: "PAR",  foundation: "PAR",  subFoundation: "ID",   acquisition: "ID",   causal: "ID",   narrativeRole: "META", scalarValence: "ID"   },

  // Deep Structure
  "NoeticComplement":   { noetic: "OPP",  element: "ID",   world: "ID",   foundation: "ID",   subFoundation: "ID",   acquisition: "ID",   causal: "ID",   narrativeRole: "ID",   scalarValence: "ID"   },
  "AcquisitionPolarity":{ noetic: "ID",   element: "ID",   world: "ID",   foundation: "ID",   subFoundation: "ID",   acquisition: "OPP",  causal: "ID",   narrativeRole: "ID",   scalarValence: "ID"   },
  "FoundationFlip":     { noetic: "ID",   element: "ID",   world: "ID",   foundation: "OPP",  subFoundation: "OPP",  acquisition: "ID",   causal: "ID",   narrativeRole: "ID",   scalarValence: "ID"   },
  "SubFoundationReversal":{ noetic: "ID", element: "ID",   world: "ID",   foundation: "ID",   subFoundation: "OPP",  acquisition: "ID",   causal: "ID",   narrativeRole: "ID",   scalarValence: "ID"   },
  "DomainPermutation":  { noetic: "ID",   element: "ID",   world: "PERM", foundation: "ID",   subFoundation: "ID",   acquisition: "ID",   causal: "ID",   narrativeRole: "ID",   scalarValence: "ID"   },
  "TemporalInversion":  { noetic: "ID",   element: "ID",   world: "ID",   foundation: "ID",   subFoundation: "ID",   acquisition: "ID",   causal: "PERM", narrativeRole: "META", scalarValence: "ID"   },
  "ScalarInversion":    { noetic: "ID",   element: "ID",   world: "ID",   foundation: "ID",   subFoundation: "ID",   acquisition: "ID",   causal: "ID",   narrativeRole: "ID",   scalarValence: "OPP"  },
  "StructuralPermutation":{ noetic: "ID", element: "ID",   world: "ID",   foundation: "ID",   subFoundation: "ID",   acquisition: "ID",   causal: "PERM", narrativeRole: "META", scalarValence: "ID"   },
  "ContextFrame":       { noetic: "ID",   element: "PAR",  world: "PAR",  foundation: "PAR",  subFoundation: "ID",   acquisition: "ID",   causal: "ID",   narrativeRole: "META", scalarValence: "ID"   },
  "Motivational":       { noetic: "ID",   element: "ID",   world: "ID",   foundation: "ID",   subFoundation: "ID",   acquisition: "ID",   causal: "ID",   narrativeRole: "OPP",  scalarValence: "ID"   },
  "Polarity":           { noetic: "ID",   element: "OPP",  world: "ID",   foundation: "OPP",  subFoundation: "OPP",  acquisition: "OPP",  causal: "ID",   narrativeRole: "ID",   scalarValence: "OPP"  },
  "CausalDensity":      { noetic: "ID",   element: "ID",   world: "ID",   foundation: "ID",   subFoundation: "ID",   acquisition: "ID",   causal: "META", narrativeRole: "META", scalarValence: "ID"   },

  // Meta-Layer
  "Attention":          { noetic: "META", element: "META", world: "META", foundation: "META", subFoundation: "META", acquisition: "META", causal: "META", narrativeRole: "META", scalarValence: "META" },
  "Value":              { noetic: "META", element: "META", world: "META", foundation: "META", subFoundation: "META", acquisition: "META", causal: "META", narrativeRole: "META", scalarValence: "META" },
  "DesireInhibition":   { noetic: "OPP",  element: "ID",   world: "ID",   foundation: "ID",   subFoundation: "ID",   acquisition: "ID",   causal: "META", narrativeRole: "META", scalarValence: "OPP"  },
  "Expectation":        { noetic: "META", element: "META", world: "META", foundation: "META", subFoundation: "META", acquisition: "META", causal: "META", narrativeRole: "META", scalarValence: "META" },
  "Attractor":          { noetic: "META", element: "ID",   world: "ID",   foundation: "ID",   subFoundation: "ID",   acquisition: "ID",   causal: "META", narrativeRole: "META", scalarValence: "META" },
  "Stability":          { noetic: "META", element: "META", world: "META", foundation: "META", subFoundation: "META", acquisition: "META", causal: "META", narrativeRole: "META", scalarValence: "META" },
  "Entropy":            { noetic: "META", element: "META", world: "META", foundation: "META", subFoundation: "META", acquisition: "META", causal: "META", narrativeRole: "META", scalarValence: "META" },

  // Special
  "Constraint":         { noetic: "META", element: "META", world: "META", foundation: "META", subFoundation: "META", acquisition: "META", causal: "META", narrativeRole: "META", scalarValence: "META" },
  "Boundary":           { noetic: "META", element: "META", world: "META", foundation: "META", subFoundation: "META", acquisition: "META", causal: "META", narrativeRole: "OPP",  scalarValence: "META" },
  "SemanticParity":     { noetic: "ID",   element: "PAR",  world: "ID",   foundation: "PAR",  subFoundation: "PAR",  acquisition: "PAR",  causal: "ID",   narrativeRole: "META", scalarValence: "ID"   },
  "AgentRole":          { noetic: "ID",   element: "ID",   world: "ID",   foundation: "ID",   subFoundation: "ID",   acquisition: "ID",   causal: "ID",   narrativeRole: "OPP",  scalarValence: "META" }
};

function getAxisOperation(mode: InversionMode, axis: AxisCode): OpCode {
  return INVERSION_GRID[mode][axis];
}
```

## 3.5 Grid Interpretation Rules

For each inversion mode and each axis AX_*:

1. **Check user flag:** If `dial.axes[axis] === false` → treat as `ID` (no-op)
2. **Lookup grid cell:** Get operation code from `INVERSION_GRID[mode][axis]`
3. **Apply operation:**

| OpCode | Action |
|--------|--------|
| `OPP` | Apply Opposite-table for that axis (polarity/complement swap) |
| `DUAL` | Apply Dual-table (A↔D, B↔C world swap) |
| `CP` | Apply Counter-pole table (+↔- flip) |
| `ID` | Pass through unchanged |
| `PERM` | Apply permutation logic (reorder, cycle, restructure) |
| `MIR` | Reverse sequence order left↔right |
| `RC` | Reverse causal direction + transform roles |
| `PAR` | Apply parallel-analogue mapping table using targetProfile |
| `META` | No raw symbol change; adjust narrative semantics/weights only |
| `DER` | Compute from other layers (context-dependent) |

---

# SECTION 4: MULTI-AGENT ARCHITECTURE

## 4.1 Agent Definitions

### Agent 1: Parser-Type-Agent

**Role:** Parse and type-check TKS equations

**Functions:**
```typescript
interface ParserTypeAgent {
  // Parse raw TKS string → AST
  parse(equation: string): TKS_AST;

  // Type-check against v7.4 TKS types
  typeCheck(ast: TKS_AST): TypeCheckResult;

  // Reject ill-typed equations
  validate(ast: TKS_AST): ValidationResult;
}
```

**Type-checking rules:**
- Verify all elements are from canonical 40-element set
- Verify all Noetics are ν₀-ν₉
- Verify Foundations are F₁-F₇
- Verify Sub-Foundations are SF_{m,w} where m∈{1..7}, w∈{a,b,c,d}
- Verify Acquisitions are from 22-acquisition set
- Verify operators are from TOOTRA set
- Check domain/codomain compatibility

---

### Agent 2: Dial-Resolver-Agent

**Role:** Build and validate InversionDialConfig

**Functions:**
```typescript
interface DialResolverAgent {
  // Parse user request → dial config
  resolve(userRequest: string): TKS_InversionDialConfig;

  // Validate mode against canonical list
  validateMode(mode: string): boolean;

  // Apply defaults for missing parameters
  applyDefaults(partial: Partial<TKS_InversionDialConfig>): TKS_InversionDialConfig;
}
```

**Default resolution:**
```typescript
const DEFAULT_DIAL: TKS_InversionDialConfig = {
  mode: "Opposite",
  axes: {
    noetic: true,
    element: true,
    world: true,
    foundation: true,
    subFoundation: true,
    acquisition: true,
    causal: false,
    narrativeRole: false,
    scalarValence: true
  },
  intensity: "soft",
  scope: "equation",
  direction: "forward",
  targetProfile: { enable: false }
};
```

---

### Agent 3: Grid-Engine-Agent

**Role:** Implement Mode × Axis routing

**Functions:**
```typescript
interface GridEngineAgent {
  // Generate transformation plan from dial config
  plan(dial: TKS_InversionDialConfig): TransformationPlan;

  // Lookup grid operation for mode+axis
  lookup(mode: InversionMode, axis: AxisCode): OpCode;

  // Apply axis flags to override grid
  applyFlags(plan: TransformationPlan, axes: AxesConfig): TransformationPlan;
}

interface TransformationPlan {
  mode: InversionMode;
  axisOps: Record<AxisCode, OpCode>;
  intensity: "soft" | "medium" | "hard";
  scope: Scope;
}
```

**Example plan output:**
```json
{
  "mode": "Opposite",
  "axisOps": {
    "noetic": "OPP",
    "element": "OPP",
    "world": "OPP",
    "foundation": "OPP",
    "subFoundation": "OPP",
    "acquisition": "OPP",
    "causal": "ID",
    "narrativeRole": "ID",
    "scalarValence": "OPP"
  },
  "intensity": "soft",
  "scope": "equation"
}
```

---

### Agent 4: Transform-Agent

**Role:** Execute symbol-level transformations

**Functions:**
```typescript
interface TransformAgent {
  // Apply transformation plan to AST
  transform(ast: TKS_AST, plan: TransformationPlan): TKS_AST;

  // Execute single axis operation
  transformAxis(ast: TKS_AST, axis: AxisCode, op: OpCode): TKS_AST;

  // Access inversion tables
  getTable(tableType: "Opposite" | "Dual" | "CounterPole" | "Parallel"): InversionTable;
}
```

**Transformation order (fixed):**
1. Noetics (superscripts)
2. Elements (base symbols)
3. Worlds (A/B/C/D)
4. Foundations (F₁-F₇)
5. Sub-Foundations (SF_{m,w})
6. Acquisitions (D_m, W_m, P_m)
7. Causal structure (→, ∘)
8. Narrative roles (Agent/Patient)
9. Scalar valence (+/-)

**Scope handling:**
```typescript
function applyByScope(ast: TKS_AST, scope: Scope, transform: Function): TKS_AST {
  switch(scope) {
    case "local":    return transformSelectedNode(ast, transform);
    case "term":     return transformTerm(ast, transform);
    case "chain":    return transformCausalChain(ast, transform);
    case "equation": return transformFullAST(ast, transform);
    case "scenario": return transformWithMeta(ast, transform);
  }
}
```

**Intensity handling:**
```typescript
function getIntensityRules(mode: InversionMode, intensity: Intensity): Rule[] {
  const allRules = MODE_RULES[mode];
  switch(intensity) {
    case "soft":   return allRules.slice(0, 1);  // Primary rule only
    case "medium": return allRules.slice(0, 3);  // Primary + secondary
    case "hard":   return allRules;               // All rules
  }
}
```

---

### Agent 5: Narrative-Agent

**Role:** Story-layer translation

**Functions:**
```typescript
interface NarrativeAgent {
  // Decode TKS → English story
  decode(ast: TKS_AST): string;

  // Encode English → TKS (for verification)
  encode(story: string): TKS_AST;

  // Generate change summary
  summarizeChanges(original: TKS_AST, transformed: TKS_AST): ChangeSummary;
}

interface ChangeSummary {
  polarity: string;      // "positive → negative"
  world: string;         // "emotional → mental"
  foundation: string;    // "love → money"
  causalDirection: string; // "forward → reversed"
  roles: string;         // "agent → patient"
  intensity: string;     // "high → low"
}
```

---

### Agent 6: Audit-Agent

**Role:** Compliance verification

**Functions:**
```typescript
interface AuditAgent {
  // Re-encode story → TKS
  reencode(story: string): TKS_AST;

  // Check type equality (modulo simplification)
  checkEquivalence(ast1: TKS_AST, ast2: TKS_AST): boolean;

  // Verify no illegal symbols
  checkSymbolCompliance(ast: TKS_AST): ComplianceResult;

  // Verify inversion semantics
  checkInversionSemantics(original: TKS_AST, transformed: TKS_AST, mode: InversionMode): boolean;

  // Full audit report
  audit(original: TKS_AST, transformed: TKS_AST, mode: InversionMode): AuditReport;
}
```

---

## 4.2 Workflow Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TKS INVERSION WORKFLOW                              │
└─────────────────────────────────────────────────────────────────────────────┘

USER REQUEST: "Invert this equation in Opposite mode, soft, equation scope."
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 0: CANON LOAD                                                          │
│ ─────────────────                                                           │
│ Load v7.4 TKS canon:                                                        │
│   • Symbol tables (40 Elements, 10 Noetics, etc.)                          │
│   • Type definitions                                                        │
│   • Element sense tables                                                    │
│   • Inversion tables (Opposite, Dual, Parallel, etc.)                      │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: PARSE & TYPE-CHECK [Parser-Type-Agent]                              │
│ ──────────────────────────────────────────────                              │
│ INPUT:  "(B5.2 +_T D3.2) → C3.1 → (D5.1 -_T D0.1_{6d})"                    │
│                                                                             │
│ ACTIONS:                                                                    │
│   1. Parse equation string → AST                                            │
│   2. Type-check each component:                                             │
│      • B5.2 : Mental.Female.AccumulatedKnowledge ✓                         │
│      • D3.2 : Physical.Negative.MaterialChaos ✓                            │
│      • C3.1 : Emotional.Negative.Fear ✓                                    │
│      • D5.1 : Physical.Female.Woman ✓                                      │
│      • D0.1 : Physical.Idea.Template ✓                                     │
│      • _{6d} : SubFoundation.Material.Physical ✓                           │
│   3. Verify domain/codomain: (B × D) → C → D ✓                             │
│                                                                             │
│ OUTPUT: Validated AST or ERROR                                              │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: RESOLVE DIAL [Dial-Resolver-Agent]                                  │
│ ─────────────────────────────────────────                                   │
│ INPUT:  User request + defaults                                             │
│                                                                             │
│ ACTIONS:                                                                    │
│   1. Parse mode: "Opposite" ✓                                               │
│   2. Parse intensity: "soft"                                                │
│   3. Parse scope: "equation"                                                │
│   4. Apply axis defaults for Opposite mode                                  │
│   5. Validate all parameters                                                │
│                                                                             │
│ OUTPUT: InversionDialConfig                                                 │
│   {                                                                         │
│     "mode": "Opposite",                                                     │
│     "axes": { noetic: true, element: true, world: true, ... },             │
│     "intensity": "soft",                                                    │
│     "scope": "equation",                                                    │
│     "direction": "forward",                                                 │
│     "targetProfile": { "enable": false }                                    │
│   }                                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: PLAN VIA GRID [Grid-Engine-Agent]                                   │
│ ────────────────────────────────────────                                    │
│ INPUT:  InversionDialConfig                                                 │
│                                                                             │
│ ACTIONS:                                                                    │
│   For each axis AX_*:                                                       │
│     1. Check if axes[AX] == false → mark operation = ID                     │
│     2. Else lookup INVERSION_GRID["Opposite"][AX]                           │
│                                                                             │
│ OUTPUT: TransformationPlan                                                  │
│   {                                                                         │
│     "mode": "Opposite",                                                     │
│     "axisOps": {                                                            │
│       "noetic": "OPP",                                                      │
│       "element": "OPP",                                                     │
│       "world": "OPP",                                                       │
│       "foundation": "OPP",                                                  │
│       "subFoundation": "OPP",                                               │
│       "acquisition": "OPP",                                                 │
│       "causal": "ID",                                                       │
│       "narrativeRole": "ID",                                                │
│       "scalarValence": "OPP"                                                │
│     },                                                                      │
│     "intensity": "soft",                                                    │
│     "scope": "equation"                                                     │
│   }                                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: APPLY TRANSFORMATIONS [Transform-Agent]                             │
│ ──────────────────────────────────────────────                              │
│ INPUT:  AST + TransformationPlan                                            │
│                                                                             │
│ ACTIONS (fixed order):                                                      │
│   1. Noetics:       5→6, 3→2 (OPP)                                          │
│   2. Elements:      B5→B6, D3→D2, C3→C2, D5→D6, D0→D0 (OPP)                │
│   3. Worlds:        (handled via element)                                   │
│   4. Foundations:   6→2 in subscript (OPP)                                  │
│   5. SubFoundations: _{6d}→_{2d} (OPP)                                      │
│   6. Acquisitions:  (none present)                                          │
│   7. Causal:        ID (no change)                                          │
│   8. NarrativeRole: ID (no change)                                          │
│   9. ScalarValence: (intensity markers adjusted)                            │
│                                                                             │
│ Scope = equation: Apply to full AST                                         │
│ Intensity = soft: Apply primary Noetic swap (2↔3) only                      │
│                                                                             │
│ OUTPUT: Transformed AST                                                     │
│   "(B6.2 +_T D2.2) → C2.1 → (D6.1 -_T D0.1_{2d})"                          │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: NARRATIVE DECODE [Narrative-Agent]                                  │
│ ─────────────────────────────────────────                                   │
│ INPUT:  Original AST + Transformed AST                                      │
│                                                                             │
│ ACTIONS:                                                                    │
│   1. Decode original:                                                       │
│      "Accumulated experiences combined with material instability            │
│       cause fear, which causes the woman to hide money."                    │
│                                                                             │
│   2. Decode transformed:                                                    │
│      "Learning combined with physical health causes joy,                    │
│       which causes the man to reveal knowledge."                            │
│                                                                             │
│   3. Summarize changes:                                                     │
│      • Polarity: negative → positive (C3→C2, D3→D2)                         │
│      • Gender: female → male (D5→D6, B5→B6)                                 │
│      • Foundation: material → wisdom (F6→F2)                                │
│      • Action: hiding → revealing (inferred)                                │
│                                                                             │
│ OUTPUT: NarrativeDecoding                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 6: AUDIT & RE-ENCODE [Audit-Agent]                                     │
│ ──────────────────────────────────────                                      │
│ INPUT:  Transformed story + Transformed AST                                 │
│                                                                             │
│ ACTIONS:                                                                    │
│   1. Re-encode Story₁ → EQ''                                                │
│      "(B6.2 +_T D2.2) → C2.1 → (D6.1 +_T D0.1_{2d})"                       │
│                                                                             │
│   2. Check type equality:                                                   │
│      EQ' vs EQ'' → DRIFT: "-_T" vs "+_T"                                    │
│                                                                             │
│   3. Resolve drift:                                                         │
│      "reveal" implies +_T (fusion), but original has -_T                    │
│      → Adjust narrative: "share" instead of "reveal"                        │
│                                                                             │
│   4. Verify no illegal symbols: ✓                                           │
│   5. Verify inversion semantics: ✓                                          │
│                                                                             │
│ OUTPUT: AuditReport { status: "PASSED", notes: [...] }                      │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 7: FINAL OUTPUT [Supervisor]                                           │
│ ────────────────────────────────                                            │
│                                                                             │
│ === INPUT EQUATION ===                                                      │
│ (B5.2 +_T D3.2) → C3.1 → (D5.1 -_T D0.1_{6d})                              │
│                                                                             │
│ === DIAL CONFIG USED ===                                                    │
│ mode: Opposite                                                              │
│ intensity: soft                                                             │
│ scope: equation                                                             │
│ axes: { noetic: true, element: true, world: true, ... }                    │
│                                                                             │
│ === TRANSFORMATION PLAN (GRID) ===                                          │
│ noetic: OPP, element: OPP, world: OPP, foundation: OPP, ...                │
│                                                                             │
│ === OUTPUT EQUATION (MODE: Opposite) ===                                    │
│ (B6.2 +_T D2.2) → C2.1 → (D6.1 -_T D0.1_{2d})                              │
│                                                                             │
│ === ENGLISH INTERPRETATIONS ===                                             │
│ Original: Accumulated experiences combined with material instability        │
│           cause fear, which causes the woman to hide money.                 │
│ Inverted: Learning combined with physical health causes joy,                │
│           which causes the man to share knowledge.                          │
│                                                                             │
│ === CHANGE SUMMARY ===                                                      │
│ - Polarity: negative → positive                                             │
│ - Gender: female → male                                                     │
│ - Foundation: material → wisdom                                             │
│ - Emotion: fear → joy                                                       │
│                                                                             │
│ === TYPE & CANON CHECK ===                                                  │
│ Status: PASSED                                                              │
│ Notes: Minor narrative adjustment for operator consistency                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# SECTION 5: WORKED EXAMPLE — PARALLEL ANALOGUE

## 5.1 User Request

```
INPUT_EQ: (B5.2 +_T D3.2) → C3.1 → (D5.1 -_T D0.1_{6d})

DIAL:
  mode: ParallelAnalogue
  axes: { element: true, foundation: true, world: true, narrativeRole: true }
  intensity: soft
  scope: equation
  targetProfile: { enable: true, fromFoundation: "F4", toFoundation: "F6" }
```

## 5.2 Step-by-Step Processing

### Step 0: Canon Load
```
✓ v7.4 symbol tables loaded
✓ Element sense tables loaded
✓ Inversion tables loaded
✓ Parallel-analogue mapping tables loaded
```

### Step 1: Parse & Type-Check
```
AST:
  CausalChain(
    Fusion(B5.2, D3.2),
    C3.1,
    Removal(D5.1, Subscripted(D0.1, SF_{6,d}))
  )

Type-check:
  B5.2 : Mental.Female ✓
  D3.2 : Physical.Negative ✓
  C3.1 : Emotional.Negative ✓
  D5.1 : Physical.Female ✓
  D0.1_{6d} : Physical.Idea @ Material.Physical ✓

STATUS: WELL-TYPED ✓
```

### Step 2: Resolve Dial
```json
{
  "mode": "ParallelAnalogue",
  "axes": {
    "noetic": false,
    "element": true,
    "foundation": true,
    "world": true,
    "subFoundation": false,
    "acquisition": false,
    "causal": false,
    "narrativeRole": true,
    "scalarValence": false
  },
  "intensity": "soft",
  "scope": "equation",
  "direction": "forward",
  "targetProfile": {
    "enable": true,
    "fromFoundation": "F4",
    "toFoundation": "F6"
  }
}
```

### Step 3: Grid Lookup
```
MODE: ParallelAnalogue

Grid lookup for each axis:
  noetic:        ID   (user flag = false → ID override)
  element:       PAR  (grid says PAR, flag = true)
  world:         PAR  (grid says PAR, flag = true)
  foundation:    PAR  (grid says PAR, flag = true)
  subFoundation: ID   (user flag = false → ID override)
  acquisition:   ID   (grid says ID)
  causal:        ID   (grid says ID)
  narrativeRole: META (grid says META, flag = true)
  scalarValence: ID   (user flag = false → ID override)

TransformationPlan:
{
  "mode": "ParallelAnalogue",
  "axisOps": {
    "noetic": "ID",
    "element": "PAR",
    "world": "PAR",
    "foundation": "PAR",
    "subFoundation": "ID",
    "acquisition": "ID",
    "causal": "ID",
    "narrativeRole": "META",
    "scalarValence": "ID"
  },
  "intensity": "soft",
  "scope": "equation"
}
```

### Step 4: Apply Transformations

**Parallel-Analogue Mapping Table (Love→Money):**
```
F4 (Companionship) → F6 (Material)
C2 (Joy/Love) → D0.1_{6d} (Money concept)
C3 (Emotional Fear of loss) → C3 (Emotional Fear of loss) [preserved]
D5 (Woman in relationship) → D5 (Woman with money)
Relationship hiding → Money hiding
```

**Transformation:**
```
Original:
  (B5.2 +_T D3.2) → C3.1 → (D5.1 -_T D0.1_{6d})

After PAR on element (where domain-relevant):
  C2.3 (love emotion, if present) → D0.1_{6d} (money)
  D4 (companionship desire) → D6 (material desire)

This equation is ALREADY in money domain (D0.1_{6d} = money template)
→ Inverse mapping: Money → Love

Inverted:
  (B5.2 +_T D3.2) → C3.1 → (D5.1 -_T C2.3_{4c})

Where:
  D0.1_{6d} (money at material foundation)
  → C2.3 (love/affection at companionship foundation)
```

**Narrative-role META adjustment:**
```
Original role: Woman hides money from partner
Inverted role: Woman withholds affection from partner
```

### Step 5: Narrative Decode
```
Original:
"Accumulated experiences combined with material instability cause fear,
 which causes the woman to hide money."

Inverted (Money → Love parallel):
"Accumulated experiences combined with material instability cause fear,
 which causes the woman to withhold affection."

Change Summary:
- Domain: material (F6) → companionship (F4)
- Target object: money → love/affection
- Subscript: _{6d} → _{4c}
- Action: hide money → withhold affection
- Causal structure: PRESERVED
- Fear: PRESERVED (same driver)
```

### Step 6: Audit
```
Re-encode check:
  Story: "...woman withholds affection"
  Encoded: (B5.2 +_T D3.2) → C3.1 → (D5.1 -_T C2.3_{4c})

  vs Transformed: (B5.2 +_T D3.2) → C3.1 → (D5.1 -_T C2.3_{4c})

  MATCH ✓

Symbol compliance:
  ✓ All elements canonical
  ✓ All subscripts valid
  ✓ No invented symbols

Inversion semantics:
  ✓ ParallelAnalogue correctly maps F6→F4
  ✓ Material domain → Companionship domain
  ✓ Causal structure preserved

STATUS: PASSED
```

### Step 7: Final Output
```
=== INPUT EQUATION ===
(B5.2 +_T D3.2) → C3.1 → (D5.1 -_T D0.1_{6d})

=== DIAL CONFIG USED ===
mode: ParallelAnalogue
intensity: soft
scope: equation
targetProfile: { fromFoundation: F4, toFoundation: F6 } [inverse applied]

=== TRANSFORMATION PLAN (GRID) ===
element: PAR, world: PAR, foundation: PAR, narrativeRole: META
(all other axes: ID)

=== OUTPUT EQUATION (MODE: ParallelAnalogue) ===
(B5.2 +_T D3.2) → C3.1 → (D5.1 -_T C2.3_{4c})

=== ENGLISH INTERPRETATIONS ===
Original: Accumulated experiences combined with material instability
          cause fear, which causes the woman to hide money.
Inverted: Accumulated experiences combined with material instability
          cause fear, which causes the woman to withhold affection.

=== CHANGE SUMMARY ===
- Domain shift: Material (F6) → Companionship (F4)
- Object: money → affection
- World: Physical (D) → Emotional (C) for target
- Action: hide → withhold
- Preserved: causal chain, fear driver, agent (woman)

=== TYPE & CANON CHECK ===
Status: PASSED
Notes: Clean parallel mapping Money→Love domain
```

---

# SECTION 6: DEFAULT AXIS PROFILES BY MODE

## 6.1 Recommended Defaults

For convenience, define typical axis configurations per mode:

```typescript
const MODE_DEFAULT_AXES: Record<InversionMode, AxesConfig> = {
  // Surface
  "Opposite":         { noetic: true,  element: true,  world: true,  foundation: true,  subFoundation: true,  acquisition: true,  causal: false, narrativeRole: false, scalarValence: true  },
  "Dual":             { noetic: false, element: true,  world: true,  foundation: false, subFoundation: false, acquisition: false, causal: false, narrativeRole: false, scalarValence: false },
  "CounterPole":      { noetic: false, element: true,  world: false, foundation: true,  subFoundation: true,  acquisition: true,  causal: false, narrativeRole: false, scalarValence: true  },
  "Mirror":           { noetic: false, element: false, world: false, foundation: false, subFoundation: false, acquisition: false, causal: true,  narrativeRole: true,  scalarValence: false },
  "ReverseCausal":    { noetic: false, element: false, world: false, foundation: false, subFoundation: false, acquisition: false, causal: true,  narrativeRole: true,  scalarValence: false },
  "ParallelAnalogue": { noetic: false, element: true,  world: true,  foundation: true,  subFoundation: false, acquisition: false, causal: false, narrativeRole: true,  scalarValence: false },

  // Deep Structure
  "NoeticComplement":    { noetic: true,  element: false, world: false, foundation: false, subFoundation: false, acquisition: false, causal: false, narrativeRole: false, scalarValence: false },
  "AcquisitionPolarity": { noetic: false, element: false, world: false, foundation: false, subFoundation: false, acquisition: true,  causal: false, narrativeRole: false, scalarValence: false },
  "FoundationFlip":      { noetic: false, element: false, world: false, foundation: true,  subFoundation: true,  acquisition: false, causal: false, narrativeRole: false, scalarValence: false },
  "SubFoundationReversal":{ noetic: false, element: false, world: false, foundation: false, subFoundation: true,  acquisition: false, causal: false, narrativeRole: false, scalarValence: false },
  "DomainPermutation":   { noetic: false, element: false, world: true,  foundation: false, subFoundation: false, acquisition: false, causal: false, narrativeRole: false, scalarValence: false },
  "TemporalInversion":   { noetic: false, element: false, world: false, foundation: false, subFoundation: false, acquisition: false, causal: true,  narrativeRole: true,  scalarValence: false },
  "ScalarInversion":     { noetic: false, element: false, world: false, foundation: false, subFoundation: false, acquisition: false, causal: false, narrativeRole: false, scalarValence: true  },
  "StructuralPermutation":{ noetic: false, element: false, world: false, foundation: false, subFoundation: false, acquisition: false, causal: true,  narrativeRole: true,  scalarValence: false },
  "ContextFrame":        { noetic: false, element: true,  world: true,  foundation: true,  subFoundation: false, acquisition: false, causal: false, narrativeRole: true,  scalarValence: false },
  "Motivational":        { noetic: false, element: false, world: false, foundation: false, subFoundation: false, acquisition: false, causal: false, narrativeRole: true,  scalarValence: false },
  "Polarity":            { noetic: false, element: true,  world: false, foundation: true,  subFoundation: true,  acquisition: true,  causal: false, narrativeRole: false, scalarValence: true  },
  "CausalDensity":       { noetic: false, element: false, world: false, foundation: false, subFoundation: false, acquisition: false, causal: true,  narrativeRole: true,  scalarValence: false },

  // Meta-Layer (all META operations)
  "Attention":        { noetic: true,  element: true,  world: true,  foundation: true,  subFoundation: true,  acquisition: true,  causal: true,  narrativeRole: true,  scalarValence: true  },
  "Value":            { noetic: true,  element: true,  world: true,  foundation: true,  subFoundation: true,  acquisition: true,  causal: true,  narrativeRole: true,  scalarValence: true  },
  "DesireInhibition": { noetic: true,  element: false, world: false, foundation: false, subFoundation: false, acquisition: false, causal: true,  narrativeRole: true,  scalarValence: true  },
  "Expectation":      { noetic: true,  element: true,  world: true,  foundation: true,  subFoundation: true,  acquisition: true,  causal: true,  narrativeRole: true,  scalarValence: true  },
  "Attractor":        { noetic: true,  element: false, world: false, foundation: false, subFoundation: false, acquisition: false, causal: true,  narrativeRole: true,  scalarValence: true  },
  "Stability":        { noetic: true,  element: true,  world: true,  foundation: true,  subFoundation: true,  acquisition: true,  causal: true,  narrativeRole: true,  scalarValence: true  },
  "Entropy":          { noetic: true,  element: true,  world: true,  foundation: true,  subFoundation: true,  acquisition: true,  causal: true,  narrativeRole: true,  scalarValence: true  },

  // Special
  "Constraint":     { noetic: true,  element: true,  world: true,  foundation: true,  subFoundation: true,  acquisition: true,  causal: true,  narrativeRole: true,  scalarValence: true  },
  "Boundary":       { noetic: true,  element: true,  world: true,  foundation: true,  subFoundation: true,  acquisition: true,  causal: true,  narrativeRole: true,  scalarValence: true  },
  "SemanticParity": { noetic: false, element: true,  world: false, foundation: true,  subFoundation: true,  acquisition: true,  causal: false, narrativeRole: true,  scalarValence: false },
  "AgentRole":      { noetic: false, element: false, world: false, foundation: false, subFoundation: false, acquisition: false, causal: false, narrativeRole: true,  scalarValence: true  }
};
```

---

# SECTION 7: ERROR HANDLING

## 7.1 Error Types

```typescript
enum InversionError {
  PARSE_ERROR = "PARSE_ERROR",           // Invalid equation syntax
  TYPE_ERROR = "TYPE_ERROR",             // Type-check failure
  INVALID_MODE = "INVALID_MODE",         // Unknown mode name
  INVALID_AXIS = "INVALID_AXIS",         // Unknown axis name
  MISSING_TARGET = "MISSING_TARGET",     // ParallelAnalogue without targetProfile
  SYMBOL_ERROR = "SYMBOL_ERROR",         // Non-canonical symbol detected
  AUDIT_FAIL = "AUDIT_FAIL",             // Re-encoding mismatch
  SEMANTIC_ERROR = "SEMANTIC_ERROR"      // Inversion semantics violated
}
```

## 7.2 Error Recovery

| Error | Recovery Action |
|-------|-----------------|
| PARSE_ERROR | Return parse diagnostics, suggest corrections |
| TYPE_ERROR | Return type error location, expected types |
| INVALID_MODE | Return canonical mode list |
| MISSING_TARGET | Prompt for targetProfile or use default (F4→F6) |
| SYMBOL_ERROR | Flag non-canonical symbol, suggest replacement |
| AUDIT_FAIL | Re-run with adjusted narrative, log drift |
| SEMANTIC_ERROR | Log violation, return partial result with warning |

---

# SECTION 8: STATUS SUMMARY

| Component | Status |
|-----------|--------|
| Dial Config Schema | ✓ Complete |
| Canonical Mode Enumeration (29) | ✓ Complete |
| Total Inversion Grid (29×9) | ✓ Complete |
| Grid Lookup Functions | ✓ Complete |
| 6-Agent Architecture | ✓ Complete |
| 7-Step Workflow Pipeline | ✓ Complete |
| Worked Example (ParallelAnalogue) | ✓ Complete |
| Default Axis Profiles | ✓ Complete |
| Error Handling | ✓ Complete |

---

# SECTION 9: RELATED FILES

```
TKS_Inversion_Engine_Full_v1.0.md      (29 mode definitions)
TKS_Six_Inversion_Types_v1.0.md        (Surface type details)
TKS_Scenario_Inversion_Knob_v1.0.md    (User API)
TKS_Inversion_Engine_v1_Phase2_Foundations.md (Foundation/SubF/Acq)
TKS_Narrative_Semantics_Rulebook_v1.0.md (Encoding/Decoding)
TKS_Symbol_Sense_Table_v1.0.md         (Element senses)
```

---

*End of TKS Inversion Dial Specification v1.0*
