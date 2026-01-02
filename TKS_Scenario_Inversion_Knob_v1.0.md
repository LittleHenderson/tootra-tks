# TKS Scenario Inversion Knob v1.0

**Grid Architecture**: Option E (Foundations 7 × Worlds 4 × NoeticDigit 10)
**Version**: 1.0
**Status**: Locked

---

## 1. Overview

The Scenario Inversion Knob transforms TKS-encoded scenarios into their structural inverses.

**Pipeline**:
```
Story/TKS Input → Parse → Compute Attractor Signature → Invert Signature → Synthesize Counter-Scenario → Output Inverted TKS + Inverted Story + Diff
```

**Anti-Attractor Rule**: The knob does NOT simply flip tokens. It:
1. Computes the attractor signature of the original scenario
2. Inverts that signature using canonical inversion tables
3. Synthesizes counter-scenarios that converge toward the inverted attractor
**Operational form**: Compute attractor A(s) under v7.x semantics, apply inversion operator Inv_axis to A(s), then synthesize counter-scenarios whose forward dynamics converge to Inv_axis(A(s)).

This ensures semantically coherent inversions rather than arbitrary token swaps.

---

## 2. Grid Architecture (Option E)

### 2.1 Tensor Axes: 7 × 4 × 10

| Axis | Dimension | Values |
|------|-----------|--------|
| Foundation | 7 | F1=Unity, F2=Wisdom, F3=Life, F4=Companionship, F5=Power, F6=Material, F7=Lust |
| World | 4 | A=Spiritual, B=Mental, C=Emotional, D=Physical |
| NoeticDigit | 10 | 0–9 mapping to N1–N10 |

**Total tensor size**: 7 × 4 × 10 = 280 cells

### 2.2 Core Element Grid (4 × 10)

The base element grid is the World × NoeticDigit slice:

```
        N1   N2   N3   N4   N5   N6   N7   N8   N9   N10
       Mind Pos  Neg  Vib  Fem  Male Rhy  Cau  Eff  Idea
      ┌─────────────────────────────────────────────────┐
  A   │ A1   A2   A3   A4   A5   A6   A7   A8   A9   A10 │  Spiritual
  B   │ B1   B2   B3   B4   B5   B6   B7   B8   B9   B10 │  Mental
  C   │ C1   C2   C3   C4   C5   C6   C7   C8   C9   C10 │  Emotional
  D   │ D1   D2   D3   D4   D5   D6   D7   D8   D9   D10 │  Physical
      └─────────────────────────────────────────────────┘
```

### 2.3 "Line Across the Square"

World inversion draws a mirror line across the 4×10 grid:

```
  A ←──────────────────────────→ D
  B ←──────────────────────────→ C
        (horizontal mirror)
```

- A ↔ D: Spiritual ↔ Physical
- B ↔ C: Mental ↔ Emotional

The grid "folds" along its horizontal center.

### 2.4 Foundation as Stacked Rows

Foundations form 7 parallel layers stacked above the 4×10 element grid:

```
       ┌─────────────────────┐
  F7   │   4×10 slice        │  Lust
  F6   │   4×10 slice        │  Material
  F5   │   4×10 slice        │  Power
  F4   │   4×10 slice        │  Companionship
  F3   │   4×10 slice        │  Life
  F2   │   4×10 slice        │  Wisdom
  F1   │   4×10 slice        │  Unity
       └─────────────────────┘
```

Each foundation layer contains a complete 4×10 element grid, enabling expressions like `B5_{F4}` (Mental-Female in Companionship context).

### 2.5 Sub-Foundation Overlay (Optional)

Sub-foundations extend the foundation axis with world qualifiers:

```
SubFoundation = (Foundation, World)
Example: F4.C = Companionship in Emotional domain
```

Inversion applies both:
1. Foundation remap: F4 → F4 (self-dual)
2. World mirror: C → B

Result: F4.C → F4.B

---

## 3. APIs

### 3.1 ScenarioInvert

```python
def ScenarioInvert(
    expr: TKSExpression,
    axes: Set[Axis],
    mode: InversionMode,
    target: Optional[TargetProfile] = None
) -> TKSExpression
```

### 3.2 InvertStory

```python
def InvertStory(
    story: str,
    axes: Set[Axis],
    mode: InversionMode,
    target: Optional[TargetProfile] = None
) -> {
    "expr_original": TKSExpression,
    "expr_inverted": TKSExpression,
    "story_inverted": str
}
```

### 3.3 ExplainInversion

```python
def ExplainInversion(
    expr_original: TKSExpression,
    expr_inverted: TKSExpression
) -> str
```

### 3.4 Axes Enumeration

```python
Axis = {
    "Noetic",        # N - Involution pairs on digit axis
    "Element",       # E - Full element inversion (world + noetic)
    "World",         # W - World mirror only (A↔D, B↔C)
    "Foundation",    # F - Foundation remap (F1↔F7, F2↔F6, F3↔F5)
    "SubFoundation", # S - Foundation + World compound
    "Acquisition",   # A - Logical negation toggle
    "Polarity"       # P - Valence flip (ν2↔ν3)
}
```

Shorthand: `{N, E, W, F, S, A, P}`

### 3.5 InversionMode

```python
InversionMode = Enum{
    Soft,      # Invert only where canonical dual/opposite exists
    Hard,      # Apply on all selected axes unconditionally
    Targeted   # Apply TargetProfile remaps; others unchanged
}
```

### 3.6 TargetProfile

```python
@dataclass
class TargetProfile:
    from_foundation: Optional[int] = None  # e.g., 4
    to_foundation: Optional[int] = None    # e.g., 6
    from_world: Optional[str] = None       # e.g., "C"
    to_world: Optional[str] = None         # e.g., "B"
```

### 3.7 Polarity Rule

The Polarity axis operates specifically on ν2↔ν3:
- ν2 (Positive valence) ↔ ν3 (Negative valence)
- Element counterparts: X2 ↔ X3 for any world X
- Soft: Only explicit polarity markers
- Hard: All elements at positions 2 or 3

---

## 4. Axis Semantics

Axis set (explicit): {World, Noetic, Element, Foundation, SubFoundation, Acquisition, Polarity/Sign}. Polarity is derived from the canonical involution pairs (ν2↔ν3, ν5↔ν6, ν8↔ν9) and their element counterparts.

### 4.1 Noetic Axis (N)

**Involution pairs** (0-indexed):
| Index | Noetic | Inverts To | Meaning |
|-------|--------|------------|---------|
| 1 | N2/Positive | N3/Negative | Polarity swap |
| 2 | N3/Negative | N2/Positive | Polarity swap |
| 4 | N5/Female | N6/Male | Gender swap |
| 5 | N6/Male | N5/Female | Gender swap |
| 7 | N8/Cause | N9/Effect | Causality swap |
| 8 | N9/Effect | N8/Cause | Causality swap |

**Self-duals** (no inversion):
- N1 (Mind), N4 (Vibration), N7 (Rhythm), N10 (Idea)

**Mode behavior**:
- Soft: Leave self-duals unchanged; invert only pairs
- Hard: Apply to all; self-duals map to self

### 4.2 Element Axis (E)

Uses `ElementInv_total` combining world + noetic:
```
ElementInv_total(Xn) = WorldInv(X) + NoeticInv(n-1) + 1
```

Examples:
- B5 → C6 (Mental-Female → Emotional-Male)
- D3 → A2 (Physical-Negative → Spiritual-Positive)

**Sense indices**: Preserved when compatible. If domain changes incompatibly, remap flag set for downstream.

### 4.3 World Axis (W)

```python
WORLD_OPP = {"A": "D", "D": "A", "B": "C", "C": "B"}
```

Applied independently of noetic for "shift perspective" without changing internal dynamics.

### 4.4 Foundation Axis (F)

```python
FOUNDATION_OPP = {1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1}
```

| Original | Inverted |
|----------|----------|
| F1/Unity | F7/Lust |
| F2/Wisdom | F6/Material |
| F3/Life | F5/Power |
| F4/Companionship | F4/Companionship (self-dual) |

### 4.5 SubFoundation Axis (S)

Compound inversion:
```
SubFoundationInv(Fm.W) = (FoundationInv(Fm), WorldInv(W))
```

Example: F2.B (Wisdom-Mental) → F6.C (Material-Emotional)

### 4.6 Acquisition Axis (A)

Logical negation toggle:
```
A  → ¬A
¬A → A
```

Applied to acquisition chains (A0, D, W, P markers).

### 4.7 Polarity Axis (P)

Focuses on ν2↔ν3 and element counterparts:
- X2 ↔ X3 for any world X
- Soft: Only explicit polarity-flagged elements
- Hard: All elements at positions 2 or 3

---

## 5. Modes

### 5.1 Soft Mode

- Invert only where canonical dual/opposite exists
- Idempotent on balanced forms (soft(soft(x)) = x)
- Preserves self-dual elements unchanged
- Use for exploratory "what if" analysis

### 5.2 Hard Mode

- Apply inversion on ALL selected axes
- Self-duals map to self (no arbitrary mutation)
- Maximally disruptive; produces distant scenarios
- Use for total opposition synthesis

### 5.3 Targeted Mode

- Apply only TargetProfile remaps
- Other elements/foundations unchanged
- Use for specific transforms (e.g., "Love→Money")

### 5.4 Soft vs Hard Comparison

**Input equation**: `B5 +T→ D3 -T→ C8`

**Axes**: {Noetic, World}

| Mode | Result | Notes |
|------|--------|-------|
| Soft | `C6 -T→ A2 +T→ B9` | Pairs inverted; self-duals unchanged |
| Hard | `C6 -T→ A2 +T→ B9` | Same (all have canonical inverses) |

**Input equation**: `A1 → A4 → A7` (all self-duals)

| Mode | Result | Notes |
|------|--------|-------|
| Soft | `D1 → D4 → D7` | World mirrored; noetics unchanged |
| Hard | `D1 → D4 → D7` | Same (self-duals remain self) |

---

## 6. Worked Examples

### 6.1 Example 1: Love → Money

**Scenario**: "She chose love over career advancement."

**Configuration**:
- Axes: {Foundation, SubFoundation, Acquisition}
- Mode: Targeted
- TargetProfile: F4 → F6 (Companionship → Material)

**expr_original**:
```
F4.C5 +T→ ¬F6.D3
[Companionship-Emotional-Female yields not-Material-Physical-Negative]
```

**expr_inverted**:
```
F6.B6 +T→ ¬F4.A2
[Material-Mental-Male yields not-Companionship-Spiritual-Positive]
```

**story_inverted**: "He chose wealth over emotional connection."

**ExplainInversion**:
```
Foundation: Companionship(F4) → Material(F6)
World: C(Emotional) → B(Mental), D(Physical) → A(Spiritual)
SubFoundation: Gender flip via world-linked noetic
Acquisition: Negation preserved
Result: Love-driven → Money-driven with inverted emotional polarity
```

### 6.2 Example 2: Emotional → Intellectual

**Scenario**: "Her anger clouded her judgment."

**Configuration**:
- Axes: {World, Noetic}
- Mode: Soft

**expr_original**:
```
C3 → C1
[Emotional-Negative affects Emotional-Mind]
```

**expr_inverted**:
```
B2 → B1
[Mental-Positive affects Mental-Mind]
```

**story_inverted**: "His optimism clarified his thinking."

**ExplainInversion**:
```
World: C(Emotional) → B(Mental)
Noetic: N3(Negative) → N2(Positive); N1(Mind) self-dual
Emotional negativity becomes mental positivity
```

### 6.3 Example 3: Total Inversion

**Scenario**: "The weak became strong through suffering."

**Configuration**:
- Axes: {Noetic, Element, World, Foundation, Acquisition, Polarity}
- Mode: Hard

**expr_original**:
```
F5.D3 -T→ F5.D6 | via F3.C9
[Power-Physical-Negative → Power-Physical-Male via Life-Emotional-Effect]
```

**expr_inverted**:
```
F3.A2 +T→ F3.A5 | via F5.B8
[Life-Spiritual-Positive → Life-Spiritual-Female via Power-Mental-Cause]
```

**story_inverted**: "The strong became gentle through joy."

**ExplainInversion**:
```
Foundation: F5(Power)→F3(Life), F3(Life)→F5(Power)
World: D→A, C→B
Noetic: N3→N2, N6→N5, N9→N8
Polarity: -T→+T
Expression remains well-typed: all elements valid, operators balanced
```

---

## 7. CLI Sketch

```
tks-invert [OPTIONS] <INPUT>

INPUT:
  --story <text>           Natural language input
  --equation <expr>        Direct TKS equation

AXES (comma-separated):
  --axes N,E,W,F,S,A,P
    N=Noetic, E=Element, W=World, F=Foundation,
    S=SubFoundation, A=Acquisition, P=Polarity

MODE:
  --mode soft|hard|targeted

TARGET PROFILE:
  --from-foundation <1-7>
  --to-foundation <1-7>
  --from-world <A|B|C|D>
  --to-world <A|B|C|D>

OUTPUT:
  --format json|text|diff
```

**Example invocations**:
```bash
tks-invert --story "She loved him" --axes W,N --mode soft
tks-invert --equation "B5+T→D3" --axes E --mode hard
tks-invert --story "Power corrupts" --axes F,A --mode targeted \
           --from-foundation 5 --to-foundation 2
```

**Output format (text)**:
```
=== ORIGINAL ===
Story: She loved him
Equation: C5 +T→ C6

=== INVERTED ===
Equation: B6 -T→ B5
Story: He resented her

=== EXPLANATION ===
World: C(Emotional) → B(Mental)
Noetic: N5↔N6 swapped
Operator: +T → -T
```

---

## 8. Status Summary

### 8.1 Fully Specified

- 7×4×10 tensor grid architecture (Option E locked)
- All 7 inversion axes with canonical mappings
- Three inversion modes (Soft/Hard/Targeted)
- Core APIs: ScenarioInvert, InvertStory, ExplainInversion
- World opposition: A↔D, B↔C
- Noetic involutions: 2↔3, 5↔6, 8↔9
- Foundation opposition: 1↔7, 2↔6, 3↔5, 4↔4
- Acquisition negation toggle
- Anti-attractor rule: invert signature, not tokens
- CLI interface specification

### 8.1a Mathematical Status

- **Formal**: axis involutions (world/noetic/element/foundation/subfoundation/acquisition/polarity), typing constraints, tensor grid (7×4×10), anti-attractor definition (invert signature then synthesize converging counter-scenario).
- **Heuristic/Empirical**: sense remap tables for narrative phrases (NLG layer), story decoding/encoding quality, operator inversion beyond +T/-T (pending), sub-foundation overlays beyond base mapping (pending).

### 8.2 Ambiguities Resolved

| Item | Resolution |
|------|------------|
| Grid shape | Locked to 7×4×10 |
| Self-duals | N1, N4, N7, N10 map to self |
| F4 | Self-dual in foundation inversion |
| Polarity scope | ν2↔ν3 and element counterparts |
| Axis order | Element → Polarity → Noetic → World → Foundation → Sub → Acquisition |
| Anti-attractor | Invert attractor signature, synthesize converging counter-scenario |

### 8.3 TODOs for v2

1. **Sense remap rules**: Full mapping table for sense indices across inversions
2. **Anti-attractor synthesis**: Richer algorithm for counter-scenario generation
3. **Sub-foundation overlay expansion**: Complete 28 sub-foundation mapping table
4. **Operator inversion taxonomy**: Extend beyond +T/-T
5. **Composition rules**: Behavior when multiple targeted transforms conflict
6. **Validation constraints**: Type-checking for well-formed inverted expressions
7. **Story synthesis quality**: NLG improvements for narrative generation

---

*End of TKS Scenario Inversion Knob Specification v1.0*
