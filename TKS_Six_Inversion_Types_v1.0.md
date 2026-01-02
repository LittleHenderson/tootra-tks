# TKS Six Inversion Types — Formal Specification v1.0

## Complete Transformation Algebra for TKS Inversion Engine

**Document:** TKS_Six_Inversion_Types_v1.0.md
**Version:** 1.0
**Date:** 2025-12-10
**Dependencies:**
- TKS Inversion Engine Phase 1 (Element, Noetic, World inversion)
- TKS Inversion Engine Phase 2 (Foundation, Sub-Foundation, Acquisition inversion)
- TKS v7.4 Canonical Definitions

---

## Overview

This document defines **six canonical inversion types** that form a complete transformation algebra for TKS scenarios:

| # | Type | Symbol | Core Operation |
|---|------|--------|----------------|
| 1 | **Opposite** | ⊖ | Semantic opposite via ElementInv_opp |
| 2 | **Dual** | ⊗ | Cross-world dual via ElementInv_dual |
| 3 | **Counter-Pole** | ⊕ | Internal pole flip (+/−) within element |
| 4 | **Mirror** | ⟷ | Structural reflection of causal graph |
| 5 | **Reverse-Causal** | ⟲ | Semantic reversal of causal dependencies |
| 6 | **Parallel Analogue** | ∥ | Domain translation to analogous structure |

Each type preserves TKS well-typedness and is formally defined with:
- Symbol-level rules
- Causal-graph rules
- Composition rules
- Validation tests

---

# Type 1: OPPOSITE Inversion (⊖)

## 1.1 Definition

**Opposite Inversion** maps each symbol to its semantic opposite using the `ElementInv_opp` function. This inverts the Noetic index while preserving the World.

```
Opp : TKS_Expr → TKS_Expr
Opp(E) = Apply ElementInv_opp to all elements in E
```

## 1.2 Symbol-Level Rules

### 1.2.1 Element Rules

```
Opp(Xn) = X(NoeticInv(n))

Where NoeticInv:
  0 → 0 (Idea, self-dual)
  1 → 1 (Mind, self-dual)
  2 → 3 (Positive → Negative)
  3 → 2 (Negative → Positive)
  4 → 4 (Vibration, self-dual)
  5 → 6 (Female → Male)
  6 → 5 (Male → Female)
  7 → 7 (Rhythm, self-dual)
  8 → 9 (Above → Below)
  9 → 8 (Below → Above)
```

### 1.2.2 Noetic Superscript Rules

```
Opp(E^k) = Opp(E)^(NoeticInv(k))
```

### 1.2.3 Foundation Rules

```
Opp(F_m) = FoundationInv(F_m)

FoundationInv:
  F₁ → F₇ (Unity → Lust)
  F₂ → F₆ (Wisdom → Material)
  F₃ → F₅ (Life → Power)
  F₄ → F₄ (Companionship, self-dual)
  F₅ → F₃ (Power → Life)
  F₆ → F₂ (Material → Wisdom)
  F₇ → F₁ (Lust → Unity)
```

### 1.2.4 Sub-Foundation Rules

```
Opp(F_{m,w}) = (FoundationInv(F_m), w)
// World preserved, Foundation inverted
```

### 1.2.5 Acquisition Rules

```
Opp(A₀) = A₀ (self-dual)
Opp(D_m) = D_{FoundationInv(m)}
Opp(W_m) = W_{FoundationInv(m)}
Opp(P_m) = P_{FoundationInv(m)}
```

## 1.3 Causal-Graph Rules

```
Opp(X → Y → Z) = Opp(X) → Opp(Y) → Opp(Z)
// Causal structure PRESERVED, symbols inverted
```

## 1.4 Operator Rules

```
Opp(X +_T Y) = Opp(X) +_T Opp(Y)
Opp(X -_T Y) = Opp(X) -_T Opp(Y)
Opp(X ×_T Y) = Opp(X) ×_T Opp(Y)
Opp(X /_T Y) = Opp(X) /_T Opp(Y)
```

## 1.5 Opposite Inversion Table (Key Examples)

| Original | Opp(Original) | Semantic Shift |
|----------|---------------|----------------|
| D5.1 (Woman) | D6.1 (Man) | Female → Male |
| D2.1 (Health) | D3.1 (Illness) | Order → Disorder |
| C2.1 (Joy) | C3.1 (Fear) | Positive → Negative |
| C5.1 (Receptivity) | C6.1 (Expression) | Female → Male |
| B8.1 (Trigger) | B9.1 (Response) | Cause → Effect |
| A2.1 (Alignment) | A3.1 (Misalignment) | Positive → Negative |

## 1.6 Laws

**Law Opp-1 (Involution):**
```
Opp(Opp(E)) = E for all expressions E
```

**Law Opp-2 (Causal Preservation):**
```
CausalStructure(Opp(E)) = CausalStructure(E)
```

**Law Opp-3 (World Preservation):**
```
World(Opp(Xn)) = World(Xn) for all elements Xn
```

---

# Type 2: DUAL Inversion (⊗)

## 2.1 Definition

**Dual Inversion** maps each symbol to its cross-world dual using `ElementInv_dual`. This inverts the World while preserving the Noetic.

```
Dual : TKS_Expr → TKS_Expr
Dual(E) = Apply ElementInv_dual to all elements in E
```

## 2.2 Symbol-Level Rules

### 2.2.1 Element Rules

```
Dual(Xn) = (WorldInv(X))n

Where WorldInv:
  A → D (Spiritual → Physical)
  B → C (Mental → Emotional)
  C → B (Emotional → Mental)
  D → A (Physical → Spiritual)
```

### 2.2.2 Noetic Superscript Rules

```
Dual(E^k) = Dual(E)^k
// Noetic preserved
```

### 2.2.3 Foundation Rules

```
Dual(F_m) = F_m
// Foundations have no cross-world dual; preserved
```

### 2.2.4 Sub-Foundation Rules

```
Dual(F_{m,w}) = (F_m, WorldInv(w))
// Foundation preserved, World inverted
```

### 2.2.5 Acquisition Rules

```
Dual(A) = A for all Acquisitions
// Acquisitions are Foundation-indexed, not World-indexed
```

## 2.3 Causal-Graph Rules

```
Dual(X → Y → Z) = Dual(X) → Dual(Y) → Dual(Z)
// Causal structure PRESERVED, worlds inverted
```

## 2.4 Operator Rules

```
Dual(X +_T Y) = Dual(X) +_T Dual(Y)
Dual(X -_T Y) = Dual(X) -_T Dual(Y)
Dual(X ×_T Y) = Dual(X) ×_T Dual(Y)
Dual(X /_T Y) = Dual(X) /_T Dual(Y)
```

## 2.5 Dual Inversion Table (Key Examples)

| Original | Dual(Original) | Semantic Shift |
|----------|----------------|----------------|
| D5.1 (Woman) | A5.1 (Divine Receptivity) | Physical → Spiritual |
| D6.1 (Man) | A6.1 (Divine Will) | Physical → Spiritual |
| C2.1 (Joy) | B2.1 (Positive Belief) | Emotional → Mental |
| C3.1 (Fear) | B3.1 (Limiting Belief) | Emotional → Mental |
| B5.1 (Learning) | C5.1 (Emotional Receptivity) | Mental → Emotional |
| A8.1 (Divine Cause) | D8.1 (Physical Trigger) | Spiritual → Physical |

## 2.6 Laws

**Law Dual-1 (Involution):**
```
Dual(Dual(E)) = E for all expressions E
```

**Law Dual-2 (Noetic Preservation):**
```
Noetic(Dual(Xn)) = Noetic(Xn) for all elements Xn
```

**Law Dual-3 (Foundation Preservation):**
```
Foundation(Dual(E_{m,w})) = Foundation(E_{m,w})
```

---

# Type 3: COUNTER-POLE Inversion (⊕)

## 3.1 Definition

**Counter-Pole Inversion** flips the internal pole (+/−) within an element's sense layer without changing the element itself. This introduces a **constructive/destructive** dimension.

```
CounterPole : TKS_Expr → TKS_Expr
CounterPole(E) = Flip internal poles of all elements in E
```

## 3.2 Pole System Definition

Each Element.Sense has two poles:

| Pole | Symbol | Meaning |
|------|--------|---------|
| **Constructive** | ⁺ | Building, nurturing, creating |
| **Destructive** | ⁻ | Breaking, withholding, destroying |

Default sense (unmarked) is assumed **constructive** (⁺).

## 3.3 Symbol-Level Rules

### 3.3.1 Element Rules

```
CounterPole(Xn.s⁺) = Xn.s⁻
CounterPole(Xn.s⁻) = Xn.s⁺
CounterPole(Xn.s) = Xn.s⁻  // Unmarked → Destructive
```

### 3.3.2 Noetic/Foundation/Acquisition Rules

```
CounterPole does NOT change:
  - Noetic indices
  - World indices
  - Foundation indices
  - Acquisitions

Only the internal pole marker changes.
```

## 3.4 Counter-Pole Table (Key Examples)

| Original | CounterPole | Semantic Shift |
|----------|-------------|----------------|
| D5.1⁺ (Nurturing Woman) | D5.1⁻ (Withholding Woman) | Giving → Withholding |
| D6.1⁺ (Protective Man) | D6.1⁻ (Dominating Man) | Protecting → Dominating |
| C2.1⁺ (Sharing Joy) | C2.1⁻ (Hoarding Joy) | Open → Closed |
| C3.1⁺ (Cautionary Fear) | C3.1⁻ (Paralyzing Fear) | Functional → Dysfunctional |
| B2.1⁺ (Empowering Belief) | B2.1⁻ (Rigid Belief) | Flexible → Rigid |
| D8.1⁺ (Enabling Authority) | D8.1⁻ (Oppressive Authority) | Enabling → Oppressing |

## 3.5 Causal-Graph Rules

```
CounterPole(X → Y → Z) = CounterPole(X) → CounterPole(Y) → CounterPole(Z)
// Structure preserved, poles flipped
```

## 3.6 Laws

**Law CP-1 (Involution):**
```
CounterPole(CounterPole(E)) = E for all expressions E
```

**Law CP-2 (Element Preservation):**
```
BaseElement(CounterPole(Xn.s)) = BaseElement(Xn.s)
```

**Law CP-3 (Structure Preservation):**
```
CausalStructure(CounterPole(E)) = CausalStructure(E)
World(CounterPole(Xn)) = World(Xn)
Noetic(CounterPole(Xn)) = Noetic(Xn)
```

---

# Type 4: MIRROR Inversion (⟷)

## 4.1 Definition

**Mirror Inversion** reflects the structural pattern of an equation across the causal axis, reversing the direction of all arrows while preserving symbol identity.

```
Mirror : TKS_Expr → TKS_Expr
Mirror(E) = Reverse all causal arrows in E
```

## 4.2 Symbol-Level Rules

```
Mirror(Xn) = Xn  // Individual symbols unchanged
Mirror(E^k) = E^k  // Superscripts unchanged
Mirror(E_{m,w}) = E_{m,w}  // Subscripts unchanged
```

## 4.3 Causal-Graph Rules (Primary)

### 4.3.1 Arrow Reversal

```
Mirror(X → Y) = Y → X
Mirror(X → Y → Z) = Z → Y → X
Mirror(X → (Y → Z)) = (Z → Y) → X
```

### 4.3.2 Sequence Reversal

```
Mirror(X ∘ Y ∘ Z) = Z ∘ Y ∘ X
```

### 4.3.3 Binary Operator Rules

```
Mirror(X +_T Y) = Y +_T X  // Commutative, but order reverses
Mirror(X -_T Y) = Y -_T X  // Order reverses (semantics change!)
Mirror(X ×_T Y) = Y ×_T X  // Order reverses
Mirror(X /_T Y) = Y /_T X  // Order reverses
```

## 4.4 Mirror Examples

| Original | Mirror | Interpretation |
|----------|--------|----------------|
| `A → B → C` | `C → B → A` | Cause chain reverses |
| `(X +_T Y) → Z` | `Z → (Y +_T X)` | Result becomes cause |
| `Fear → Hiding → Isolation` | `Isolation → Hiding → Fear` | Effect chain becomes cause chain |

## 4.5 Semantic Interpretation

Mirror inversion swaps **agent/patient** relationships:

| Original | Mirrored |
|----------|----------|
| "She fears him" | "He fears her" |
| "A causes B" | "B causes A" |
| "X leads to Y" | "Y leads to X" |

## 4.6 Laws

**Law M-1 (Involution):**
```
Mirror(Mirror(E)) = E for all expressions E
```

**Law M-2 (Symbol Preservation):**
```
Elements(Mirror(E)) = Elements(E) as multisets
```

**Law M-3 (Structure Reversal):**
```
CausalOrder(Mirror(E)) = Reverse(CausalOrder(E))
```

---

# Type 5: REVERSE-CAUSAL Inversion (⟲)

## 5.1 Definition

**Reverse-Causal Inversion** reverses the causal order AND re-computes the semantic functions of each node. Unlike Mirror (which only flips arrows), Reverse-Causal transforms what each position MEANS.

```
ReverseCausal : TKS_Expr → TKS_Expr
ReverseCausal(E) = Mirror(E) ∘ TransformSemantics(E)
```

## 5.2 Symbol-Level Rules

### 5.2.1 Position-Based Transformation

In a causal chain `X → Y → Z`:
- **X** is the CAUSE (source)
- **Y** is the MEDIATOR (process)
- **Z** is the EFFECT (result)

Reverse-Causal transforms based on position:

```
ReverseCausal(X → Y → Z) = Cause⁻¹(Z) → Med⁻¹(Y) → Effect⁻¹(X)

Where:
  Cause⁻¹(E) = Apply 8→9 transformation (Above → Below)
  Med⁻¹(E) = E (mediator role preserved)
  Effect⁻¹(E) = Apply 9→8 transformation (Below → Above)
```

### 5.2.2 Element Transformation

```
// For elements in CAUSE position:
ReverseCausal_cause(Xn) = X(n') where:
  if n = 8: n' = 9
  if n = 9: n' = 8
  else: n' = n

// For elements in EFFECT position:
ReverseCausal_effect(Xn) = X(n') where:
  if n = 8: n' = 9
  if n = 9: n' = 8
  else: n' = n
```

### 5.2.3 Acquisition Re-computation

```
If original has RPM chain A₀ → D_m → W_m → P_m:
ReverseCausal produces: P_m → W_m → D_m → A₀

With semantic reinterpretation:
- P_m becomes "starting condition" (what you had)
- D_m becomes "resulting desire" (what you want after)
```

## 5.3 Causal-Graph Rules

```
ReverseCausal(X → Y) = Transform(Y) → Transform(X)
ReverseCausal(X → Y → Z) = Transform(Z) → Transform(Y) → Transform(X)

// Where Transform adjusts 8↔9 based on new position
```

## 5.4 Reverse-Causal Examples

| Original | ReverseCausal | Semantic Shift |
|----------|---------------|----------------|
| `C3.1 → D7.1` | `D7.1 → C3.1` | "Fear causes habit" → "Habit causes fear" |
| `Fear → Hiding → Isolation` | `Isolation → Hiding → Fear` | Effect becomes cause, with role shifts |
| `B8.1 → C3.1 → D7.1` | `D7.1 → C3.1 → B9.1` | Cause(B8) becomes Effect(B9) |

## 5.5 Laws

**Law RC-1 (Involution):**
```
ReverseCausal(ReverseCausal(E)) = E for all expressions E
```

**Law RC-2 (Structure Reversal):**
```
CausalOrder(ReverseCausal(E)) = Reverse(CausalOrder(E))
```

**Law RC-3 (Role Transformation):**
```
CauseRole(ReverseCausal(E)) = EffectRole(E) transformed
EffectRole(ReverseCausal(E)) = CauseRole(E) transformed
```

---

# Type 6: PARALLEL ANALOGUE Inversion (∥)

## 6.1 Definition

**Parallel Analogue Inversion** replaces all symbols with their analogous structure in a parallel domain while preserving the structural pattern and causal flow.

```
Parallel : TKS_Expr × DomainMap → TKS_Expr
Parallel(E, map) = Apply analogous element mapping to E
```

## 6.2 Domain Analogy Maps

### 6.2.1 Standard Analogy Pairs

| Domain A | Domain B | Structural Analogy |
|----------|----------|-------------------|
| Love (F₄) | Money (F₆) | Relationship ↔ Transaction |
| Wisdom (F₂) | Power (F₅) | Knowledge ↔ Authority |
| Health (F₃) | Unity (F₁) | Vitality ↔ Spiritual Wholeness |
| Emotion (C) | Intellect (B) | Feeling ↔ Thinking |
| Physical (D) | Spiritual (A) | Matter ↔ Spirit |

### 6.2.2 Analogy Map Definition

```typescript
type AnalogyMap = {
  source: Foundation | World,
  target: Foundation | World,
  elementMap: Map<Element, Element>
}
```

### 6.2.3 Standard Analogy Maps

**Love → Money Analogy:**
```
AnalogMap_LoveMoney = {
  source: F₄,
  target: F₆,
  elementMap: {
    C2.3 (Love) → D0.1_{6d} (Money concept)
    C5.1 (Emotional Receptivity) → D5.2 (Material Receptacle)
    D6.1 (Partner/Man) → D6.2 (Financial Structure)
    D5.1 (Partner/Woman) → D5.2 (Material Vessel)
    C3.1 (Fear of rejection) → C3.1 (Fear of loss)
    C2.2 (Attraction) → C2.2 (Material attraction)
  }
}
```

**Health → Power Analogy:**
```
AnalogMap_HealthPower = {
  source: F₃,
  target: F₅,
  elementMap: {
    D2.1 (Physical Health) → D8.3 (Material Authority)
    D3.1 (Illness) → D9.1 (Foundation/Subordination)
    D4.1 (Energy) → D4.1 (Power intensity)
    C2.1 (Vitality joy) → C2.1 (Power satisfaction)
  }
}
```

**Emotional → Intellectual Analogy:**
```
AnalogMap_EmotionIntellect = {
  source: C,
  target: B,
  elementMap: {
    C0 → B0 (Concept)
    C1 → B1 (Awareness)
    C2 → B2 (Positive)
    C3 → B3 (Negative)
    C4 → B4 (Intensity)
    C5 → B5 (Receptive)
    C6 → B6 (Projective)
    C7 → B7 (Pattern)
    C8 → B8 (Trigger)
    C9 → B9 (Response)
  }
}
```

## 6.3 Symbol-Level Rules

### 6.3.1 Element Mapping

```
Parallel(Xn.s, map) = map.elementMap[Xn].s
// If not in map, preserve element unchanged
```

### 6.3.2 Foundation Mapping

```
Parallel(F_m, map) = map.target if m = map.source, else F_m
```

### 6.3.3 Structural Preservation

```
Parallel preserves:
  - Causal order (X → Y → Z stays X' → Y' → Z')
  - Operator types (+_T, -_T, ×_T, /_T)
  - Superscript positions
  - Subscript structure (only Foundation index changes)
```

## 6.4 Causal-Graph Rules

```
Parallel(X → Y → Z, map) = Parallel(X, map) → Parallel(Y, map) → Parallel(Z, map)
Parallel(X +_T Y, map) = Parallel(X, map) +_T Parallel(Y, map)
```

## 6.5 Parallel Examples

### Example: Love Story → Money Story

**Original (Love Domain):**
```
"A woman desires love but fears rejection from her partner"
(D5.1 +_T D₄) +_T (C3.1 +_T C2.3)_{4c} → D6.1
```

**Parallel (Money Domain):**
```
"A woman desires wealth but fears loss from her finances"
(D5.1 +_T D₆) +_T (C3.1 +_T D0.1_{6d})_{6c} → D6.2
```

### Example: Health Story → Power Story

**Original (Health Domain):**
```
"His illness drains his energy"
D3.1 -_T D4.1_{3d}
```

**Parallel (Power Domain):**
```
"His subordination drains his authority"
D9.1 -_T D8.3_{5d}
```

## 6.6 Laws

**Law P-1 (Structure Preservation):**
```
CausalStructure(Parallel(E, map)) = CausalStructure(E)
```

**Law P-2 (Domain Shift):**
```
Foundation(Parallel(E_{m,w}, map)) = map.target if m = map.source
```

**Law P-3 (Composition):**
```
Parallel(E, map₁ ∘ map₂) = Parallel(Parallel(E, map₁), map₂)
```

---

# Section 7: Composition Rules

## 7.1 Composition Table

The six inversions can be composed. Here's the composition behavior:

| A ∘ B | Opp | Dual | CP | Mirror | RC | Parallel |
|-------|-----|------|-------|--------|------|----------|
| **Opp** | Id | Total | Opp∘CP | Opp∘M | Opp∘RC | Opp∘P |
| **Dual** | Total | Id | Dual∘CP | Dual∘M | Dual∘RC | Dual∘P |
| **CP** | CP∘Opp | CP∘Dual | Id | CP∘M | CP∘RC | CP∘P |
| **Mirror** | M∘Opp | M∘Dual | M∘CP | Id | RC' | M∘P |
| **RC** | RC∘Opp | RC∘Dual | RC∘CP | RC' | Id | RC∘P |
| **Parallel** | P∘Opp | P∘Dual | P∘CP | P∘M | P∘RC | P∘P' |

Where:
- **Id** = Identity (A ∘ A = Id for involutions)
- **Total** = Opp ∘ Dual = ElementInv_total
- **RC'** = Modified Reverse-Causal
- **P∘P'** = Chain of domain mappings

## 7.2 Key Composition Identities

### 7.2.1 Involution Laws

```
Opp ∘ Opp = Id
Dual ∘ Dual = Id
CP ∘ CP = Id
Mirror ∘ Mirror = Id
RC ∘ RC = Id
```

### 7.2.2 Total Inversion

```
Total = Opp ∘ Dual = Dual ∘ Opp
Total(Xn) = ElementInv_total(Xn) = (WorldInv(X))(NoeticInv(n))
```

### 7.2.3 Commutativity

```
CP commutes with all symbol-level inversions:
  CP ∘ Opp = Opp ∘ CP
  CP ∘ Dual = Dual ∘ CP

Mirror commutes with symbol-level inversions:
  Mirror ∘ Opp = Opp ∘ Mirror
  Mirror ∘ Dual = Dual ∘ Mirror
```

### 7.2.4 Non-Commutativity

```
RC and Parallel generally do NOT commute with others:
  RC ∘ Opp ≠ Opp ∘ RC (in general)
  Parallel depends on domain map
```

## 7.3 Composition Examples

### Example: Opp ∘ Mirror

```
Original: D5.1 → C3.1 → D7.1
"Woman causes fear causes habit"

Mirror first:
  D7.1 → C3.1 → D5.1
  "Habit causes fear causes woman"

Then Opp:
  D7.1 → C2.1 → D6.1
  "Habit causes joy causes man"
```

### Example: Dual ∘ CP

```
Original: D5.1⁺ (Nurturing woman)

Dual first:
  A5.1⁺ (Divine nurturing receptivity)

Then CP:
  A5.1⁻ (Divine withholding receptivity)
```

---

# Section 8: Validation Tests

## 8.1 Test Suite Structure

Each inversion type requires:
1. **Identity Test:** Inv(Inv(E)) = E
2. **Type Preservation Test:** Result is well-typed TKS
3. **Semantic Coherence Test:** Inverted meaning is interpretable
4. **Composition Test:** Composing with other inversions works correctly

## 8.2 Test Cases

### Test 8.2.1: Opposite Inversion

```
Input:  (D5.1 +_T C3.1)_{4c}
        "Woman with fear in relationship context"

Opp:    (D6.1 +_T C2.1)_{4c}
        "Man with joy in relationship context"

Verify:
  ✓ D5→D6 (Female→Male)
  ✓ C3→C2 (Negative→Positive)
  ✓ F₄→F₄ (self-dual, preserved)
  ✓ Opp(Opp(E)) = E
```

### Test 8.2.2: Dual Inversion

```
Input:  (D5.1 +_T C3.1)_{4c}
        "Woman with fear in relationship context"

Dual:   (A5.1 +_T B3.1)_{4b}
        "Divine receptivity with limiting belief in mental relationship"

Verify:
  ✓ D→A (Physical→Spiritual)
  ✓ C→B (Emotional→Mental)
  ✓ c→b (Emotional world→Mental world)
  ✓ Dual(Dual(E)) = E
```

### Test 8.2.3: Counter-Pole Inversion

```
Input:  D5.1⁺ +_T D6.1⁺
        "Nurturing woman with protective man"

CP:     D5.1⁻ +_T D6.1⁻
        "Withholding woman with dominating man"

Verify:
  ✓ Elements unchanged (D5.1, D6.1)
  ✓ Only poles flipped (⁺→⁻)
  ✓ CP(CP(E)) = E
```

### Test 8.2.4: Mirror Inversion

```
Input:  D5.1 → C3.1 → D7.1
        "Woman causes fear causes habit"

Mirror: D7.1 → C3.1 → D5.1
        "Habit causes fear causes woman"

Verify:
  ✓ Elements preserved (D5.1, C3.1, D7.1)
  ✓ Order reversed
  ✓ Mirror(Mirror(E)) = E
```

### Test 8.2.5: Reverse-Causal Inversion

```
Input:  B8.1 → C3.1 → D7.1
        "Mental trigger causes fear causes physical habit"

RC:     D7.1 → C3.1 → B9.1
        "Physical habit causes fear causes mental response"

Verify:
  ✓ Order reversed
  ✓ B8 (cause position) → B9 (effect position)
  ✓ RC(RC(E)) = E
```

### Test 8.2.6: Parallel Analogue Inversion

```
Input:  (D5.1 +_T D₄)_{4c}
        "Woman desires love in relationship context"

Parallel(Love→Money):
        (D5.1 +_T D₆)_{6c}
        "Woman desires wealth in material context"

Verify:
  ✓ Structure preserved
  ✓ D₄→D₆ (acquisition mapped)
  ✓ F₄→F₆ (foundation mapped)
  ✓ c preserved (emotional world)
```

---

# Section 9: Master Test Page

## 9.1 Sample Equation

**Original Story:**
> "A woman fears losing control and hides money from her partner."

**Original TKS Equation:**
```
E₀ = (D5.1 +_T C3.1) → (D5.1 -_T D0.1_{6d}) → D6.1
     [Woman + Fear] → [Woman hides money] → [affects Partner]

Context: _{4c, 6c} (Relationship-Emotional, Material-Emotional)
```

## 9.2 All Six Inversions Applied

### Inversion 1: OPPOSITE (⊖)

```
Opp(E₀) = (D6.1 +_T C2.1) → (D6.1 -_T D0.1_{2d}) → D5.1

Decoded: "A man feels joy and reveals knowledge to his partner"

Changes:
  • D5.1 (Woman) → D6.1 (Man)
  • C3.1 (Fear) → C2.1 (Joy)
  • D0.1_{6d} (Money) → D0.1_{2d} (Knowledge)
  • D6.1 (Male partner) → D5.1 (Female partner)
  • F₆ (Material) → F₂ (Wisdom)
```

### Inversion 2: DUAL (⊗)

```
Dual(E₀) = (A5.1 +_T B3.1) → (A5.1 -_T A0.1_{6a}) → A6.1

Decoded: "Divine receptivity with limiting beliefs conceals spiritual
         template from divine will"

Changes:
  • D→A (Physical→Spiritual)
  • C→B (Emotional→Mental)
  • All worlds shifted: d→a, c→b
```

### Inversion 3: COUNTER-POLE (⊕)

```
CP(E₀) = (D5.1⁻ +_T C3.1⁻) → (D5.1⁻ -_T D0.1_{6d}⁻) → D6.1⁻

Decoded: "A withholding woman with paralyzing fear hoards money
         from her dominating partner"

Changes:
  • All elements gain ⁻ pole
  • Nurturing → Withholding
  • Functional fear → Paralyzing fear
  • Protective partner → Dominating partner
```

### Inversion 4: MIRROR (⟷)

```
Mirror(E₀) = D6.1 → (D5.1 -_T D0.1_{6d}) → (D5.1 +_T C3.1)

Decoded: "The partner affects the woman hiding money,
         which causes woman and fear"

Changes:
  • Causal order reversed
  • Effect becomes cause
  • All symbols preserved
```

### Inversion 5: REVERSE-CAUSAL (⟲)

```
RC(E₀) = D6.1 → (D5.1 -_T D0.1_{6d}) → (D5.1 +_T C3.1)
         with semantic role transformation:

         D6.1 [now: initiating cause]
         → (hiding action) [mediator]
         → (woman + fear) [now: resulting effect]

Decoded: "The partner's behavior triggers money-hiding,
         which produces the woman's fear"

Changes:
  • Causal order reversed
  • Partner becomes initiator
  • Fear becomes result, not cause
```

### Inversion 6: PARALLEL ANALOGUE (∥)

**Using Love→Health analogy map:**

```
Parallel(E₀, Love→Health) =
  (D5.1 +_T C3.1) → (D5.1 -_T D2.1_{3d}) → D6.1

Decoded: "A woman fears and conceals her health status
         from her partner"

Changes:
  • F₆ (Material/Money) → F₃ (Health)
  • D0.1_{6d} (Money) → D2.1_{3d} (Health)
  • Structural pattern preserved
  • Domain shifted to health/vitality
```

## 9.3 Summary Comparison Table

| Inversion | Result Overview | Key Transformation |
|-----------|-----------------|-------------------|
| **Original** | Woman + Fear → Hides Money → Partner | Baseline |
| **Opposite** | Man + Joy → Reveals Knowledge → Woman | All polarities flipped |
| **Dual** | Spirit + Belief → Conceals Template → Will | All worlds shifted |
| **Counter-Pole** | Withholding + Paralyzing → Hoards → Dominating | Internal poles flipped |
| **Mirror** | Partner → Hiding → Woman + Fear | Causal order reversed |
| **Reverse-Causal** | Partner initiates → Hiding → Woman fears | Causal roles transformed |
| **Parallel** | Woman + Fear → Hides Health → Partner | Domain shifted |

## 9.4 Type Verification

All six inversions produce well-typed TKS expressions:

| Inversion | Type Check | Domain/Codomain |
|-----------|------------|-----------------|
| Opposite | ✓ PASS | Element → Element |
| Dual | ✓ PASS | Element → Element |
| Counter-Pole | ✓ PASS | Element± → Element∓ |
| Mirror | ✓ PASS | Graph → Graph |
| Reverse-Causal | ✓ PASS | Graph → Graph |
| Parallel | ✓ PASS | (Element × Map) → Element |

---

# Section 10: Integration with Scenario Inversion Knob

## 10.1 Mapping to Knob Axes

The six inversion types map to the Knob's axis system:

| Inversion Type | Primary Axis | Secondary Axes |
|----------------|--------------|----------------|
| Opposite | Element (E) | Noetic (N), Foundation (F) |
| Dual | World (W) | - |
| Counter-Pole | (New) Pole (±) | - |
| Mirror | (Structural) | - |
| Reverse-Causal | (Structural) | Noetic (8↔9) |
| Parallel | Foundation (F) | (with TargetProfile) |

## 10.2 Extended Knob API

```typescript
type InversionType =
  | "Opposite"
  | "Dual"
  | "CounterPole"
  | "Mirror"
  | "ReverseCausal"
  | "Parallel"

function ApplyInversionType(
  expr: TKS_Expr,
  type: InversionType,
  options?: {
    analogyMap?: AnalogyMap,  // For Parallel
    preserveSelfDuals?: boolean  // For Soft mode
  }
): TKS_Expr
```

## 10.3 CLI Extension

```
============================================================
  TKS SCENARIO INVERSION KNOB v2.0 (Extended)
============================================================

Select inversion type:
  [1] Opposite     - Semantic opposites (2↔3, 5↔6, 8↔9)
  [2] Dual         - Cross-world (A↔D, B↔C)
  [3] Counter-Pole - Internal pole (+/-)
  [4] Mirror       - Structural reflection
  [5] Reverse-Causal - Causal role transformation
  [6] Parallel     - Domain analogy
  [7] Custom       - Select axes manually
```

---

# Section 11: Status Summary

## 11.1 Fully Specified

| Component | Status |
|-----------|--------|
| Opposite Inversion | ✓ Complete |
| Dual Inversion | ✓ Complete |
| Counter-Pole Inversion | ✓ Complete |
| Mirror Inversion | ✓ Complete |
| Reverse-Causal Inversion | ✓ Complete |
| Parallel Analogue Inversion | ✓ Complete |
| Composition Rules | ✓ Complete |
| Validation Tests | ✓ Complete |
| Master Test Page | ✓ Complete |

## 11.2 Ambiguities Resolved

| Ambiguity | Resolution |
|-----------|------------|
| Counter-Pole default state | Unmarked = Constructive (⁺) |
| Mirror operator handling | Reverse operand order |
| RC semantic transformation | 8↔9 swap in cause/effect positions |
| Parallel with unmapped elements | Preserve unchanged |

## 11.3 Recommended TODOs for v2

| Priority | TODO |
|----------|------|
| High | Full Counter-Pole sense table |
| High | Analogy map library (all 21 Foundation pairs) |
| Medium | Composition optimization |
| Medium | Partial inversion (apply to subtrees) |
| Low | Inversion strength gradients |

---

*End of TKS Six Inversion Types Specification v1.0*
