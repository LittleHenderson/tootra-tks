# TKS Operator Selection Heuristics - Quick Reference

## Overview
The `_choose_operator_for_pair()` function in `anti_attractor.py` selects operators deterministically based on element noetics.

## Rule Priority (highest to lowest)

### Rule 1: Cause → Effect (Priority 1)
**Trigger**: N8 → N9
**Operator**: `->`
**Example**: `B8 -> D9` (mental cause leads to physical effect)
**Rationale**: Natural causal flow in TKS narrative

### Rule 2: Rhythm/Pattern (Priority 2)
**Trigger**: N7 → N7
**Operator**: `o` (sequential composition)
**Example**: `D7 o C7` (physical habit then emotional pattern)
**Rationale**: Sequential/cyclical patterns use composition

### Rule 3: Same Polarity (Priority 3)
**Trigger**: Both positive OR both negative
**Operator**: `*T` (intensify)
**Polarity Sets**:
- Positive: {N2, N5, N8} (Positive, Female, Cause)
- Negative: {N3, N6, N9} (Negative, Male, Effect)

**Examples**:
- `B2 *T D2` (positive belief intensified by positive health)
- `C3 *T A3` (fear intensified by spiritual misalignment)

**Rationale**: Same-polarity elements amplify each other

### Rule 4: Opposite Polarity (Priority 4)
**Trigger**: Positive → Negative OR Negative → Positive
**Operator**: `/T` (conflict)
**Examples**:
- `B2 /T C3` (positive belief conflicts with fear)
- `D5 /T A6` (female receptivity conflicts with male projection)

**Rationale**: Opposite polarities create conflict/opposition

### Rule 5: Neutral Noetics (Priority 5)
**Trigger**: Either element has neutral noetic
**Operator**: `+T` (combine together)
**Neutral Set**: {N1, N4, N7, N10} (Mind, Vibration, Rhythm, Idea)

**Examples**:
- `B1 +T D4` (mental clarity combined with physical energy)
- `A10 +T C5` (divine blueprint combined with emotional receptivity)

**Rationale**: Neutral elements combine harmoniously

### Rule 6: Default Fallback (Priority 6)
**Trigger**: None of above rules apply
**Operator**: `->`
**Rationale**: Default to causal narrative flow

## Noetic Classification

### Positive Noetics (attracts, orders, elevates)
- **N2**: Positive (joy, alignment, order)
- **N5**: Female (receptive, nurturing, open)
- **N8**: Cause (trigger, elevation, above)

### Negative Noetics (repels, disorders, grounds)
- **N3**: Negative (fear, misalignment, chaos)
- **N6**: Male (projective, assertive, structure)
- **N9**: Effect (result, consequence, below)

### Neutral Noetics (self-dual, transformative)
- **N1**: Mind (awareness, consciousness)
- **N4**: Vibration (intensity, energy)
- **N7**: Rhythm (pattern, cycle, repetition)
- **N10**: Idea (concept, template, potential)

## Complete Operator Set (ALLOWED_OPS)

### TOOTRA Operators
- `+T`: Together with (combination)
- `-T`: Without (removal)
- `*T`: Intensified by (amplification)
- `/T`: In conflict with (opposition)

### Composition Operators
- `o`: Then (sequential composition)
- `->`: Causes (forward causal)
- `<-`: Caused by (reverse causal)

### Basic Operators
- `+`: And (addition)
- `-`: Minus (subtraction)

## Usage in Counter-Scenario Synthesis

When `synthesize_counter_scenario()` is called with `prefer_causal=True`:
1. Elements are selected by frequency from inverted signature
2. For each adjacent pair of elements, `_choose_operator_for_pair()` is called
3. Operator is chosen deterministically based on rules above
4. Result: varied, semantically appropriate expressions

## Example Transformation

**Before** (monotonous):
```
Original: B2 -> B2 -> D5
Counter:  C3 -> C3 -> A6  (all arrows)
```

**After** (with heuristics):
```
Original: B2 -> B2 -> D5
Counter:  C3 *T C3 /T A6  (varied operators)
          (neg) (intensify same) (conflict opposite)
```

## Testing Heuristics

Run demonstration:
```bash
python demo_anti_attractor_improvements.py
```

Run full test suite:
```bash
python tests/test_anti_attractor.py
```

## Canon Compliance

All operators selected are guaranteed to be in ALLOWED_OPS.
No operator outside the canonical set will ever be generated.

**Guardrails**:
- ✓ Worlds: A/B/C/D only
- ✓ Noetics: 1-10 only
- ✓ Operators: ALLOWED_OPS only
- ✓ Foundations: 1-7 only
- ✓ Deterministic (no randomness)

## Design Philosophy

1. **Semantic Appropriateness**: Operators reflect element relationships
2. **Variety**: Use full range of TKS operators
3. **Determinism**: Same inputs always produce same outputs
4. **Simplicity**: Clear, prioritized rules
5. **Canon Compliance**: Strict adherence to TKS constraints
