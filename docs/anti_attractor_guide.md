# TKS Anti-Attractor Synthesis Guide

## Overview

The Anti-Attractor Synthesis module provides functionality to analyze TKS expressions and generate counter-scenarios that represent escape routes from attractor basins. This is useful for:

- Breaking negative thought/emotion patterns
- Creating therapeutic interventions
- Understanding system dynamics
- Generating alternative scenarios

## Core Concepts

### Attractor Signature

An attractor signature captures the statistical and structural characteristics of a TKS expression:

- **Element Counts**: Frequency distribution of (world, noetic) pairs
- **Foundation Tags**: Set of foundation IDs present in the expression
- **Polarity**: Overall valence (-1 negative, 0 neutral, +1 positive)
- **Ops Distribution**: Frequency of operators used
- **Dominant World/Noetic**: Most frequent world and noetic values

### Inversion Rules

The module applies canonical TKS inversions:

| Dimension | Inversion Rule | Examples |
|-----------|----------------|----------|
| **World** | A↔D, B↔C | Spiritual↔Physical, Mental↔Emotional |
| **Noetic** | 2↔3, 5↔6, 8↔9 | Positive↔Negative, Female↔Male, Cause↔Effect |
| **Foundation** | 1↔7, 2↔6, 3↔5 | Unity↔Lust, Wisdom↔Material, Life↔Power |
| **Polarity** | +1↔-1 | Positive↔Negative orientation |

Self-dual elements (N1, N4, N7, N10, F4) remain unchanged.

## API Reference

### Functions

#### `compute_attractor_signature(expr: TKSExpression) -> AttractorSignature`

Analyzes a TKS expression and extracts its attractor signature.

**Parameters:**
- `expr`: TKS expression to analyze

**Returns:**
- `AttractorSignature` object containing:
  - `element_counts`: Dict[(world, noetic), count]
  - `foundation_tags`: Set[int]
  - `polarity`: int (-1, 0, +1)
  - `ops_distribution`: Dict[str, int]
  - `dominant_world`: str
  - `dominant_noetic`: int

**Example:**
```python
from anti_attractor import compute_attractor_signature
from scenario_inversion import parse_equation

expr = parse_equation("B2 -> C2 +T D2")
sig = compute_attractor_signature(expr)

print(f"Polarity: {sig.polarity}")  # 1 (positive)
print(f"Dominant world: {sig.dominant_world}")  # "B"
```

#### `invert_signature(sig: AttractorSignature) -> AttractorSignature`

Inverts an attractor signature using canonical TKS inversions.

**Parameters:**
- `sig`: Original attractor signature

**Returns:**
- Inverted `AttractorSignature`

**Example:**
```python
from anti_attractor import compute_attractor_signature, invert_signature

sig = compute_attractor_signature(expr)
inv_sig = invert_signature(sig)

print(f"Original polarity: {sig.polarity}")  # 1
print(f"Inverted polarity: {inv_sig.polarity}")  # -1
```

#### `synthesize_counter_scenario(inv_sig: AttractorSignature) -> TKSExpression`

Synthesizes a counter-scenario expression from an inverted signature.

**Parameters:**
- `inv_sig`: Inverted attractor signature

**Returns:**
- `TKSExpression` representing the counter-scenario

**Details:**
- Selects top 1-3 most frequent inverted element pairs
- Connects them with causal arrows (`->`) or combination (`+T`)
- Attaches inverted foundations
- Validates all generated elements and operators are canonical

**Example:**
```python
from anti_attractor import compute_attractor_signature, invert_signature, synthesize_counter_scenario

sig = compute_attractor_signature(expr)
inv_sig = invert_signature(sig)
counter = synthesize_counter_scenario(inv_sig)

print(counter.elements)  # ['C3', 'B3', 'A3']
```

#### `anti_attractor(expr: TKSExpression) -> TKSExpression`

**Main Entry Point**: Generates a counter-scenario for a given attractor in one step.

**Parameters:**
- `expr`: Original TKS expression (attractor)

**Returns:**
- Counter-scenario `TKSExpression` (anti-attractor)

**Example:**
```python
from anti_attractor import anti_attractor
from scenario_inversion import parse_equation, DecodeStory

# Original expression
expr = parse_equation("B2 -> C2 +T D2")
print(DecodeStory(expr))
# "Positive belief. This leads to joy and health."

# Generate counter-scenario
counter = anti_attractor(expr)
print(DecodeStory(counter))
# "Fear. This leads to limiting belief. This leads to spiritual misalignment."
```

### Wrapper Function in scenario_inversion.py

#### `AntiAttractorInvert(expr: TKSExpression, return_signature: bool = False) -> Dict[str, Any]`

High-level wrapper for anti-attractor synthesis, consistent with other inversion functions.

**Parameters:**
- `expr`: TKS expression to invert
- `return_signature`: If True, include attractor signature in result

**Returns:**
- Dictionary with keys:
  - `expr_inverted`: Counter-scenario expression
  - `signature`: Attractor signature (if `return_signature=True`)

**Example:**
```python
from scenario_inversion import parse_equation, AntiAttractorInvert

expr = parse_equation("C3 -> B3 -> D3")
result = AntiAttractorInvert(expr, return_signature=True)

print(result['signature'].polarity)  # -1 (negative)
print(result['expr_inverted'].elements)  # ['B2', 'C2', 'A2']
```

## CLI Usage

The `run_scenario_inversion.py` script supports anti-attractor synthesis via the `--anti-attractor` flag.

### Basic Usage

```bash
# From equation
python scripts/run_scenario_inversion.py --equation "B2 -> C2 +T D2" --anti-attractor

# From story
python scripts/run_scenario_inversion.py --story "She loved him" --anti-attractor
```

### Output Format

```
============================================================
  TKS ANTI-ATTRACTOR SYNTHESIS
============================================================

=== ORIGINAL ===
Equation: B2 -> C2 +T D2

=== ATTRACTOR SIGNATURE ===
Element counts: {('B', 2): 1, ('C', 2): 1, ('D', 2): 1}
Dominant world: B
Dominant noetic: N2
Polarity: 1 (positive)
Foundation tags: []

=== INVERTED ===
Equation: C3 -> B3 -> A3
Story: Fear. This leads to limiting belief. This leads to spiritual misalignment.

============================================================
```

### JSON Output

```bash
python scripts/run_scenario_inversion.py --equation "B2 -> C2" --anti-attractor --format json
```

## Use Cases

### 1. Breaking Negative Patterns

**Scenario**: Someone stuck in a fear-based pattern

```python
# Negative attractor
expr = parse_equation("C3 -> B3 -> D3")
# Story: "Fear. This leads to limiting belief. This leads to illness."

# Generate escape route
counter = anti_attractor(expr)
# Story: "Joy. This leads to positive belief. This leads to health."
```

### 2. Therapeutic Interventions

**Scenario**: Creating alternative pathways from negative foundations

```python
from scenario_inversion import TKSExpression

# Lust-driven pattern (F7)
expr = TKSExpression(
    elements=["D7", "C7", "B7"],
    ops=["->", "->"],
    foundations=[(7, None)],
    acquisitions=[],
    raw=""
)

counter = anti_attractor(expr)
# Counter will have F1 (Unity) foundation instead of F7 (Lust)
```

### 3. System Analysis

**Scenario**: Understanding polarity dynamics

```python
sig = compute_attractor_signature(expr)

if sig.polarity < 0:
    print("System is in negative attractor basin")
    counter = anti_attractor(expr)
    print(f"Positive escape route: {DecodeStory(counter)}")
```

## Algorithm Details

### Polarity Calculation

Polarity is calculated by counting positive vs negative noetics:

- **Positive noetics**: N2 (Positive), N5 (Female), N8 (Cause)
- **Negative noetics**: N3 (Negative), N6 (Male), N9 (Effect)

```
if positive_count > negative_count:
    polarity = +1
elif negative_count > positive_count:
    polarity = -1
else:
    polarity = 0
```

### Element Selection

The synthesis algorithm selects 1-3 elements based on frequency:

1. Sort inverted element pairs by count (descending)
2. Take top 1-3 pairs (at least 1, at most 3)
3. Create canonical TKS elements from pairs

### Operator Selection

Operators are chosen based on original expression patterns:

- If causal operators (`->`, `<-`) dominate: use causal chain
- Otherwise: use combination operators (`+T`)

## Validation

All generated elements, operators, and foundations are validated against canonical TKS rules:

- **Worlds**: Must be A, B, C, or D
- **Noetics**: Must be 1-10
- **Foundations**: Must be 1-7
- **Operators**: Must be in `ALLOWED_OPS` set

Invalid outputs raise `ValueError` with descriptive messages.

## Examples

See `examples/anti_attractor_demo.py` for comprehensive demonstrations:

```bash
python examples/anti_attractor_demo.py
```

Demos include:
1. Basic anti-attractor synthesis
2. Inverting negative attractors
3. Foundation inversion
4. Multi-world attractor inversion
5. Polarity analysis

## Integration with Existing Tools

The anti-attractor module integrates seamlessly with existing TKS tools:

```python
from scenario_inversion import InvertStory, AntiAttractorInvert

# Standard inversion (axis-based)
result1 = InvertStory("She loved him", axes={"Element"}, mode="soft")

# Anti-attractor synthesis (signature-based)
result2 = AntiAttractorInvert(parse_equation("C2 -> D5 +T D6"))

# Both return compatible TKSExpression objects
```

## Theoretical Background

Anti-attractor synthesis is based on:

1. **Dynamical Systems Theory**: Identifying escape routes from basins
2. **TKS Inversion Algebra**: Canonical opposite/dual operations
3. **Frequency Analysis**: Most common patterns define attractor strength
4. **Polarity Dynamics**: Positive/negative valence as fundamental axis

The synthesized counter-scenarios represent configurations that would pull the system away from the original attractor basin, providing therapeutic or exploratory value.

## Limitations

- Only considers top 1-3 most frequent elements (simplification)
- Does not account for temporal dynamics or causal chains
- Assumes canonical inversions are therapeutically relevant
- No validation against narrative coherence

## Future Enhancements

Potential improvements:
- Weighted element selection based on clinical relevance
- Temporal unfolding of counter-scenarios
- Integration with narrative coherence checking
- Multi-stage escape route planning
- Attractor basin visualization
