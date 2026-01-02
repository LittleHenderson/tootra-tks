# Anti-Attractor Synthesis - Quick Reference

## One-Line Usage

```python
from anti_attractor import anti_attractor
from scenario_inversion import parse_equation, DecodeStory

counter = anti_attractor(parse_equation("B2 -> C2 +T D2"))
print(DecodeStory(counter))
# "Fear. This leads to limiting belief. This leads to spiritual misalignment."
```

## Command Line

```bash
# From equation
python scripts/run_scenario_inversion.py --equation "B2 -> C2 +T D2" --anti-attractor

# From story
python scripts/run_scenario_inversion.py --story "She loved him" --anti-attractor

# JSON output
python scripts/run_scenario_inversion.py --equation "C3 -> B3" --anti-attractor --format json
```

## Core Functions

### Main API
```python
from anti_attractor import anti_attractor

# One-step synthesis
counter_expr = anti_attractor(original_expr)
```

### Step-by-Step
```python
from anti_attractor import compute_attractor_signature, invert_signature, synthesize_counter_scenario

sig = compute_attractor_signature(expr)
inv_sig = invert_signature(sig)
counter = synthesize_counter_scenario(inv_sig)
```

### High-Level Wrapper
```python
from scenario_inversion import AntiAttractorInvert

result = AntiAttractorInvert(expr, return_signature=True)
counter = result['expr_inverted']
sig = result['signature']
```

## Inversion Rules

| Type | Rule |
|------|------|
| **World** | A↔D, B↔C |
| **Noetic** | 2↔3, 5↔6, 8↔9 (1,4,7,10 self-dual) |
| **Foundation** | 1↔7, 2↔6, 3↔5 (4 self-dual) |
| **Polarity** | +1↔-1, 0↔0 |

## Signature Fields

```python
sig = compute_attractor_signature(expr)

sig.element_counts      # Dict[(world, noetic), count]
sig.foundation_tags     # Set[int]
sig.polarity            # -1, 0, or +1
sig.ops_distribution    # Dict[operator, count]
sig.dominant_world      # "A", "B", "C", or "D"
sig.dominant_noetic     # 1-10
```

## Common Patterns

### Break Negative Pattern
```python
negative = parse_equation("C3 -> B3 -> D3")
positive = anti_attractor(negative)
```

### Check Polarity
```python
sig = compute_attractor_signature(expr)
if sig.polarity < 0:
    print("Negative attractor - generating positive counter")
    counter = anti_attractor(expr)
```

### Transform Foundation
```python
# Input has F7 (Lust)
# Counter will have F1 (Unity)
counter = anti_attractor(expr)
```

## Testing

```bash
# Run test suite
python test_anti_attractor.py

# Run demos
python examples/anti_attractor_demo.py
```

## Files

| File | Purpose |
|------|---------|
| `anti_attractor.py` | Core module |
| `test_anti_attractor.py` | Unit tests |
| `examples/anti_attractor_demo.py` | Demos |
| `docs/anti_attractor_guide.md` | Full docs |
| `README_ANTI_ATTRACTOR.md` | Overview |

## Quick Examples

### Example 1: Positive to Negative
```python
expr = parse_equation("B2 -> C2 +T D2")
# "Positive belief. This leads to joy and health."

counter = anti_attractor(expr)
# "Fear. This leads to limiting belief. This leads to spiritual misalignment."
```

### Example 2: Negative to Positive
```python
expr = parse_equation("C3 -> B3 -> D3")
# "Fear. This leads to limiting belief. This leads to illness."

counter = anti_attractor(expr)
# "Joy. This leads to positive belief. This leads to health."
```

### Example 3: With Foundations
```python
from scenario_inversion import TKSExpression

expr = TKSExpression(
    elements=["A2", "B2"],
    ops=["->"],
    foundations=[(1, None)],  # Unity
    acquisitions=[],
    raw=""
)

counter = anti_attractor(expr)
# counter.foundations will have F7 (Lust) instead of F1 (Unity)
```

## Typical Workflow

```python
# 1. Parse or create expression
expr = parse_equation("B2 -> C2 +T D2")

# 2. Compute signature (optional, for analysis)
sig = compute_attractor_signature(expr)
print(f"Polarity: {sig.polarity}")
print(f"Dominant: {sig.dominant_world}{sig.dominant_noetic}")

# 3. Generate counter-scenario
counter = anti_attractor(expr)

# 4. Decode to story
story = DecodeStory(counter)
print(f"Counter-story: {story}")
```

## Tips

- Counter-scenarios have 1-3 elements (most characteristic inverted pairs)
- Operators chosen based on original pattern (causal vs combination)
- All outputs validated against canonical TKS rules
- Self-dual elements (N1, N4, N7, N10, F4) stay unchanged
- Polarity calculated from N2/N5/N8 (positive) vs N3/N6/N9 (negative)

## Error Handling

```python
try:
    counter = anti_attractor(expr)
except ValueError as e:
    print(f"Invalid expression: {e}")
```

All generated elements/operators/foundations are validated. Invalid outputs raise `ValueError`.

## See Also

- Full documentation: `docs/anti_attractor_guide.md`
- Implementation details: `IMPLEMENTATION_SUMMARY.md`
- TKS overview: `README_ANTI_ATTRACTOR.md`
