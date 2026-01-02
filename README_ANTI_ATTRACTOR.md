# TKS Anti-Attractor Synthesis Module

Generate counter-scenario expressions that represent escape routes from attractor basins in the TKS (Tootra Knowledge System).

## Quick Start

```python
from anti_attractor import anti_attractor
from scenario_inversion import parse_equation, DecodeStory

# Create an attractor expression
expr = parse_equation("B2 -> C2 +T D2")
print(DecodeStory(expr))
# "Positive belief. This leads to joy and health."

# Generate counter-scenario
counter = anti_attractor(expr)
print(DecodeStory(counter))
# "Fear. This leads to limiting belief. This leads to spiritual misalignment."
```

## Installation

The module is part of the TKS project. No additional installation required.

## Features

- **Attractor Signature Computation**: Analyze TKS expressions to extract statistical patterns
- **Canonical Inversion**: Apply TKS world/noetic/foundation inversions
- **Counter-Scenario Synthesis**: Generate alternative scenarios that pull away from attractors
- **Polarity Analysis**: Identify positive/negative orientation
- **CLI Integration**: Use via command-line interface
- **Full Validation**: All outputs validated against canonical TKS rules

## API

### Main Functions

```python
# One-step anti-attractor synthesis
from anti_attractor import anti_attractor
counter = anti_attractor(expr)

# Step-by-step analysis
from anti_attractor import compute_attractor_signature, invert_signature, synthesize_counter_scenario

sig = compute_attractor_signature(expr)
inv_sig = invert_signature(sig)
counter = synthesize_counter_scenario(inv_sig)

# High-level wrapper
from scenario_inversion import AntiAttractorInvert
result = AntiAttractorInvert(expr, return_signature=True)
```

## CLI Usage

```bash
# From equation
python scripts/run_scenario_inversion.py --equation "B2 -> C2 +T D2" --anti-attractor

# From story
python scripts/run_scenario_inversion.py --story "She loved him" --anti-attractor

# JSON output
python scripts/run_scenario_inversion.py --equation "C3 -> B3" --anti-attractor --format json
```

## Examples

Run the comprehensive demo:

```bash
python examples/anti_attractor_demo.py
```

Or run the test suite:

```bash
python test_anti_attractor.py
```

## Use Cases

### 1. Breaking Negative Thought Patterns

```python
# Identify negative pattern
negative_pattern = parse_equation("C3 -> B3 -> D3")
# "Fear. This leads to limiting belief. This leads to illness."

# Generate positive counter-pattern
positive_counter = anti_attractor(negative_pattern)
# "Joy. This leads to positive belief. This leads to health."
```

### 2. Therapeutic Interventions

```python
# Analyze attractor signature
sig = compute_attractor_signature(expr)
print(f"Polarity: {sig.polarity}")  # -1 (negative)
print(f"Dominant world: {sig.dominant_world}")  # "C" (Emotional)

# Create escape route
counter = anti_attractor(expr)
```

### 3. Foundation Transformation

```python
from scenario_inversion import TKSExpression

# Lust-driven pattern (F7)
expr = TKSExpression(
    elements=["D7", "C7"],
    ops=["->"],
    foundations=[(7, None)],  # Lust
    acquisitions=[],
    raw=""
)

counter = anti_attractor(expr)
# Counter will have F1 (Unity) foundation instead of F7 (Lust)
```

## Inversion Rules

| Dimension | Rule | Examples |
|-----------|------|----------|
| World | A↔D, B↔C | Spiritual↔Physical, Mental↔Emotional |
| Noetic | 2↔3, 5↔6, 8↔9 | Positive↔Negative, Female↔Male, Cause↔Effect |
| Foundation | 1↔7, 2↔6, 3↔5, 4↔4 | Unity↔Lust, Wisdom↔Material, Life↔Power, Companionship (self-dual) |
| Polarity | +1↔-1 | Positive↔Negative |

## Documentation

See `docs/anti_attractor_guide.md` for comprehensive documentation including:
- Detailed API reference
- Algorithm details
- Theoretical background
- Advanced examples
- Integration guide

## Files

```
anti_attractor.py                  # Core module
test_anti_attractor.py             # Test suite
examples/anti_attractor_demo.py    # Comprehensive demos
docs/anti_attractor_guide.md       # Full documentation
scripts/run_scenario_inversion.py  # CLI with --anti-attractor flag
```

## Testing

```bash
# Run full test suite
python test_anti_attractor.py

# Expected output:
# [PASS] Test 1 passed
# [PASS] Test 2 passed
# [PASS] Test 3 passed
# [PASS] Test 4 passed
# [PASS] Test 5 passed
# [PASS] Test 6 passed
# ALL TESTS PASSED [SUCCESS]
```

## Dependencies

- `scenario_inversion`: TKS expression handling
- `inversion.engine`: Canonical inversion mappings
- `narrative.constants`: Validation functions

All dependencies are part of the TKS project.

## Theory

Anti-attractor synthesis is based on:

1. **Attractor Basin Analysis**: Identify system states that pull toward specific configurations
2. **Canonical Inversion Algebra**: Apply TKS opposite/dual operations systematically
3. **Frequency-Based Selection**: Most common patterns define attractor strength
4. **Polarity Dynamics**: Positive/negative valence as fundamental organizing principle

The generated counter-scenarios represent configurations that would move the system away from the original attractor basin, providing therapeutic or exploratory pathways.

## Contributing

This module follows TKS coding standards:
- All generated elements/operators validated against canonical rules
- Comprehensive test coverage
- Full documentation with examples
- Integration with existing TKS tools

## License

Part of the TKS (Tootra Knowledge System) project.

## Contact

For questions or issues, refer to the main TKS project documentation.
