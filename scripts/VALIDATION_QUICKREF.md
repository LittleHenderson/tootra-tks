# TKS Canonical Validation - Quick Reference

## Import
```python
from scripts.canonical_validator import (
    validate_canonical,
    validate_entry,
    compute_validation_metrics
)
```

## Validate Expression
```python
from narrative import EncodeStory

expr = EncodeStory("A teacher causes growth")
is_valid, errors = validate_canonical(expr)

if not is_valid:
    for error in errors:
        print(error)
```

## Validate Entry Dict
```python
entry = {
    "expr_elements": ["A5", "D8"],
    "expr_ops": ["->"]
}

is_valid, errors = validate_entry(entry)
```

## Compute Metrics
```python
entries = load_jsonl("augmented.jsonl")
metrics = compute_validation_metrics(entries)

print(f"Pass rate: {metrics['pass_rate']:.1%}")
```

## Canon Constraints

### Worlds
- **Valid**: A, B, C, D
- **Mapping**: A=Spiritual, B=Mental, C=Emotional, D=Physical

### Noetics
- **Valid**: 1-10
- **Involutions**: 2↔3, 5↔6, 8↔9
- **Self-duals**: 1, 4, 7, 10

### Foundations
- **Valid**: 1-7
- **Opposites**: 1↔7, 2↔6, 3↔5, 4=4

### Operators
- **Valid**: `+`, `-`, `+T`, `-T`, `->`, `<-`, `*T`, `/T`, `o`
- **TOOTRA**: `+T`, `-T`, `*T`, `/T`
- **Causal**: `->`, `<-`
- **Sequential**: `o`
- **Basic**: `+`, `-`

## Error Messages

```
Element 0 'X5': Invalid world 'X' (must be A/B/C/D)
Element 1 'D15': Invalid noetic '15' (must be 1-10)
Operator 0 '++': Invalid operator (must be in {+, -, +T, ...})
Foundation 2: Invalid foundation '8' (must be 1-7)
Structural inconsistency: 3 elements require 2 operators, got 1
```

## Testing
```bash
python tests/test_canonical_validator.py
python tests/fuzz_pipeline.py
```

## Files
- `scripts/canonical_validator.py` - Core validator
- `tests/test_canonical_validator.py` - Test suite
- `docs/Agent3_Validation_Implementation.md` - Full docs
