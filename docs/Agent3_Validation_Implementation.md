# Agent 3: Canonical Validation Implementation

## Overview

This document describes the implementation of canonical validation for the TKS augmented data pipeline.

## Canonical Constraints

The TKS system has strict canonical constraints that all generated data must satisfy:

### Worlds
- **Valid values**: A, B, C, D only
- **Mapping**:
  - A = Spiritual
  - B = Mental
  - C = Emotional
  - D = Physical

### Noetics
- **Valid values**: 1-10
- **Involution pairs**:
  - 2 ↔ 3 (Positive ↔ Negative)
  - 5 ↔ 6 (Female ↔ Male)
  - 8 ↔ 9 (Cause ↔ Effect)
- **Self-duals**: 1, 4, 7, 10 (Mind, Vibration, Rhythm, Idea)

### Foundations
- **Valid values**: 1-7
- **Opposites**:
  - 1 ↔ 7 (Unity ↔ Lust)
  - 2 ↔ 6 (Wisdom ↔ Material)
  - 3 ↔ 5 (Life ↔ Power)
  - 4 = 4 (Companionship is self-dual)

### Operators
- **Valid operators**: `+`, `-`, `+T`, `-T`, `->`, `<-`, `*T`, `/T`, `o`
- **Semantics**:
  - `+T` = together with (TOOTRA combine)
  - `-T` = without (TOOTRA remove)
  - `*T` = intensified by (TOOTRA multiply)
  - `/T` = in conflict with (TOOTRA divide)
  - `o` = then (sequential composition)
  - `->` = causes (causal forward)
  - `<-` = caused by (causal reverse)
  - `+` = and (basic combine)
  - `-` = minus (basic subtract)

## Implementation

### Module: `scripts/canonical_validator.py`

Contains the core validation logic:

#### Function: `validate_canonical(expr) -> Tuple[bool, List[str]]`

Validates a TKS expression against canonical constraints.

**Parameters:**
- `expr`: TKSExpression object (from narrative or scenario_inversion module)

**Returns:**
- `(is_valid, errors)`: Tuple of boolean and list of error messages

**Validation checks:**
1. All element worlds in {A, B, C, D}
2. All noetics in {1..10}
3. All foundations in {1..7} (if present)
4. All operators in ALLOWED_OPS
5. Structural consistency (len(ops) == len(elements) - 1)

**Example:**
```python
from canonical_validator import validate_canonical
from narrative import EncodeStory

expr = EncodeStory("A spiritual teacher causes growth")
is_valid, errors = validate_canonical(expr)

if not is_valid:
    print(f"Validation failed:")
    for error in errors:
        print(f"  - {error}")
```

#### Function: `validate_entry(entry) -> Tuple[bool, List[str]]`

Validates a data entry dict with `expr_elements` and `expr_ops` fields.

**Parameters:**
- `entry`: Dict with keys `expr_elements` (list of element strings) and `expr_ops` (list of operator strings)

**Returns:**
- `(is_valid, errors)`: Tuple of boolean and list of error messages

**Example:**
```python
entry = {
    "story": "A teacher causes growth",
    "expr_elements": ["A5", "D8"],
    "expr_ops": ["->"]
}

is_valid, errors = validate_entry(entry)
```

#### Function: `compute_validation_metrics(entries) -> dict`

Computes aggregate validation metrics for a corpus.

**Parameters:**
- `entries`: List of entry dicts

**Returns:**
- Dict with metrics:
  - `total`: Total entries
  - `valid`: Count of valid entries
  - `invalid`: Count of invalid entries
  - `pass_rate`: Validation pass rate (0-1)
  - `error_counts`: Dict of error type -> count

**Example:**
```python
metrics = compute_validation_metrics(augmented_entries)
print(f"Pass rate: {metrics['pass_rate']:.1%}")
print(f"Error breakdown: {metrics['error_counts']}")
```

## Integration with Augmentation Pipeline

### Validation Points

The augmentation pipeline (`scripts/generate_augmented_data.py`) validates at multiple stages:

1. **Input validation**: Check original corpus entries
2. **Output validation**: Check each generated augmentation
3. **Final validation**: Compute aggregate metrics

### Strict vs Lenient Mode

**Strict mode** (default):
- Invalid entries are rejected and not included in output
- Validation failures are logged as errors
- Pipeline fails if pass rate < min_pass_rate threshold

**Lenient mode**:
- Invalid entries are included in output with `validator_pass=False` flag
- Validation failures are logged as warnings
- Pipeline continues regardless of pass rate

### Configuration

```python
config = AugmentationConfig(
    validate_canonical=True,     # Enable validation
    min_pass_rate=0.90,          # 90% minimum pass rate (strict mode)
)
```

### Output Format

Each entry in the augmented corpus includes validation metadata:

```json
{
  "story": "Inverted story text",
  "expr": "B3 -> D5",
  "expr_elements": ["B3", "D5"],
  "expr_ops": ["->"],
  "aug_type": "inverted",
  "axes": ["W", "N"],
  "validator_pass": true,
  "validation_errors": []
}
```

### Metrics

The augmentation metrics JSON includes validation statistics:

```json
{
  "validator_pass_rate": 0.95,
  "world_validity": 0.98,
  "noetic_validity": 0.97,
  "operator_validity": 1.0,
  "structural_validity": 0.96,
  "validation_failures": 50
}
```

## Usage Examples

### Example 1: Validate single expression

```python
from narrative import EncodeStory
from scripts.canonical_validator import validate_canonical

# Encode story
expr = EncodeStory("A spiritual teacher causes emotional growth")

# Validate
is_valid, errors = validate_canonical(expr)

if is_valid:
    print("Expression is canon-valid!")
else:
    print(f"Validation failed with {len(errors)} errors:")
    for error in errors:
        print(f"  - {error}")
```

### Example 2: Validate augmented corpus

```python
from scripts.canonical_validator import compute_validation_metrics
from scripts.generate_augmented_data import load_jsonl

# Load augmented data
entries = load_jsonl("data/pilot/augmented.jsonl")

# Compute metrics
metrics = compute_validation_metrics(entries)

print(f"Validation Summary:")
print(f"  Total entries:    {metrics['total']}")
print(f"  Valid entries:    {metrics['valid']}")
print(f"  Invalid entries:  {metrics['invalid']}")
print(f"  Pass rate:        {metrics['pass_rate']:.1%}")
print(f"  Error breakdown:")
for error_type, count in metrics['error_counts'].items():
    print(f"    {error_type}: {count}")
```

### Example 3: Filter invalid entries

```python
from scripts.canonical_validator import validate_entry
from scripts.generate_augmented_data import load_jsonl, write_jsonl

# Load data
entries = load_jsonl("data/pilot/augmented.jsonl")

# Filter to valid only
valid_entries = []
for entry in entries:
    is_valid, errors = validate_entry(entry)
    if is_valid:
        valid_entries.append(entry)

# Save filtered corpus
write_jsonl("data/pilot/augmented_valid_only.jsonl", valid_entries)

print(f"Filtered {len(entries)} -> {len(valid_entries)} entries")
```

## Testing

### Unit Tests

The validator is tested in `tests/fuzz_pipeline.py` which validates the complete encode → invert → decode pipeline.

Run tests:
```bash
cd C:\Users\wakil\downloads\everthing-tootra-tks
python tests/fuzz_pipeline.py
```

Expected output:
```
================================================================================
TKS PIPELINE FUZZ TESTING
================================================================================
Testing 14 stories × 4 modes
= 56 total tests

[PASS] | Simple emotion                | Opposite
[PASS] | Simple emotion                | Dual
[PASS] | Simple emotion                | Mirror
[PASS] | Simple emotion                | ReverseCausal
...

================================================================================
SUMMARY
================================================================================
Total tests:  56
Passed:       56
Failed:       0
Pass rate:    100.0%

CANONICAL VALIDATION:
  Allowed worlds:      ['A', 'B', 'C', 'D']
  Allowed noetics:     1-10
  Allowed foundations: 1-7
  Allowed operators:   ['+', '+T', '-', '-T', '->', '<-', '*T', '/T', 'o']
```

## Error Messages

The validator provides detailed error messages for debugging:

### Invalid World
```
Element 0 'X5': Invalid world 'X' (must be A/B/C/D)
```

### Invalid Noetic
```
Element 1 'D15': Invalid noetic '15' (must be 1-10)
```

### Invalid Operator
```
Operator 0 '++': Invalid operator (must be in {'+', '-', '+T', '-T', '->', '<-', '*T', '/T', 'o'})
```

### Invalid Foundation
```
Foundation 2: Invalid foundation '8' (must be 1-7)
```

### Structural Error
```
Structural inconsistency: 3 elements require 2 operators, got 1
```

## Best Practices

1. **Always validate before saving**: Run validation on all generated data before writing to disk
2. **Track metrics**: Monitor validation pass rates to detect pipeline issues
3. **Use strict mode in production**: Set `min_pass_rate=0.95` or higher for training data
4. **Log failures**: Save validation errors for debugging and analysis
5. **Test edge cases**: Validate boundary conditions (e.g., noetic=10, foundation=7)

## Future Enhancements

Potential improvements for Phase 2:

1. **Semantic validation**: Check for semantically invalid combinations (e.g., A10 +T A10)
2. **Foundation consistency**: Validate element-foundation alignments
3. **Sense validation**: Check sense indices are valid for element type
4. **Acquisition validation**: Verify acquisition markers are consistent
5. **Performance optimization**: Batch validation for large corpora
6. **Auto-correction**: Attempt to fix common validation errors

## References

- `narrative/constants.py`: Canonical mappings (WORLDS, NOETIC_NAMES, FOUNDATIONS, ALLOWED_OPS)
- `tests/fuzz_pipeline.py`: Validation function examples
- `TKS_Narrative_Semantics_Rulebook_v1.0.md`: Canonical semantics specification
- `TKS_Symbol_Sense_Table_v1.0.md`: Element sense mappings

## Summary

The canonical validation system ensures all augmented data adheres to TKS formal constraints. The `validate_canonical()` function provides robust checking of worlds, noetics, foundations, and operators, with detailed error reporting. Integration with the augmentation pipeline enables quality control at multiple stages, supporting both strict and lenient validation modes.
