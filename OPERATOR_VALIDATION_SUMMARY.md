# Operator Validation Implementation Summary

## Overview
Extended operator handling per TKS_Narrative_Semantics_Rulebook_v1.0.md and enforced validation for unknown operators.

## Changes Made

### 1. Rulebook Analysis
**Operators Defined in Rulebook** (lines 33-35):
- **TOOTRA Operators**: `+_T`, `-_T`, `×_T`, `/_T` (4 operators)
- **Composition**: `∘` (sequential), `→` (causal/RPM) (2 operators)
- **Reverse causal**: `←` (mentioned in contexts)

**Representation in Code**:
- `×_T` → `*T` (multiplication/intensification)
- `/_T` → `/T` (division/conflict)
- `∘` → `o` (sequential composition)
- `→` → `->` (causal arrow)
- `←` → `<-` (reverse causal)

### 2. Updated Files

#### `narrative/constants.py`
**Added operators** (lines 573-594):
- `*T`: "intensified by" (×_T from rulebook)
- `/T`: "in conflict with" (/_T from rulebook)
- `o`: "then" (∘ sequential composition from rulebook)

**New constant**:
```python
ALLOWED_OPS: Set[str] = {"+T", "-T", "*T", "/T", "o", "->", "<-", "+", "-"}
```
Total: 9 canonical operators

**Enhanced `is_valid_operator()`** (lines 1074-1084):
- Now uses `ALLOWED_OPS` set
- Added detailed docstring explaining all operator types

**Extended `VERB_TO_OPERATOR`** (lines 597-649):
- Added intensification verbs → `*T`: amplifies, intensifies, multiplies, modulates
- Added conflict verbs → `/T`: opposes, fights, divides, conflicts
- Updated sequence verbs → `o`: then, after, followed

#### `narrative/encoder.py`
**Updated imports** (lines 22-41):
- Added `ALLOWED_OPS`
- Added `is_valid_operator`

**Enhanced `parse_equation()`** (lines 520-609):
- Added `strict` parameter (default: False)
- Validates operators against `ALLOWED_OPS`
- **Strict mode**: Raises `ValueError` for unknown operators with clear error message
- **Non-strict mode**: Skips unknown operators gracefully (no error)
- Handles all new operators: `*T`, `/T`, `o`
- Improved tokenization with regex for standalone 'o' operator

#### `narrative/decoder.py`
**Updated `OPERATOR_TEMPLATES`** (lines 41-55):
- Added `*T`: "{left} intensified by {right}" / "intensified by"
- Added `/T`: "{left} in conflict with {right}" / "in conflict with"
- Added `o`: "First {left}, then {right}" / "then"

### 3. Test Coverage

#### Added 21 new tests in `tests/test_narrative_encoder.py`:

**Operator validation tests**:
1. `test_allowed_ops_canonical()` - Verify all rulebook operators in ALLOWED_OPS
2. `test_allowed_ops_count()` - Verify exactly 9 operators
3. `test_is_valid_operator_canonical()` - Test validation of canonical operators
4. `test_is_valid_operator_unknown()` - Test rejection of unknown operators
5. `test_operator_validation_strict_mode()` - Test strict mode raises ValueError
6. `test_operator_validation_non_strict_mode()` - Test non-strict mode skips unknown ops

**New operator tests**:
7. `test_new_operators_multiply()` - Test *T encoding/decoding
8. `test_new_operators_divide()` - Test /T encoding/decoding
9. `test_new_operators_composition()` - Test o encoding/decoding
10. `test_parse_equation_all_operators()` - Test parsing all operators together

**Story encoding tests**:
11. `test_encode_story_intensification()` - Test stories with intensification verbs
12. `test_encode_story_conflict()` - Test stories with conflict verbs
13. `test_encode_story_sequence()` - Test stories with sequential markers

**Template tests**:
14. `test_operator_templates_coverage()` - Verify decoder has templates for all ops
15. `test_operator_templates_multiply()` - Verify *T template has "intensified"
16. `test_operator_templates_divide()` - Verify /T template has "conflict"
17. `test_operator_templates_composition()` - Verify o template has "then"/"first"

**Verb mapping tests**:
18. `test_verb_to_operator_intensification()` - Test intensification verbs → *T
19. `test_verb_to_operator_conflict()` - Test conflict verbs → /T
20. `test_verb_to_operator_sequence()` - Test sequence verbs → o

**Integration test**:
21. `test_roundtrip_new_operators()` - Test encode→decode roundtrip with new ops

## Operator Definitions from Rulebook

### A.3.3 TOOTRA Multiplication (×_T → *T)
**Meaning**: Amplification; modulation; exponential combination
**Reading**: "X amplified by Y" / "X intensified through Y"
**Example**: `C3.1 *T C3.1` = "fear amplified by fear" = "panic"

### A.3.4 TOOTRA Division (/_T → /T)
**Meaning**: Conflict; opposition; division of forces
**Reading**: "X opposed by Y" / "X in conflict with Y"
**Example**: `B2.1 /T B3.1` = "positive belief in conflict with limiting belief"

### A.3.5 Sequential Composition (∘ → o)
**Meaning**: Temporal sequence; "then"
**Reading**: "First X, then Y"
**Example**: `C3.1 o D7.1 o D8.1` = "fear, then habit, then elevation"

## Test Results
All 21 new tests pass successfully:
```
All narrative encoder/decoder tests passed!
```

## Validation Behavior

### Strict Mode (`strict=True`)
```python
# Raises ValueError for unknown operators
parse_equation("B5 ~T D3", strict=True)
# ValueError: Unknown operator '~T' detected in strict mode.
# Allowed operators: ['+', '+T', '-', '-T', '->', '<-', '*T', '/T', 'o']
```

### Non-Strict Mode (`strict=False`, default)
```python
# Skips unknown operators silently
expr = parse_equation("B5 ~T D3 +T C2", strict=False)
# Result: elements=["B5", "D3", "C2"], ops=["+T"]
# The ~T operator is skipped
```

## Example Usage

### Multiplication/Intensification
```python
expr = parse_equation("C3 *T C3")
story = DecodeStory(expr)
# Output: "Fear intensified by fear."
```

### Division/Conflict
```python
expr = parse_equation("B2 /T B3")
story = DecodeStory(expr)
# Output: "Positive belief in conflict with limiting belief."
```

### Sequential Composition
```python
expr = parse_equation("C3 o D7 o D8")
story = DecodeStory(expr)
# Output: "First fear, then habit. Then physical authority."
```

## Documentation
All new operators are explicitly documented in:
- Rulebook reference comments in constants.py
- Docstrings in `is_valid_operator()`
- Docstrings in `parse_equation()`
- Template definitions in decoder.py

## Backward Compatibility
- Original operators (`+T`, `-T`, `->`, `<-`, `+`, `-`) remain unchanged
- `OPERATOR_TOKENS` still exists for backward compatibility (now references `ALLOWED_OPS`)
- Default behavior (non-strict mode) is lenient and won't break existing code

## Compliance with Rulebook
All operators are **explicitly defined** in TKS_Narrative_Semantics_Rulebook_v1.0.md:
- Lines 33-35: Operator declaration
- Lines 149-253: Detailed operator semantics
- Lines 469-502: Reading templates

**No new operators were invented** - only those documented in the rulebook were added.
