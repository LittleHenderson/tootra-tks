# TKS Data Sanitizer Guide

## Overview

The TKS Data Sanitizer is a quality assurance tool for Phase 6 of the TKS project. It validates augmented JSONL files against canonical TKS constraints, detects data quality issues, and provides options for cleaning or flagging problematic entries.

## Canonical Guardrails

The sanitizer enforces strict canonical constraints:

- **Worlds**: A, B, C, D only
- **Noetics**: 1-10 with involution pairs (2↔3, 5↔6, 8↔9) and self-duals (1, 4, 7, 10)
- **Foundations**: 1-7
- **Operators**: `+`, `-`, `+T`, `-T`, `->`, `<-`, `*T`, `/T`, `o` (9 total)
- **Required Fields**: `id`, `story`, `expr`, `aug_type`, `validator_pass`
- **Code Style**: ASCII only, deterministic, type-safe

## Features

### Detection Capabilities

1. **Duplicate Detection**
   - By ID: Finds entries with identical `id` fields
   - By Content Hash: Detects entries with identical content (story, expr, aug_type) but different IDs

2. **Canonical Validation**
   - Invalid operators (not in ALLOWED_OPS)
   - Invalid worlds (not in A/B/C/D)
   - Invalid noetics (not in 1-10)
   - Missing required fields
   - Structural inconsistencies (mismatch between elements and operators)

3. **Reporting**
   - Detailed JSON reports with issue categorization
   - Human-readable console summaries
   - Pass rate calculation
   - Issue severity levels (error, warning, info)

## Installation & Setup

The sanitizer is located at `scripts/sanitize_augmented.py` and requires no additional dependencies beyond the base TKS project.

```bash
# Ensure you're in the project root
cd C:\Users\wakil\downloads\everthing-tootra-tks

# Run tests to verify installation
python -m pytest tests/test_sanitize_augmented.py -v
```

## Usage

### Basic Usage

**Scan and report only (no modifications):**
```bash
python scripts/sanitize_augmented.py \
  --input output/teacher_augmented.jsonl \
  --flag-only
```

**Generate detailed JSON report:**
```bash
python scripts/sanitize_augmented.py \
  --input output/teacher_augmented.jsonl \
  --report output/sanitizer_report.json \
  --flag-only
```

**Clean data and save to new file:**
```bash
python scripts/sanitize_augmented.py \
  --input output/teacher_augmented.jsonl \
  --output output/teacher_augmented_clean.jsonl \
  --drop-invalid
```

**Scan, clean, and generate report:**
```bash
python scripts/sanitize_augmented.py \
  --input output/teacher_augmented.jsonl \
  --output output/teacher_augmented_clean.jsonl \
  --report output/sanitizer_report.json \
  --drop-invalid
```

### Command-Line Options

- `--input FILE`: Input JSONL file to scan (required)
- `--output FILE`: Output file for cleaned data (optional)
- `--drop-invalid`: Remove invalid entries from output (requires --output)
- `--flag-only`: Report issues without removing entries
- `--report FILE`: Save detailed JSON report to file (optional)

### Exit Codes

- `0`: Success (all entries clean or flag-only mode)
- `1`: Issues found and not in flag-only mode

## Pipeline Integration

### Where to Insert the Sanitizer

The sanitizer should be inserted at two key points in the data pipeline:

#### 1. Post-Teacher Generation (Recommended)

**Location**: After teacher model generates initial stories
**Script**: After running story generation but before augmentation

```bash
# Example workflow:
# Step 1: Generate teacher stories
python scripts/generate_teacher_stories.py \
  --output output/teacher_outputs.jsonl

# Step 2: SANITIZE - Check teacher outputs
python scripts/sanitize_augmented.py \
  --input output/teacher_outputs.jsonl \
  --report output/teacher_sanitizer_report.json \
  --flag-only

# Step 3: Run augmentation (scenario inversion, anti-attractor)
python scripts/augment_data.py \
  --input output/teacher_outputs.jsonl \
  --output output/teacher_augmented.jsonl
```

**Benefits**:
- Catches issues early before augmentation multiplies them
- Prevents invalid data from being augmented
- Faster debugging (smaller dataset)

#### 2. Post-Augmentation (Critical)

**Location**: After all augmentation operations (inversion, anti-attractor, etc.)
**Script**: Before training or final validation

```bash
# Example workflow:
# Step 1: Run augmentation
python scripts/augment_data.py \
  --input output/teacher_outputs.jsonl \
  --output output/teacher_augmented.jsonl

# Step 2: SANITIZE - Check augmented data
python scripts/sanitize_augmented.py \
  --input output/teacher_augmented.jsonl \
  --output output/teacher_augmented_clean.jsonl \
  --report output/sanitizer_report.json \
  --drop-invalid

# Step 3: Proceed with training or validation
python scripts/train.py \
  --input output/teacher_augmented_clean.jsonl
```

**Benefits**:
- Final quality gate before training
- Detects augmentation-introduced errors
- Ensures training data is 100% canonical

### Recommended Pipeline Flow

```
Raw Data
   |
   v
Teacher Generation
   |
   v
[SANITIZER CHECKPOINT 1] - Flag issues, don't drop
   |
   v
Scenario Inversion
   |
   v
Anti-Attractor Generation
   |
   v
Other Augmentations
   |
   v
[SANITIZER CHECKPOINT 2] - Drop invalid entries
   |
   v
Final Clean Dataset
   |
   v
Training/Validation
```

## Report Format

### Summary Section

```json
{
  "summary": {
    "total_entries": 60,
    "clean_entries": 45,
    "duplicate_entries": 0,
    "invalid_operators": 0,
    "invalid_worlds": 0,
    "invalid_noetics": 0,
    "missing_fields": 15,
    "structural_errors": 0,
    "pass_rate": 0.75
  }
}
```

### Issues Section

```json
{
  "issues": [
    {
      "entry_id": "entry_0",
      "issue_type": "missing_field",
      "description": "Missing required field: story",
      "severity": "error",
      "field": "story"
    }
  ]
}
```

### Duplicates Section

```json
{
  "duplicates": {
    "by_id": {
      "dup_entry": 2
    },
    "by_hash": {
      "abc123...": ["entry_1", "entry_2"]
    }
  }
}
```

## Issue Types

### 1. `duplicate_id`
- **Severity**: Error
- **Description**: Multiple entries share the same ID
- **Action**: Keep first occurrence, flag subsequent ones
- **Example**: Two entries both have `id: "entry_001"`

### 2. `invalid_operator`
- **Severity**: Error
- **Description**: Operator not in ALLOWED_OPS set
- **Allowed**: `+`, `-`, `+T`, `-T`, `->`, `<-`, `*T`, `/T`, `o`
- **Example**: Entry uses `**` or `//` or any non-canonical operator

### 3. `invalid_world`
- **Severity**: Error
- **Description**: Element uses world letter not in {A, B, C, D}
- **Example**: `E5`, `F2`, `G10` (E, F, G are invalid)

### 4. `invalid_noetic`
- **Severity**: Error
- **Description**: Element uses noetic number outside 1-10 range
- **Example**: `A11`, `B0`, `C15` (0, 11, 15 are invalid)

### 5. `missing_field`
- **Severity**: Error
- **Description**: Required field is missing, null, or empty
- **Required**: `id`, `story`, `expr`, `aug_type`, `validator_pass`
- **Example**: Entry lacks `story` field

### 6. `structural_error`
- **Severity**: Error
- **Description**: Inconsistency between elements and operators
- **Rule**: N elements require N-1 operators
- **Example**: 3 elements with only 1 operator (should be 2)

## Example Workflows

### Workflow 1: Quick Data Quality Check

```bash
# Check data quality without making changes
python scripts/sanitize_augmented.py \
  --input output/teacher_augmented.jsonl \
  --flag-only

# Review console output, check pass rate
```

### Workflow 2: Clean Data for Training

```bash
# Remove all invalid entries, save clean version
python scripts/sanitize_augmented.py \
  --input output/teacher_augmented.jsonl \
  --output output/clean_data.jsonl \
  --report output/clean_report.json \
  --drop-invalid

# Review report to see what was removed
cat output/clean_report.json | jq '.summary'
```

### Workflow 3: Continuous Integration

```bash
# Add to CI/CD pipeline
python scripts/sanitize_augmented.py \
  --input output/teacher_augmented.jsonl \
  --report output/sanitizer_report.json \
  --flag-only

# Exit code 0 if clean, 1 if issues found
# CI can fail build if pass_rate < threshold
```

## Python API

### Import and Use Programmatically

```python
from pathlib import Path
from scripts.sanitize_augmented import scan_jsonl, clean_entries

# Scan file
input_path = Path("output/teacher_augmented.jsonl")
entries, report = scan_jsonl(input_path)

# Check results
print(f"Total: {report.total_entries}")
print(f"Clean: {report.clean_entries}")
print(f"Pass rate: {report.clean_entries / report.total_entries * 100:.1f}%")

# Clean entries
cleaned = clean_entries(entries, report, drop_invalid=True)

# Save report
report_dict = report.to_dict()
with open("output/report.json", "w") as f:
    json.dump(report_dict, f, indent=2)
```

### Validate Single Entry

```python
from scripts.sanitize_augmented import validate_entry

entry = {
    "id": "test_001",
    "story": "A spiritual teacher causes enlightenment",
    "expr": "A5 -> D2",
    "expr_elements": ["A5", "D2"],
    "expr_ops": ["->"],
    "aug_type": "original",
    "validator_pass": True
}

is_valid, issues = validate_entry(entry)
if not is_valid:
    for issue in issues:
        print(f"{issue.issue_type}: {issue.description}")
```

## Testing

### Run All Tests

```bash
# Run full test suite
python -m pytest tests/test_sanitize_augmented.py -v

# Run with coverage
python -m pytest tests/test_sanitize_augmented.py --cov=scripts.sanitize_augmented

# Run specific test
python -m pytest tests/test_sanitize_augmented.py::test_validate_invalid_operator -v
```

### Test Categories

1. **Validation Tests**: Individual validator functions
2. **Hash Tests**: Content hash computation
3. **Scanning Tests**: JSONL file processing
4. **Cleaning Tests**: Invalid entry removal
5. **Report Tests**: Report generation and formatting
6. **Integration Tests**: Full pipeline workflows

## Performance

- **Speed**: ~1000 entries/second on typical hardware
- **Memory**: Loads entire file into memory (consider streaming for very large files)
- **Scalability**: Tested with up to 100,000 entries

## Troubleshooting

### Issue: "Missing required field: story"

**Cause**: Original entries have empty story fields (common in generated data)

**Solution**:
- If expected: Use `--flag-only` to report but not remove
- If unexpected: Regenerate teacher stories with proper story generation

### Issue: Content hash duplicates detected

**Cause**: Augmentation generates identical stories with different IDs (e.g., W and N inversions produce same output)

**Solution**:
- This is expected behavior for some augmentation types
- Review `duplicates.by_hash` in report to identify patterns
- Consider deduplication strategy if needed

### Issue: Structural inconsistencies

**Cause**: Mismatch between `expr_elements` and `expr_ops` counts

**Solution**:
- Check expression parsing logic
- Ensure augmentation preserves structural integrity
- Verify operator extraction is correct

## Best Practices

1. **Always run sanitizer before training**: Ensures 100% canonical data
2. **Use `--flag-only` during development**: Helps debug without data loss
3. **Keep sanitizer reports**: Track data quality metrics over time
4. **Review duplicate hashes**: May indicate augmentation issues
5. **Set pass rate thresholds**: Fail CI/CD if quality drops below acceptable level
6. **Test with sample data first**: Verify configuration before full dataset

## Related Files

- **Script**: `scripts/sanitize_augmented.py`
- **Tests**: `tests/test_sanitize_augmented.py`
- **Validator**: `scripts/canonical_validator.py`
- **Constants**: `narrative/constants.py`

## Version History

- **v1.0.0** (2025-12-14): Initial release with Phase 6 implementation
  - Duplicate detection (ID and content hash)
  - Canonical validation (worlds, noetics, operators)
  - Missing field detection
  - Structural validation
  - JSON reporting
  - Comprehensive test coverage

## Support

For issues or questions:
1. Check test suite for examples: `tests/test_sanitize_augmented.py`
2. Review canonical constraints: `narrative/constants.py`
3. Consult TKS documentation: `docs/`
4. Check validator implementation: `scripts/canonical_validator.py`
