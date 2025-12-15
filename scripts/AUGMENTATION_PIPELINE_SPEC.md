# TKS DATA AUGMENTATION PIPELINE - DESIGN SPECIFICATION v1.0

## OVERVIEW

Generates augmented training data for TKS-LLM fine-tuning by applying semantic transformations to TKS narrative scenarios. Combines two complementary techniques:

1. **Scenario Inversion (InvertStory API)** - Axis-based semantic negation
2. **Anti-Attractor Synthesis (AntiAttractorInvert API)** - Pattern repulsion

This specification defines the complete augmentation pipeline from input corpus to validated output dataset with comprehensive metrics tracking.

## DESIGN PHILOSOPHY

The augmentation pipeline operates on the principle that robust TKS understanding requires exposure to diverse semantic configurations. By generating:

- **Inverted scenarios**: Teach the model canonical opposites and transformations
- **Anti-attractors**: Teach the model to recognize and avoid attractor traps

We create a training corpus that spans the full TKS semantic space while maintaining strict canonical compliance.

---

## INPUT/OUTPUT SPECIFICATION

### INPUT FORMAT

JSONL file where each line is a JSON object with:

**Required fields:**
- `"story"` (str): Natural language narrative text
  **OR**
- `"equation"` (str): TKS equation notation (e.g., "B5 -> D3 +T C8")

**Optional fields:**
- `"id"` (str): Unique identifier for source scenario
- `"metadata"` (dict): Additional metadata to preserve

**Example input line:**
```json
{
    "id": "story_001",
    "story": "A spiritual teacher causes mental growth in students",
    "metadata": {"source": "pilot_corpus", "category": "education"}
}
```

**OR equation format:**
```json
{
    "id": "eq_042",
    "equation": "A5 -> B2 +T D6",
    "metadata": {"foundation": 2, "world_focus": "B"}
}
```

### OUTPUT FORMAT

JSONL file where each line represents one scenario (original or augmented):

```json
{
    // Original fields preserved
    "story": "Inverted or original story text",
    "equation": "B3 <- D8 -T A2",  // Always populated (decoded if needed)

    // Augmentation metadata (NEW)
    "aug_type": "inversion" | "anti_attractor" | "original",
    "source_id": "story_001",  // Links to parent scenario

    // Validation metadata (NEW)
    "validator_pass": true | false,
    "validation_errors": ["error1", "error2"] | null,

    // Transformation metadata (for augmented scenarios)
    "transform_metadata": {
        // For inversion type:
        "axes": ["W", "N"],
        "mode": "soft",
        "explanation": "Inverted world and noetic dimensions",

        // For anti_attractor type:
        "signature": {
            "original_elements": {"B5": 2, "D3": 1},
            "inverted_elements": {"C6": 2, "A2": 1},
            "foundation_shift": [2, 6]
        },
        "distance": 0.85  // Semantic distance from original
    },

    // Original metadata preserved
    "metadata": {...}
}
```

**Example augmented output:**
```json
{
    "id": "story_001_inv_WN_001",
    "story": "A physical student prevents emotional stagnation in teachers",
    "equation": "D6 <- C3 -T A5",
    "aug_type": "inversion",
    "source_id": "story_001",
    "validator_pass": true,
    "validation_errors": null,
    "transform_metadata": {
        "axes": ["W", "N"],
        "mode": "soft",
        "explanation": "World inversion (A↔D, B↔C) + Noetic inversion (5↔6, 2↔3)"
    }
}
```

### METRICS OUTPUT

JSON file (same basename as output + `.metrics.json`) containing:

```json
{
    // Count statistics
    "original_count": 100,
    "inverted_count": 300,  // Multiple axes combinations per original
    "anti_attractor_count": 100,
    "total_augmented": 500,
    "validation_failures": 5,

    // Ratio metrics
    "augmentation_ratio": 4.0,  // Total augmented / original
    "inversion_ratio": 3.0,     // Inversions / original
    "anti_attractor_ratio": 1.0, // Anti-attractors / original

    // Validation metrics
    "validator_pass_rate": 0.99,      // Overall pass rate
    "world_validity": 1.0,            // % with valid worlds (A/B/C/D)
    "noetic_validity": 0.98,          // % with valid noetics (1-10)
    "foundation_validity": 1.0,       // % with valid foundations (1-7)
    "operator_validity": 0.99,        // % with valid operators
    "structural_validity": 0.97,      // % with valid structure

    // Timing
    "start_time": "2025-12-14T10:30:00",
    "end_time": "2025-12-14T10:35:42",
    "duration_seconds": 342.5,

    // Configuration snapshot
    "config": {
        "axes_combinations": [["W"], ["N"], ["W", "N"]],
        "inversion_mode": "soft",
        "use_anti_attractor": true,
        "validation_strict": true
    }
}
```

---

## PROCESSING PIPELINE

### STEP 1: CORPUS LOADING

Load input JSONL corpus and parse each line:

a) Read JSONL file line by line
b) Parse JSON object
c) Extract `story` OR `equation` field
d) Preserve `id` and `metadata` for passthrough
e) Track malformed lines separately (report but skip)

**Error handling:**
- Skip malformed JSON lines with warning
- Require either "story" or "equation" field (fail if neither)
- Generate auto-ID if missing: `f"auto_{line_number:06d}"`

### STEP 2: ENCODING (IF NEEDED)

For each entry, ensure we have a TKS expression:

a) If `"equation"` field exists: `parse_equation(equation, strict=True)`
b) If only `"story"` field exists: `EncodeStory(story, strict=True)`
c) Validate expression using canonical checks
d) Cache expression with source metadata

**Validation at encoding:**
- `strict=True` by default (use `--lenient` to disable)
- Raises `ValueError` for unknown tokens/invalid operators
- Reports detailed error messages for debugging

### STEP 3: INVERSION GENERATION

For each validated TKS expression, generate inversions:

a) For each axes combination in config (default: `[{"W"}, {"N"}, {"W","N"}]`):
   1. Call `InvertStory(expr, axes=axes, mode=mode)`
   2. Extract inverted expression and explanation
   3. Validate inverted expression (canonical checks)
   4. If valid: decode to story text using `DecodeStory`
   5. If invalid: log failure, skip this inversion
   6. Create output record with metadata

**Axes combinations (configurable via CLI):**
- Default: `{"W"}`, `{"N"}`, `{"W", "N"}` (3 inversions per original)
- Can specify: `--axes W N F E` (generates all individual + combinations)
- Supported axes: N, E, W, F, S, A, P (see AXES_MAP in scenario_inversion.py)

**Inversion modes:**
- `"soft"` (default): Standard opposite mapping (WORLD_OPP, NOETIC_OPPOSITE)
- `"hard"`: Aggressive inversion with compound transforms
- `"targeted"`: Profile-based remapping (requires `--target-profile`)

### STEP 4: ANTI-ATTRACTOR GENERATION

For each validated TKS expression (if `--use-anti-attractor`):

a) Compute attractor signature: `compute_attractor_signature(expr)`
b) Call `AntiAttractorInvert(expr)` to get counter-scenario
c) Validate counter-scenario expression (canonical checks)
d) If valid: decode to story text using `DecodeStory`
e) If invalid: log failure, skip this anti-attractor
f) Create output record with signature metadata

**Anti-attractor synthesis:**
- Inverts dominant (world, noetic) element patterns
- Inverts foundation tags (F1↔F7, F2↔F6, F3↔F5)
- Maintains structural coherence (operator distribution)
- Maximizes semantic distance from original

### STEP 5: VALIDATION

For each generated scenario (original + augmented), run canonical validation:

**Validation checks:**
1. **World validity**: All worlds in `{A, B, C, D}`
2. **Noetic validity**: All noetics in `[1..10]`
3. **Foundation validity**: All foundations in `[1..7]`
4. **Operator validity**: All operators in `ALLOWED_OPS`
5. **Structural validity**: `len(ops) == len(elements) - 1`

**Canonical constraints (from narrative/constants.py):**
- **WORLDS**: A (Spiritual), B (Mental), C (Emotional), D (Physical)
- **NOETICS**: 1-10 (Mind, Positive, Negative, Vibration, Female, Male, Rhythm, Cause, Effect, Idea)
- **FOUNDATIONS**: 1-7 (Unity, Wisdom, Life, Companionship, Power, Material, Lust)
- **ALLOWED_OPS**: `{"+T", "-T", "*T", "/T", "o", "->", "<-", "+", "-"}`

**Validation modes:**
- `strict=True` (default): Fail on any validation error
- `strict=False` (`--lenient`): Warn but include with `validator_pass=false`

### STEP 6: OUTPUT WRITING

Write all scenarios (original + augmented) to output JSONL:

a) For each scenario:
   1. Build output JSON object (see OUTPUT FORMAT above)
   2. Write as single line to output file
   3. Track counts for metrics

b) Preserve field ordering for readability:
   - `id`, `story`, `equation` first
   - `aug_type`, `source_id` next
   - `validator_pass`, `validation_errors`
   - `transform_metadata` (if augmented)
   - `metadata` (original passthrough) last

### STEP 7: METRICS COMPUTATION

After processing all scenarios, compute aggregate metrics:

a) **Count statistics:**
   - `original_count`, `inverted_count`, `anti_attractor_count`
   - `validation_failures` (`validator_pass == false`)

b) **Ratio metrics:**
   - `augmentation_ratio = (inverted + anti) / original`
   - `inversion_ratio = inverted / original`
   - `anti_attractor_ratio = anti / original`

c) **Validation metrics:**
   - `validator_pass_rate = valid / total`
   - Component-level rates (world, noetic, operator, structural)

d) **Timing:**
   - `duration_seconds = end_time - start_time`

### STEP 8: METRICS WRITING

Write metrics JSON to output file:

a) Create metrics file: `output_path.with_suffix(".metrics.json")`
b) Write formatted JSON with `indent=2` for readability
c) Include config snapshot for reproducibility

---

## DEFAULT CONFIGURATION

### AXES DEFAULTS

**Default axes combinations (if `--axes` not specified):**
```python
[{"W"}, {"N"}, {"W", "N"}]
```

This generates **3 inversions per original scenario**:
- World-only inversion (A↔D, B↔C)
- Noetic-only inversion (per NOETIC_OPPOSITE mapping)
- Combined world + noetic inversion

**Rationale:** These three combinations provide comprehensive coverage of the primary semantic dimensions while maintaining reasonable augmentation ratios.

**Custom axes (via CLI `--axes` flag):**
- `--axes W N F` → Individual inversions: `{W}`, `{N}`, `{F}`
- `--axes W N` → Individual inversions: `{W}`, `{N}`
- `--axes W N --combine` → All combinations: `{W}`, `{N}`, `{W,N}`

### MODE DEFAULTS

**Default inversion mode: `"soft"`**
- Uses standard opposite mappings (WORLD_OPP, NOETIC_OPPOSITE)
- Preserves semantic coherence
- Recommended for training data generation

**Alternative modes:**
- `"hard"`: More aggressive inversion, may produce edge cases
- `"targeted"`: Requires `--target-profile` specification

### VALIDATION DEFAULTS

**Default validation: `strict=True`**
- Enforces canonical compliance at encoding AND augmentation
- Rejects scenarios with unknown tokens/invalid operators
- Recommended for production training corpora

**Lenient mode (`--lenient`):**
- Allows unknown tokens with warnings
- Still validates canonical constraints (worlds, noetics, ops)
- Marks validation failures with `validator_pass=false`
- Useful for exploratory corpus development

### ANTI-ATTRACTOR DEFAULTS

**Default: disabled** (`--use-anti-attractor` to enable)
- Conservative default to avoid over-augmentation
- Enable when training requires counter-scenario exposure

**Number of anti-attractor elements: 3** (via `--anti-elements`)
- Controls complexity of synthesized counter-scenarios
- 3 is balanced for narrative coherence

---

## CANONICAL GUARDRAILS

### WORLD CONSTRAINTS

**ONLY `{A, B, C, D}` permitted:**
- **A**: Spiritual world
- **B**: Mental world
- **C**: Emotional world
- **D**: Physical world

Any other world letter (E, F, X, etc.) triggers validation failure.

### NOETIC CONSTRAINTS

**ONLY 1-10 permitted:**

| Noetic | Name               | Noetic | Name               |
|--------|--------------------|--------|--------------------|
| 1      | Mind/Awareness     | 6      | Male/Masculine     |
| 2      | Positive/Expansion | 7      | Rhythm/Cycle       |
| 3      | Negative/Contraction | 8    | Cause/Initiative   |
| 4      | Vibration/Frequency | 9     | Effect/Response    |
| 5      | Female/Feminine    | 10     | Idea/Form          |

Any noetic < 1 or > 10 triggers validation failure.

### FOUNDATION CONSTRAINTS

**ONLY 1-7 permitted:**

| Foundation | Name           | Foundation | Name           |
|------------|----------------|------------|----------------|
| 1          | Unity (F1)     | 5          | Power (F5)     |
| 2          | Wisdom (F2)    | 6          | Material (F6)  |
| 3          | Life (F3)      | 7          | Lust (F7)      |
| 4          | Companionship (F4) |        |                |

Any foundation < 1 or > 7 triggers validation failure.

### OPERATOR CONSTRAINTS

**ONLY operators from `ALLOWED_OPS`:**
- `+T`: TOOTRA addition/combination
- `-T`: TOOTRA subtraction/negation
- `*T`: TOOTRA multiplication/intensification
- `/T`: TOOTRA division/conflict
- `o`: Sequential composition
- `->`: Causal forward
- `<-`: Causal reverse
- `+`: Basic addition
- `-`: Basic subtraction

Any operator not in this set triggers validation failure.

### STRUCTURAL CONSTRAINTS

**Valid TKS expression structure:**
- `len(ops) == len(elements) - 1`
- Elements alternate with operators in token sequence
- Each element matches pattern: `[ABCD][1-10]`

**Example valid:** `"B5 -> D3 +T C8"` (3 elements, 2 operators)
**Example invalid:** `"B5 D3 -> C8"` (missing operator between B5 and D3)

---

## ERROR HANDLING & EDGE CASES

### MALFORMED INPUT
- **Scenario**: JSONL line is not valid JSON
- **Action**: Skip line, log warning, continue processing
- **Tracking**: Count in `metrics["malformed_lines"]`

### MISSING REQUIRED FIELDS
- **Scenario**: Line has neither "story" nor "equation" field
- **Action**: Skip line, log error, continue processing
- **Tracking**: Count in `metrics["skipped_missing_fields"]`

### ENCODING FAILURES (strict mode)
- **Scenario**: `EncodeStory` raises `ValueError` for unknown tokens
- **Action**: Skip scenario, log detailed error, continue processing
- **Tracking**: Count in `metrics["encoding_failures"]`
- **Details**: Include first 5 unknown tokens in error message

### INVERSION FAILURES
- **Scenario**: `InvertStory` produces invalid TKS expression
- **Action**: Skip this inversion, log validation errors, continue with other axes
- **Tracking**: Count in `metrics["inversion_failures"]`
- **Granularity**: Per-axis tracking (e.g., "W_failures", "N_failures")

### ANTI-ATTRACTOR FAILURES
- **Scenario**: `AntiAttractorInvert` produces invalid TKS expression
- **Action**: Skip anti-attractor, log errors, continue processing
- **Tracking**: Count in `metrics["anti_attractor_failures"]`

### VALIDATION FAILURES (lenient mode)
- **Scenario**: Generated scenario fails canonical validation
- **Action**: Include in output with `validator_pass=false`
- **Tracking**: Component-level failure tracking (world, noetic, operator, structural)
- **Usage**: Can filter out during training data preparation

### ZERO AUGMENTATIONS
- **Scenario**: All augmentation attempts fail for a scenario
- **Action**: Include only original scenario in output
- **Tracking**: Count in `metrics["zero_augmentation_count"]`
- **Warning**: Report if ratio exceeds threshold (e.g., >10% of corpus)

---

## DATA QUALITY & SANITIZATION (Phase 4)

### OVERVIEW

After augmentation, an optional sanitization step ensures data quality by detecting and handling:
- Duplicate entries (by ID and content hash)
- Invalid operators, worlds, or noetics
- Missing required fields
- Structural inconsistencies

The sanitizer (`scripts/sanitize_augmented.py`) provides flexible options for quality control.

### SANITIZATION FEATURES

**Detection capabilities:**
1. **Duplicate detection**
   - By ID: Identifies entries with identical IDs
   - By content hash: Identifies entries with identical content but different IDs
2. **Canonical validation**
   - Invalid operators (not in ALLOWED_OPS)
   - Invalid worlds (not in A/B/C/D)
   - Invalid noetics (not in 1-10)
   - Invalid foundations (not in 1-7)
3. **Structural validation**
   - Missing required fields (id, story, expr, aug_type, validator_pass)
   - Operator/element count mismatches
4. **Detailed reporting**
   - Per-issue tracking with severity levels
   - Summary statistics and pass rates
   - JSON report export for analysis

**Operation modes:**
- `--flag-only`: Report issues without removing entries
- `--drop-invalid`: Remove invalid entries from output
- `--report FILE`: Save detailed JSON report

### SANITIZER USAGE

#### Basic Scan (Report Only)

```bash
python scripts/sanitize_augmented.py \
    --input data/pilot/augmented.jsonl \
    --flag-only
```

**Output:** Console report showing all detected issues.

#### Clean and Save

```bash
python scripts/sanitize_augmented.py \
    --input data/pilot/augmented.jsonl \
    --output data/pilot/augmented_clean.jsonl \
    --drop-invalid
```

**Output:**
- Clean JSONL file with invalid entries removed
- Console report showing what was removed

#### Detailed Analysis

```bash
python scripts/sanitize_augmented.py \
    --input data/pilot/augmented.jsonl \
    --output data/pilot/augmented_clean.jsonl \
    --drop-invalid \
    --report data/pilot/sanitization_report.json
```

**Output:**
- Clean JSONL file
- Console summary
- Detailed JSON report with per-entry issue tracking

### REPORT FORMAT

The sanitization report includes:

```json
{
  "summary": {
    "total_entries": 500,
    "clean_entries": 485,
    "duplicate_entries": 5,
    "invalid_operators": 3,
    "invalid_worlds": 2,
    "invalid_noetics": 4,
    "missing_fields": 1,
    "structural_errors": 0,
    "pass_rate": 0.97
  },
  "issues": [
    {
      "entry_id": "entry_042",
      "issue_type": "invalid_operator",
      "description": "Invalid operator '**' at position 0 (must be in {'+', '-', '+T', '-T', '->', '<-', '*T', '/T', 'o'})",
      "severity": "error",
      "field": "expr_ops"
    }
  ],
  "duplicates": {
    "by_id": {
      "entry_001": 2
    },
    "by_hash": {
      "a3f5b8c...": ["entry_010", "entry_011"]
    }
  }
}
```

### INTEGRATION INTO PIPELINE

The sanitizer can be used as an optional post-processing step:

```bash
# Step 1: Generate augmented data
python scripts/generate_augmented_data.py \
    --input data/pilot/stories.jsonl \
    --output data/pilot/augmented.jsonl \
    --axes W N \
    --use-anti-attractor

# Step 2: Sanitize and clean
python scripts/sanitize_augmented.py \
    --input data/pilot/augmented.jsonl \
    --output data/pilot/augmented_clean.jsonl \
    --drop-invalid \
    --report data/pilot/quality_report.json

# Step 3: Use cleaned data for training
python scripts/train_with_augmented.py \
    --data data/pilot/augmented_clean.jsonl \
    --model-name tks-pilot
```

### QUALITY METRICS

The sanitizer provides comprehensive quality metrics:

- **Pass rate**: Percentage of entries passing all validations
- **Duplicate rate**: Percentage of entries with duplicate IDs or content
- **Validation breakdown**: Per-validator pass rates (world, noetic, operator, structure)
- **Issue distribution**: Count of each issue type

**Recommended thresholds for production:**
- Pass rate: ≥ 95%
- Duplicate rate: ≤ 1%
- Missing fields: 0%

### CANONICAL ENFORCEMENT

The sanitizer enforces the same canonical constraints as the augmentation pipeline:

- **Worlds**: Only A, B, C, D
- **Noetics**: Only 1-10 (with involution pairs: 2↔3, 5↔6, 8↔9; self-duals: 1, 4, 7, 10)
- **Foundations**: Only 1-7
- **Operators**: Only {+, -, +T, -T, ->, <-, *T, /T, o}
- **Structure**: len(ops) = len(elements) - 1

---

## USAGE EXAMPLES

### BASIC USAGE (Default Settings)

```bash
python scripts/generate_augmented_data.py \
    --input data/pilot/stories.jsonl \
    --output data/pilot/augmented.jsonl
```

**Generates:**
- 3 inversions per scenario (W, N, W+N)
- No anti-attractors (disabled by default)
- Strict validation
- Metrics saved to `data/pilot/augmented.metrics.json`

### FULL AUGMENTATION (All Techniques)

```bash
python scripts/generate_augmented_data.py \
    --input data/pilot/stories.jsonl \
    --output data/pilot/augmented_full.jsonl \
    --axes W N F \
    --use-anti-attractor \
    --validate
```

**Generates:**
- 3 inversions per scenario (W, N, F individually)
- 1 anti-attractor per scenario
- 4x augmentation ratio (4 augmented per original)
- Strict validation with detailed pass/fail tracking

### CUSTOM AXES COMBINATIONS

```bash
python scripts/generate_augmented_data.py \
    --input data/pilot/stories.jsonl \
    --output data/pilot/augmented_custom.jsonl \
    --axes W N E F
```

**Generates individual inversions for each axis:**
- `{W}`: World inversion only
- `{N}`: Noetic inversion only
- `{E}`: Element inversion only
- `{F}`: Foundation inversion only
- 4 inversions per scenario

### LENIENT MODE (Exploratory)

```bash
python scripts/generate_augmented_data.py \
    --input data/experimental/raw_stories.jsonl \
    --output data/experimental/augmented.jsonl \
    --lenient \
    --use-anti-attractor
```

**Generates:**
- Allows unknown tokens with warnings
- Still validates canonical constraints
- Marks validation failures (don't exclude)
- Useful for corpus development iteration

### PRODUCTION MODE (High Quality)

```bash
python scripts/generate_augmented_data.py \
    --input data/training/validated_corpus.jsonl \
    --output data/training/final_augmented.jsonl \
    --axes W N \
    --mode soft \
    --min-pass-rate 0.95 \
    --save-metrics \
    --verbose
```

**Generates:**
- Strict validation (reject invalid scenarios)
- Minimum 95% pass rate required
- Detailed metrics and progress reporting
- Suitable for final training data preparation

---

## IMPLEMENTATION NOTES

### PHASE 1 (Current): Scaffolding & Specification

**Status:** Complete

**Deliverables:**
- Data class definitions (`AugmentationConfig`, `AugmentationMetrics`)
- Function signatures with detailed docstrings
- CLI argument parsing
- This comprehensive design specification

### PHASE 2 (Next): Core Implementation

**Tasks:**
1. Implement `load_corpus()` with JSONL parsing
2. Implement `generate_inverted_scenarios()` using InvertStory API
3. Implement `generate_anti_attractor_pairs()` using AntiAttractorInvert API
4. Implement `validate_canonical()` with component-level checks
5. Implement `compute_validator_pass_rate()` and `compute_augmentation_ratio()`
6. Implement `save_augmented_corpus()` and `save_metrics()`
7. Implement `augment_corpus()` pipeline orchestration

### PHASE 3 (Future): Optimization & Scaling

**Enhancements:**
- Parallel processing for large corpora (multiprocessing)
- Progress bars (tqdm integration)
- Checkpoint/resume capability
- Deduplication by signature hash
- Quality scoring for ranking augmentations

---

## DEPENDENCIES

### Core TKS modules
- `scenario_inversion.py`: InvertStory, EncodeStory, DecodeStory
- `anti_attractor.py`: AntiAttractorInvert, compute_attractor_signature
- `narrative.constants`: ALLOWED_OPS, validation helpers

### Standard library
- `argparse`: CLI argument parsing
- `json`: JSONL reading/writing
- `pathlib`: Path handling
- `dataclasses`: Configuration and metrics structures
- `datetime`: Timestamp tracking

---

## AUTHOR & VERSION

- **Author**: TKS-LLM Training Integration Team
- **Date**: 2025-12-14
- **Version**: 1.0.0 (Phase 1 Complete - Design Specification)
- **Status**: SPECIFICATION COMPLETE - Ready for Phase 2 Implementation
