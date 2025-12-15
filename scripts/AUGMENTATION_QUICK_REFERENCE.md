# TKS Augmentation Pipeline - Quick Reference

## At a Glance

**Purpose**: Generate augmented training data for TKS-LLM fine-tuning
**Input**: JSONL file with `story` or `equation` fields
**Output**: JSONL file with original + augmented scenarios + metrics JSON

---

## Input/Output Formats

### Input (JSONL)
```json
{"id": "story_001", "story": "A spiritual teacher causes mental growth", "metadata": {...}}
```
OR
```json
{"id": "eq_001", "equation": "A5 -> B2 +T D6", "metadata": {...}}
```

### Output (JSONL)
```json
{
  "id": "story_001_inv_WN_001",
  "story": "A physical student prevents emotional stagnation",
  "equation": "D6 <- C3 -T A5",
  "aug_type": "inversion",
  "source_id": "story_001",
  "validator_pass": true,
  "validation_errors": null,
  "transform_metadata": {
    "axes": ["W", "N"],
    "mode": "soft",
    "explanation": "World + Noetic inversion"
  }
}
```

---

## Processing Steps

1. **Load** JSONL corpus, parse each line
2. **Encode** stories → TKS expressions (or parse equations)
3. **Invert** across specified axes (default: W, N, W+N)
4. **Generate** anti-attractors (if `--use-anti-attractor`)
5. **Validate** all scenarios against canonical constraints
6. **Write** output JSONL with metadata
7. **Compute** and save metrics JSON

---

## Default Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| **Axes** | `{W}`, `{N}`, `{W,N}` | 3 inversions per scenario |
| **Mode** | `soft` | Standard opposite mapping |
| **Validation** | `strict=True` | Reject invalid scenarios |
| **Anti-attractor** | Disabled | Enable with `--use-anti-attractor` |

---

## Canonical Constraints

| Constraint | Valid Values | Examples |
|------------|--------------|----------|
| **Worlds** | A, B, C, D only | A=Spiritual, B=Mental, C=Emotional, D=Physical |
| **Noetics** | 1-10 only | 1=Mind, 2=Positive, 3=Negative, 5=Female, 6=Male |
| **Foundations** | 1-7 only | 1=Unity, 2=Wisdom, 3=Life, 5=Power, 7=Lust |
| **Operators** | `+T`, `-T`, `*T`, `/T`, `o`, `->`, `<-`, `+`, `-` | From ALLOWED_OPS |
| **Structure** | `len(ops) == len(elements) - 1` | "B5 -> D3 +T C8" ✓ |

---

## Common Usage Patterns

### Basic (Default Settings)
```bash
python scripts/generate_augmented_data.py \
    --input data/pilot/stories.jsonl \
    --output data/pilot/augmented.jsonl
```
- 3 inversions per scenario (W, N, W+N)
- No anti-attractors
- Strict validation
- Auto-saves metrics

### Full Augmentation
```bash
python scripts/generate_augmented_data.py \
    --input data/pilot/stories.jsonl \
    --output data/pilot/augmented.jsonl \
    --axes W N F \
    --use-anti-attractor
```
- 3 inversions + 1 anti-attractor = 4x ratio
- Strict validation
- Comprehensive coverage

### Lenient (Exploratory)
```bash
python scripts/generate_augmented_data.py \
    --input data/experimental/stories.jsonl \
    --output data/experimental/augmented.jsonl \
    --lenient
```
- Allows unknown tokens (with warnings)
- Marks validation failures
- Useful for corpus development

---

## Metadata Fields

### `aug_type`
- `"original"`: Source scenario (unchanged)
- `"inversion"`: Axis-based inverted scenario
- `"anti_attractor"`: Counter-scenario via pattern repulsion

### `validator_pass`
- `true`: Passes all canonical checks
- `false`: Fails one or more checks (see `validation_errors`)

### `transform_metadata` (for augmented scenarios)
**Inversion:**
```json
{
  "axes": ["W", "N"],
  "mode": "soft",
  "explanation": "World inversion (A↔D, B↔C) + Noetic inversion"
}
```

**Anti-attractor:**
```json
{
  "signature": {
    "original_elements": {"B5": 2, "D3": 1},
    "inverted_elements": {"C6": 2, "A2": 1}
  },
  "distance": 0.85
}
```

---

## Metrics Output

Saved to `<output>.metrics.json`:

```json
{
  "original_count": 100,
  "inverted_count": 300,
  "anti_attractor_count": 100,
  "augmentation_ratio": 4.0,
  "validator_pass_rate": 0.99,
  "world_validity": 1.0,
  "noetic_validity": 0.98,
  "operator_validity": 0.99,
  "duration_seconds": 342.5
}
```

---

## Error Handling

| Error Type | Action | Tracking |
|------------|--------|----------|
| Malformed JSON | Skip line, warn | `metrics["malformed_lines"]` |
| Missing fields | Skip line, error | `metrics["skipped_missing_fields"]` |
| Encoding failure | Skip scenario, log | `metrics["encoding_failures"]` |
| Inversion failure | Skip inversion, continue | `metrics["inversion_failures"]` |
| Anti-attractor failure | Skip anti, continue | `metrics["anti_attractor_failures"]` |
| Validation failure (lenient) | Include with `validator_pass=false` | Component-level tracking |

---

## CLI Flags Reference

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--input` | path | required | Input JSONL corpus |
| `--output` | path | required | Output JSONL file |
| `--axes` | N E W F S A P | W N | Axes for inversion |
| `--mode` | soft/hard/targeted | soft | Inversion mode |
| `--use-anti-attractor` | flag | false | Enable anti-attractors |
| `--anti-elements` | int | 3 | Elements in anti-attractor |
| `--validate` | flag | true | Run validation |
| `--lenient` | flag | false | Allow unknown tokens |
| `--min-pass-rate` | float | 0.90 | Min validation pass rate |
| `--save-metrics` | flag | true | Save metrics JSON |
| `--verbose` | flag | true | Verbose output |

---

## See Also

- **Full Design Spec**: `scripts/AUGMENTATION_PIPELINE_SPEC.md`
- **Implementation**: `scripts/generate_augmented_data.py`
- **Scenario Inversion**: `scenario_inversion.py`
- **Anti-Attractor**: `anti_attractor.py`
- **Narrative Encoder**: `narrative/encoder.py`
- **Constants**: `narrative/constants.py`
