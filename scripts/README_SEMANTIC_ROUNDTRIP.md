# Semantic Round-trip Fidelity Metrics

**Quick Reference Guide**

## What It Does

Measures semantic drift in story→equation→story round-trips using:
- **Semantic similarity** (embedding-based, 0-1 scale)
- **Element overlap** (Jaccard similarity for TKS elements, 0-1 scale)
- **Fidelity score** (combined: 60% semantic + 40% element)

## Quick Start

```bash
# Basic usage with default test stories
python3 scripts/semantic_roundtrip_metrics.py --model output/phase5_models/final_model.pt

# Custom stories
python3 scripts/semantic_roundtrip_metrics.py --model <model> --stories test_stories.txt

# Save results
python3 scripts/semantic_roundtrip_metrics.py --model <model> --output results.json
```

## Installation

Required (already installed):
- PyTorch
- NumPy

Optional (for better semantic similarity):
```bash
pip install sentence-transformers
```

## Files Created

| File | Purpose |
|------|---------|
| `scripts/semantic_roundtrip_metrics.py` | Main validation script |
| `tests/test_semantic_roundtrip.py` | Unit tests |
| `examples/demo_semantic_roundtrip.py` | Interactive demo |
| `docs/SEMANTIC_ROUNDTRIP_GUIDE.md` | Comprehensive guide |

## Default Test Stories

1. "She felt fear and anxiety."
2. "A man thought about power."
3. "Love caused joy in the heart."
4. "The mind vibrates with ideas."
5. "Effect follows cause in rhythm."

## Key Functions

```python
from scripts.semantic_roundtrip_metrics import (
    compute_semantic_similarity,     # Compare stories (0-1)
    compute_element_overlap,         # Compare equations (0-1)
    compute_roundtrip_fidelity,      # Full round-trip
    run_roundtrip_validation,        # Batch validation
)
```

## Interpreting Results

### Fidelity Score
- **≥0.8**: Excellent - High consistency maintained
- **0.6-0.8**: Good - Acceptable for most uses
- **0.4-0.6**: Moderate - May need improvement
- **<0.4**: Poor - Significant semantic drift

### Element Overlap (Jaccard)
- **1.0**: Perfect - Identical element sets
- **0.67**: Good - 2/3 overlap (e.g., A1,B5,C3 vs A1,B5)
- **0.0**: None - No common elements

## Examples

### Run with verbose output
```bash
python3 scripts/semantic_roundtrip_metrics.py \
    --model output/phase5_models/final_model.pt \
    --verbose
```

### Compare multiple models
```bash
for model in output/*/final_model.pt; do
    python3 scripts/semantic_roundtrip_metrics.py \
        --model "$model" \
        --output "${model%.pt}_fidelity.json"
done
```

### Lower temperature for consistency
```bash
python3 scripts/semantic_roundtrip_metrics.py \
    --model output/phase5_models/final_model.pt \
    --temperature 0.5
```

## Testing

```bash
# Run unit tests
python3 tests/test_semantic_roundtrip.py

# Run demo
python3 examples/demo_semantic_roundtrip.py
```

## TKS Element Support

Recognizes patterns like:
- Base: `A1`, `B5`, `C3`, `D10`
- Extended: `A1_d2`, `B5^r3`, `C3_d1^r2`
- Case-insensitive: `a1` → `A1`

## Output Format

Console:
```
======================================================================
ROUND-TRIP VALIDATION - 5 Stories
======================================================================

Story 1/5: She felt fear and anxiety....
  → Fidelity: 0.847 (semantic: 0.912, element: 0.750)
  → Elements: 3 extracted, 2 regenerated

======================================================================
AGGREGATE RESULTS
======================================================================

Fidelity Score:
  Mean: 0.823 ± 0.045
  Range: [0.765, 0.891]
```

JSON (with `--output`):
```json
{
  "metadata": {...},
  "results": [...],
  "aggregate": {
    "mean_fidelity": 0.823,
    "mean_semantic_similarity": 0.884,
    "mean_element_overlap": 0.726
  }
}
```

## CLI Options

```
--model MODEL           Model checkpoint path (required)
--stories STORIES       Test stories file (optional)
--temperature FLOAT     Sampling temperature (default: 0.7)
--output FILE           Save results to JSON
--verbose               Detailed output
--device {cpu,cuda}     Force device
```

## Documentation

Full guide: `/mnt/c/Users/wakil/Downloads/Everthing-Tootra-TKS/docs/SEMANTIC_ROUNDTRIP_GUIDE.md`

## Quick Troubleshooting

**Low fidelity?**
- Train model more
- Use lower temperature (0.5)
- Check test story distribution

**No elements extracted?**
- Model not trained on TKS notation
- Stories don't contain elements
- Check equation prompts

**Import errors?**
- Script works without sentence-transformers (uses fallback)
- Install with: `pip install sentence-transformers`

## Related Scripts

- `scripts/test_roundtrip.py` - Original round-trip testing
- `scripts/quick_train.py` - Model training
- `scripts/evaluate_pilot.py` - Pilot evaluation

---

**Author**: TKS-LLM Validation Agent
**Date**: 2025-12-22
**Version**: 1.0
