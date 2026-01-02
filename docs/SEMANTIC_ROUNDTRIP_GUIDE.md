# Semantic Round-trip Fidelity Metrics - User Guide

## Overview

The semantic round-trip metrics system provides comprehensive validation for TKS story↔equation transformations. It measures how well the model maintains semantic consistency through complete round-trip cycles.

**Location**: `/mnt/c/Users/wakil/Downloads/Everthing-Tootra-TKS/scripts/semantic_roundtrip_metrics.py`

## Key Features

1. **Embedding-based Semantic Similarity**
   - Uses sentence-transformers for deep semantic comparison
   - Captures meaning beyond surface-level text matching
   - Range: [0, 1] where 1.0 = perfect semantic match

2. **TKS Element Overlap Analysis**
   - Extracts TKS elements from equations and stories
   - Computes Jaccard similarity for structural consistency
   - Supports extended notation (e.g., A1_d2^r3)

3. **Combined Fidelity Scoring**
   - Weighted metric: 60% semantic + 40% element overlap
   - Balances meaning preservation with structural accuracy
   - Aggregate statistics across multiple test cases

4. **Comprehensive Reporting**
   - Individual round-trip results
   - Aggregate metrics with mean/std/min/max
   - Element extraction statistics
   - JSON export for further analysis

## Installation

### Core Dependencies (Required)

The script works with existing dependencies:
- PyTorch
- NumPy

### Optional: Enhanced Semantic Similarity

For better semantic similarity (recommended but not required):

```bash
pip install sentence-transformers
```

**Fallback**: If sentence-transformers is not installed, the script automatically falls back to character-level Jaccard similarity.

## Usage

### Basic Usage - Default Test Stories

Run validation with the 5 default test stories:

```bash
python3 scripts/semantic_roundtrip_metrics.py \
    --model output/phase5_models/final_model.pt
```

Default stories:
1. "She felt fear and anxiety."
2. "A man thought about power."
3. "Love caused joy in the heart."
4. "The mind vibrates with ideas."
5. "Effect follows cause in rhythm."

### Custom Test Stories

#### From Text File (one story per line)

```bash
python3 scripts/semantic_roundtrip_metrics.py \
    --model output/phase5_models/final_model.pt \
    --stories my_test_stories.txt
```

Example `my_test_stories.txt`:
```
She felt fear and anxiety.
A man thought about power.
Love caused joy in the heart.
```

#### From JSON File

```bash
python3 scripts/semantic_roundtrip_metrics.py \
    --model output/phase5_models/final_model.pt \
    --stories stories.json
```

Example `stories.json`:
```json
[
  "She felt fear and anxiety.",
  "A man thought about power.",
  "Love caused joy in the heart."
]
```

#### From JSONL File

```bash
python3 scripts/semantic_roundtrip_metrics.py \
    --model output/phase5_models/final_model.pt \
    --stories output/teacher_augmented.jsonl
```

The script extracts the `story` field from each JSON object.

### Advanced Options

#### Adjust Generation Temperature

```bash
# Lower temperature = more deterministic (recommended for validation)
python3 scripts/semantic_roundtrip_metrics.py \
    --model output/phase5_models/final_model.pt \
    --temperature 0.5

# Higher temperature = more creative
python3 scripts/semantic_roundtrip_metrics.py \
    --model output/phase5_models/final_model.pt \
    --temperature 1.0
```

#### Save Results to JSON

```bash
python3 scripts/semantic_roundtrip_metrics.py \
    --model output/phase5_models/final_model.pt \
    --output output/roundtrip_results.json
```

#### Verbose Output

```bash
python3 scripts/semantic_roundtrip_metrics.py \
    --model output/phase5_models/final_model.pt \
    --verbose
```

Shows intermediate steps:
- Story → Equation generation
- Equation → Story regeneration
- Detailed metrics for each step

#### Specify Device

```bash
# Force CPU
python3 scripts/semantic_roundtrip_metrics.py \
    --model output/phase5_models/final_model.pt \
    --device cpu

# Force CUDA
python3 scripts/semantic_roundtrip_metrics.py \
    --model output/phase5_models/final_model.pt \
    --device cuda
```

## Output Format

### Console Output

```
======================================================================
TKS SEMANTIC ROUND-TRIP FIDELITY VALIDATION
======================================================================

Device: cuda
Temperature: 0.7

Loaded sentence-transformers model: all-MiniLM-L6-v2
Loading model from: output/phase5_models/final_model.pt
  Vocab size: 256
  Device: cuda

Using default test stories (5 stories)

======================================================================
ROUND-TRIP VALIDATION - 5 Stories
======================================================================


Story 1/5: She felt fear and anxiety....
  → Fidelity: 0.847 (semantic: 0.912, element: 0.750)
  → Elements: 3 extracted, 2 regenerated

Story 2/5: A man thought about power....
  → Fidelity: 0.791 (semantic: 0.856, element: 0.667)
  → Elements: 2 extracted, 2 regenerated

[... more stories ...]

======================================================================
AGGREGATE RESULTS
======================================================================

Fidelity Score:
  Mean: 0.823 ± 0.045
  Range: [0.765, 0.891]

Semantic Similarity:
  Mean: 0.884 ± 0.038

Element Overlap:
  Mean: 0.726 ± 0.062

Element Statistics:
  Total extracted: 15
  Total regenerated: 13
  Unique extracted: 8
  Unique regenerated: 7

  Most common elements:
    A1: 4
    B5: 3
    C3: 2
    D10: 2
    A1_D2: 1

======================================================================
VALIDATION COMPLETE
======================================================================

Overall Fidelity: 0.823
Stories Processed: 5
Elements Extracted: 15

✓ Excellent fidelity (≥0.8)
```

### JSON Output

When using `--output results.json`:

```json
{
  "metadata": {
    "model_checkpoint": "output/phase5_models/final_model.pt",
    "num_stories": 5,
    "temperature": 0.7,
    "device": "cuda"
  },
  "results": [
    {
      "original_story": "She felt fear and anxiety.",
      "extracted_equation": "A1_d2 ⊕ B5 → C10",
      "regenerated_story": "She experienced fear and worry.",
      "semantic_similarity": 0.912,
      "element_overlap": 0.750,
      "fidelity_score": 0.847,
      "extracted_elements": ["A1_D2", "B5", "C10"],
      "regenerated_elements": ["A1_D2", "B5"]
    }
  ],
  "aggregate": {
    "mean_fidelity": 0.823,
    "std_fidelity": 0.045,
    "min_fidelity": 0.765,
    "max_fidelity": 0.891,
    "mean_semantic_similarity": 0.884,
    "std_semantic_similarity": 0.038,
    "mean_element_overlap": 0.726,
    "std_element_overlap": 0.062,
    "num_stories": 5,
    "element_statistics": {
      "total_extracted": 15,
      "total_regenerated": 13,
      "unique_extracted": 8,
      "unique_regenerated": 7,
      "most_common_elements": [
        ["A1", 4],
        ["B5", 3]
      ]
    }
  }
}
```

## Metrics Explanation

### Semantic Similarity (0-1)

Measures how well the regenerated story captures the meaning of the original story.

- **1.0**: Perfect semantic match (identical meaning)
- **0.8-1.0**: Excellent (minor paraphrasing)
- **0.6-0.8**: Good (same general meaning, different wording)
- **0.4-0.6**: Moderate (partial meaning preserved)
- **0.0-0.4**: Poor (different meaning)

**Method**: Cosine similarity between sentence embeddings

### Element Overlap (0-1)

Measures consistency of TKS elements through the round-trip.

- **1.0**: Identical element sets
- **0.8-1.0**: Excellent (1-2 elements difference)
- **0.6-0.8**: Good (moderate overlap)
- **0.4-0.6**: Fair (partial overlap)
- **0.0-0.4**: Poor (little/no overlap)

**Method**: Jaccard similarity (intersection / union)

### Fidelity Score (0-1)

Combined metric balancing semantic and structural consistency.

**Formula**: `fidelity = 0.6 × semantic_similarity + 0.4 × element_overlap`

**Interpretation**:
- **≥0.8**: Excellent - Model maintains high fidelity
- **0.6-0.8**: Good - Acceptable for most use cases
- **0.4-0.6**: Moderate - May need improvement
- **<0.4**: Poor - Significant drift

**Weights Rationale**:
- 60% semantic: Meaning preservation is primary goal
- 40% element: Structural consistency is important but secondary

## Understanding Element Extraction

### Supported Patterns

The script recognizes TKS elements in various formats:

```python
# Base elements (World + Noetic)
A1, A2, ..., A10    # Aziluth world
B1, B2, ..., B10    # Beriah world
C1, C2, ..., C10    # Yetzirah world
D1, D2, ..., D10    # Assiah world

# Extended notation (with modifiers)
A1_d2      # Desire modifier
A1^r3      # Resonance modifier
A1_d2^r3   # Combined modifiers

# Case insensitive
a1, A1, a1_d2, A1_D2  # All normalized to A1, A1_D2
```

### Example Extractions

```python
extract_tks_elements("Using A1 and B5 gives C10")
# → ['A1', 'B5', 'C10']

extract_tks_elements("Complex: A1_d2 ⊕ B5^r3 → C3_d1^r2")
# → ['A1_D2', 'B5^R3', 'C3_D1^R2']

extract_tks_elements("Story: She felt (A1) fear leading to (B5) wisdom")
# → ['A1', 'B5']
```

## Integration with Existing Workflows

### Use in Training Pipeline

```bash
# After training, validate model fidelity
python3 scripts/quick_train.py \
    --data output/teacher_augmented.jsonl \
    --output output/new_model \
    --epochs 10

# Run round-trip validation
python3 scripts/semantic_roundtrip_metrics.py \
    --model output/new_model/final_model.pt \
    --stories output/teacher_augmented.jsonl \
    --output output/new_model/fidelity_results.json
```

### Use in CI/CD

```yaml
# .github/workflows/validation.yml
- name: Run Semantic Round-trip Validation
  run: |
    python3 scripts/semantic_roundtrip_metrics.py \
      --model output/phase5_models/final_model.pt \
      --output validation_results.json

    # Check fidelity threshold
    python3 -c "
    import json, sys
    with open('validation_results.json') as f:
        result = json.load(f)
    fidelity = result['aggregate']['mean_fidelity']
    if fidelity < 0.6:
        print(f'Fidelity {fidelity:.3f} below threshold 0.6')
        sys.exit(1)
    "
```

### Use with Different Models

The script works with any model using the `SimpleTransformer` architecture:

```bash
# Test different checkpoints
for model in output/*/final_model.pt; do
    echo "Testing $model"
    python3 scripts/semantic_roundtrip_metrics.py \
        --model "$model" \
        --output "${model%.pt}_fidelity.json"
done
```

## API Reference

### Core Functions

#### `compute_semantic_similarity(story1: str, story2: str) -> float`

Compare two stories using embedding similarity.

**Returns**: Similarity score [0, 1]

```python
from scripts.semantic_roundtrip_metrics import compute_semantic_similarity

sim = compute_semantic_similarity(
    "She felt fear and anxiety.",
    "She experienced fear and worry."
)
print(f"Similarity: {sim:.3f}")  # → Similarity: 0.912
```

#### `compute_element_overlap(eq1: str, eq2: str) -> float`

Compare TKS elements between equations using Jaccard similarity.

**Returns**: Jaccard similarity [0, 1]

```python
from scripts.semantic_roundtrip_metrics import compute_element_overlap

overlap = compute_element_overlap(
    "A1_d2 ⊕ B5",
    "A1_D2 and B5"
)
print(f"Overlap: {overlap:.3f}")  # → Overlap: 1.000
```

#### `compute_roundtrip_fidelity(...) -> dict`

Perform full round-trip validation.

**Returns**: Dictionary with results

```python
from scripts.semantic_roundtrip_metrics import (
    compute_roundtrip_fidelity,
    load_model
)
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, tokenizer = load_model('output/phase5_models/final_model.pt', device)

result = compute_roundtrip_fidelity(
    "She felt fear and anxiety.",
    model,
    tokenizer,
    verbose=True
)

print(f"Fidelity: {result['fidelity_score']:.3f}")
```

#### `run_roundtrip_validation(...) -> dict`

Run validation on multiple stories.

**Returns**: Dictionary with individual results and aggregates

```python
from scripts.semantic_roundtrip_metrics import (
    run_roundtrip_validation,
    load_model,
    DEFAULT_TEST_STORIES
)
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, tokenizer = load_model('output/phase5_models/final_model.pt', device)

results = run_roundtrip_validation(
    DEFAULT_TEST_STORIES,
    model,
    tokenizer
)

print(f"Mean fidelity: {results['aggregate']['mean_fidelity']:.3f}")
```

## Troubleshooting

### Issue: ImportError for sentence-transformers

**Solution**: Install the package or use fallback mode

```bash
# Option 1: Install sentence-transformers
pip install sentence-transformers

# Option 2: Script automatically uses fallback
# (character-level Jaccard similarity)
```

### Issue: Low fidelity scores

**Possible causes**:
1. Model not trained enough → Train for more epochs
2. Temperature too high → Use lower temperature (0.5-0.7)
3. Test stories too complex → Start with simpler stories
4. Data mismatch → Ensure test stories match training distribution

### Issue: No elements extracted

**Possible causes**:
1. Model not generating TKS notation → Check training data
2. Stories don't contain elements → Use equation-focused prompts
3. Extended notation not recognized → Check regex pattern

### Issue: CUDA out of memory

**Solution**: Use CPU or smaller batch

```bash
# Force CPU
python3 scripts/semantic_roundtrip_metrics.py \
    --model output/phase5_models/final_model.pt \
    --device cpu
```

## Best Practices

1. **Start with Default Stories**: Validate basic functionality first
2. **Use Lower Temperature**: 0.5-0.7 for more consistent results
3. **Save Results**: Always use `--output` for reproducibility
4. **Track Over Time**: Compare fidelity across training checkpoints
5. **Domain-Specific Stories**: Create test sets matching your use case
6. **Aggregate Analysis**: Don't rely on single examples

## Examples

### Example 1: Quick Validation

```bash
python3 scripts/semantic_roundtrip_metrics.py \
    --model output/phase5_models/final_model.pt
```

### Example 2: Comprehensive Analysis

```bash
python3 scripts/semantic_roundtrip_metrics.py \
    --model output/phase5_models/final_model.pt \
    --stories output/teacher_augmented.jsonl \
    --temperature 0.6 \
    --output output/comprehensive_fidelity.json \
    --verbose
```

### Example 3: Compare Models

```bash
# Create comparison script
cat > compare_models.sh << 'EOF'
#!/bin/bash
for model in output/*/final_model.pt; do
    name=$(basename $(dirname $model))
    python3 scripts/semantic_roundtrip_metrics.py \
        --model "$model" \
        --output "output/fidelity_${name}.json"
done

# Aggregate results
python3 -c "
import json, glob
results = []
for f in glob.glob('output/fidelity_*.json'):
    with open(f) as fp:
        data = json.load(fp)
        results.append({
            'model': data['metadata']['model_checkpoint'],
            'fidelity': data['aggregate']['mean_fidelity']
        })
results.sort(key=lambda x: x['fidelity'], reverse=True)
print('Model Comparison:')
for r in results:
    print(f\"{r['model']}: {r['fidelity']:.3f}\")
"
EOF

chmod +x compare_models.sh
./compare_models.sh
```

## Related Documentation

- `/mnt/c/Users/wakil/Downloads/Everthing-Tootra-TKS/scripts/test_roundtrip.py` - Original round-trip testing
- `/mnt/c/Users/wakil/Downloads/Everthing-Tootra-TKS/docs/TRAINING_INTEGRATION_PLAN.md` - Training pipeline
- `/mnt/c/Users/wakil/Downloads/Everthing-Tootra-TKS/tests/test_semantic_roundtrip.py` - Unit tests

## Citation

If you use this validation framework in your work:

```bibtex
@misc{tks_semantic_roundtrip,
  title={Semantic Round-trip Fidelity Metrics for TKS Story-Equation Validation},
  author={TKS-LLM Validation Agent},
  year={2025},
  howpublished={TKS-LLM Project}
}
```
