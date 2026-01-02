# Dual-Task Story-Equation Converter

## Overview

The `convert_story_equation.py` script converts story-equation paired data into training format for bidirectional learning, enabling the model to learn both:
- **Story-to-Equation**: Translating natural language narratives into TKS equations
- **Equation-to-Story**: Generating natural language narratives from TKS equations

## Files

### Scripts
- **`scripts/convert_story_equation.py`**: Main converter script with full canon validation
- **`scripts/test_tokenizer_coverage.py`**: Tests tokenizer coverage for all required tokens

### Data
- **Input (source)**:
  - `data/story_equation_pairs_train.jsonl` (680 samples)
  - `data/story_equation_pairs_holdout.jsonl` (120 samples)

- **Output (training format)**:
  - `output/teacher_story_eq_train.jsonl`
  - `output/teacher_story_eq_holdout.jsonl`

## Data Formats

### Source Format
```json
{
  "story": "This equation describes how...",
  "equation": "A1 +T B2",
  "expr_elements": ["A1", "B2"],
  "expr_ops": ["+T"],
  "direction": "story_to_eq",
  "pair_id": "pair_0000"
}
```

Each pair appears **twice** in the source data - once for each direction.

### Target Format (Training)
```json
{
  "task_type": "story_to_equation",
  "input": "Given this TKS narrative:\n\n...\n\nTranslate this...",
  "target": "A1 +T B2",
  "metadata": {
    "elements": ["A1", "B2"],
    "elements_base": ["A1", "B2"],
    "operators": ["+T"],
    "pair_id": "pair_0000",
    "validation_errors": null,
    "validation_warnings": null
  },
  "direction": "story_to_eq",
  "canon_score": 1.0
}
```

## Canon Validation

The converter implements **strict canonical validation** on all equation components:

### Worlds (Strict)
- **Valid**: A, B, C, D only
- **Invalid**: Y, Z, E-X, or any other letters

### Noetics (Strict)
- **Valid**: 1-10 only
- **Invalid**: 0, 11+, or any out-of-range values

### Operators (Strict - exactly 9)
- **Valid**: `+`, `-`, `+T`, `-T`, `->`, `<-`, `*T`, `/T`, `o`
- **Invalid**: Any other operator symbols

### Extended Notation

#### Senses
- **Format**: `^k` where k = 1-9
- **Example**: `A1^3`, `B4^7`
- **Invalid**: `^0`, `^10+`

#### Foundations
- **Format**: `_dk` where k = 1-7
- **Example**: `A1_d3`, `B4_d5`
- **Correspondence**:
  - `_d1` = Unity
  - `_d2` = Wisdom
  - `_d3` = Life
  - `_d4` = Companionship
  - `_d5` = Power
  - `_d6` = Material
  - `_d7` = Lust

#### Combined Extended Notation
- **Format**: `[World][Noetic]^[sense]_d[foundation]`
- **Example**: `A7^6_d3`, `C9^7_d1`, `D10^1_d1`

## Usage

### Convert Training Data
```bash
python scripts/convert_story_equation.py \
    --input data/story_equation_pairs_train.jsonl \
    --output output/teacher_story_eq_train.jsonl
```

### Convert Holdout Data
```bash
python scripts/convert_story_equation.py \
    --input data/story_equation_pairs_holdout.jsonl \
    --output output/teacher_story_eq_holdout.jsonl
```

### Strict Mode (Skip Invalid Entries)
```bash
python scripts/convert_story_equation.py \
    --input data/story_equation_pairs_train.jsonl \
    --output output/teacher_story_eq_train.jsonl \
    --strict
```

## Validation Results

Running the converter on the Track 2 data:

```
Converting: data/story_equation_pairs_train.jsonl
============================================================
  Total entries:          680
  Converted:              680
  Skipped:                  0

  By direction:
    eq_to_story             340
    story_to_eq             340

  Canon validation:
    Valid (score=1.0):    680
    Invalid (score<1):      0
    Average score:      1.000
============================================================
```

All 680 training samples and 120 holdout samples pass canon validation with perfect scores.

## Tokenizer Coverage

Run the coverage test:
```bash
python scripts/test_tokenizer_coverage.py
```

### Test Results

✓ **Base Elements**: All 40 elements (A1-D10) present
✓ **Operators**: All 9 canonical operators present
✓ **Extended Notation**: Senses (^1-^9) and foundations (_d1-_d7) supported
✓ **Direction Tags**: Both `story_to_eq` and `eq_to_story` encodable
✓ **Vocabulary Size**: 127 tokens used out of 1000 capacity

### Tokenizer Note

The current tokenizer in `scripts/quick_train.py` uses a **hybrid approach**:
1. **Whole-token operators**: `/T`, `+T`, `-T`, `*T`, `->`, `<-` are stored as complete tokens
2. **Character-level encoding**: Other text is encoded character-by-character

This works correctly for **training** but the character-level decode function has limitations when reconstructing operators. For production use, consider using a proper tokenizer like:
- HuggingFace `AutoTokenizer` with custom vocabulary
- SentencePiece with TKS-specific tokens
- BPE tokenizer pre-trained on TKS corpus

## Bidirectional Training Architecture

### Task Types

1. **story_to_equation**
   - **Input**: Natural language narrative
   - **Output**: TKS equation string
   - **Use case**: User describes intent → system generates formal equation

2. **equation_to_story**
   - **Input**: TKS equation with elements
   - **Output**: Natural language narrative
   - **Use case**: System explains equation → user understands meaning

### Training Strategy

The dual-task format enables:
- **Multi-task learning**: Single model handles both directions
- **Consistency regularization**: Paired samples enforce bidirectional consistency
- **Improved generalization**: Model learns deeper equation-story relationships

### Model Input Format

#### Story → Equation
```
Given this TKS narrative:

[STORY TEXT]

Translate this into a TKS equation using canonical notation (worlds A,B,C,D; noetics 1-10).
```

#### Equation → Story
```
Given the TKS equation: [EQUATION]

Elements: [ELEMENT_LIST]

Generate a natural language narrative describing this TKS working.
```

## Integration with Training Pipeline

### 1. Data Preparation
```bash
# Convert paired data
python scripts/convert_story_equation.py \
    --input data/story_equation_pairs_train.jsonl \
    --output output/teacher_story_eq_train.jsonl

python scripts/convert_story_equation.py \
    --input data/story_equation_pairs_holdout.jsonl \
    --output output/teacher_story_eq_holdout.jsonl
```

### 2. Training
The converted files are compatible with existing training scripts:
- `scripts/quick_train.py`
- `scripts/train_enhanced.py`
- `scripts/train_capacity.py`

Simply update the data path to point to the story-equation files:
```python
train_dataset = TKSDataset("output/teacher_story_eq_train.jsonl", tokenizer)
holdout_dataset = TKSDataset("output/teacher_story_eq_holdout.jsonl", tokenizer)
```

### 3. Evaluation
Use `scripts/phase6_eval.py` with both task types:
```python
# Test story → equation
result = model.generate(story_prompt)

# Test equation → story
result = model.generate(equation_prompt)
```

## Canon Guardrails Summary

| Component | Valid Range | Example | Notes |
|-----------|-------------|---------|-------|
| Worlds | A, B, C, D | `A1`, `D10` | Exactly 4 worlds |
| Noetics | 1-10 | `A1`, `B10` | Exactly 10 noetics |
| Operators | +, -, +T, -T, ->, <-, *T, /T, o | `A1 +T B2` | Exactly 9 operators |
| Senses | ^1 to ^9 | `A1^3` | 9 sense levels |
| Foundations | _d1 to _d7 | `A1_d3` | 7 foundations |
| Combined | Both | `A7^6_d3` | Sense + foundation |

## Error Handling

### Validation Errors
The converter reports validation errors but continues processing:
```
Warning: Canon validation failed for entry 42:
  ERROR: Element validation: Non-canonical world 'Y' in element 'Y5'
  ERROR: Operator validation: Invalid operator '**' (must be one of: +, -, +T, ...)
```

### Strict Mode
With `--strict` flag, entries with canon violations are skipped entirely.

### Canon Score
Each entry receives a `canon_score` (0.0-1.0):
- **1.0**: Perfect canon compliance
- **<1.0**: Has warnings or errors
- **Deduction**: -0.2 per error, -0.05 per warning

## Future Enhancements

1. **Multi-provider augmentation**: Generate additional story-equation pairs using ensemble teacher models
2. **Difficulty stratification**: Categorize pairs by complexity for curriculum learning
3. **Interactive refinement**: Allow manual correction of low-score entries
4. **Synthetic data generation**: Create new pairs from canonical patterns
5. **Cross-validation**: Ensure story↔equation consistency through roundtrip testing

## References

- **Canon Validator**: `teacher/validator.py`
- **Teacher Format Converter**: `scripts/convert_teacher_format.py`
- **Training Scripts**: `scripts/quick_train.py`, `scripts/train_enhanced.py`
- **Evaluation**: `scripts/phase6_eval.py`

---

**Track 2 Agent F**: Converter/Loader
**Status**: ✓ Complete
**Canon Compliance**: 100% (800/800 samples validated)
