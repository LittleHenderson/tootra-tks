# Agent A: Mixed Fine-Tune Dataset Generation Report

**Date:** 2025-12-15
**Task:** Build 1500-sample mixed fine-tune dataset for improving mixed-holdout performance

---

## Executive Summary

Successfully generated 1500 canonical TKS equation samples with 100% validation against canon guardrails. Dataset split 85/15 into train (1275 samples) and holdout (225 samples) sets.

---

## Task Requirements

### 1. Dataset Composition
- **Total Samples:** 1500
- **Distribution:**
  - 40% standard short equations (2-3 elements) = 600 samples
  - 30% medium equations (4 elements) = 450 samples
  - 20% long chains (5-6 elements) = 300 samples
  - 10% CI-style terse format = 150 samples

### 2. Canon Guardrails (STRICT)
- **Worlds:** A, B, C, D only
- **Noetics:** 1-10 only
- **Operators:** +, -, +T, -T, ->, <-, *T, /T, o (exactly 9)
- **Senses:** ^1-^9
- **Foundations:** _d1-_d7

### 3. Train/Holdout Split
- **Train:** 85% = 1275 samples
- **Holdout:** 15% = 225 samples

---

## Implementation

### Scripts Created

1. **C:\Users\wakil\downloads\everthing-tootra-tks\scripts\create_mixed_finetune.py**
   - Main generation script
   - Implements canonical validation
   - Creates stratified samples across all categories
   - Ensures uniqueness and diversity

2. **C:\Users\wakil\downloads\everthing-tootra-tks\scripts\validate_mixed_finetune.py**
   - Comprehensive validation script
   - Checks canon compliance
   - Reports detailed statistics
   - Identifies any violations

### Output Files

1. **C:\Users\wakil\downloads\everthing-tootra-tks\output\mixed_finetune_train.jsonl**
   - 1275 training samples
   - File size: 183 KB
   - Format: JSONL with fields: story, expr_elements, expr_ops, aug_type

2. **C:\Users\wakil\downloads\everthing-tootra-tks\output\mixed_finetune_holdout.jsonl**
   - 225 holdout samples
   - File size: 32 KB
   - Same format as training set

---

## Validation Results

### Canon Compliance: 100% ✓

**All 1500 entries validated successfully:**
- ✓ All worlds are canonical (A, B, C, D)
- ✓ All noetics are canonical (1-10)
- ✓ All operators are canonical (9 operators)
- ✓ All senses are valid (^1-^9)
- ✓ All foundations are valid (_d1-_d7)

**Zero errors found.**

---

## Dataset Statistics

### Train Set (1275 samples)

**Chain Length Distribution:**
| Elements | Count | Percentage |
|----------|-------|------------|
| 2        | 321   | 25.2%      |
| 3        | 277   | 21.7%      |
| 4        | 421   | 33.0%      |
| 5        | 138   | 10.8%      |
| 6        | 118   | 9.3%       |

**World Distribution:**
- A: 1083 occurrences
- B: 1125 occurrences
- C: 1143 occurrences
- D: 1204 occurrences

**Noetic Distribution (N1-N10):**
All 10 noetics well-represented, ranging from 398 to 502 occurrences each.

**Operator Distribution:**
All 9 canonical operators well-distributed, ranging from 330 to 397 occurrences each.

**Extended Notation:**
- Senses (^1-^9): 955 total annotations
- Foundations (_d1-_d7): 708 total annotations

### Holdout Set (225 samples)

**Chain Length Distribution:**
| Elements | Count | Percentage |
|----------|-------|------------|
| 2        | 59    | 26.2%      |
| 3        | 52    | 23.1%      |
| 4        | 70    | 31.1%      |
| 5        | 24    | 10.7%      |
| 6        | 20    | 8.9%       |

**World Distribution:**
- A: 210 occurrences
- B: 188 occurrences
- C: 185 occurrences
- D: 211 occurrences

**Noetic Distribution (N1-N10):**
All 10 noetics represented, ranging from 71 to 92 occurrences each.

**Operator Distribution:**
All 9 canonical operators present, ranging from 53 to 79 occurrences each.

**Extended Notation:**
- Senses (^1-^9): 157 total annotations
- Foundations (_d1-_d7): 124 total annotations

---

## Sample Data

### Short Equation (2 elements)
```json
{
  "story": "From B7 -> D3",
  "expr_elements": ["B7", "D3"],
  "expr_ops": ["->"],
  "aug_type": "original"
}
```

### Medium Equation (4 elements)
```json
{
  "story": "The equation D6 -> B10 o D8 /T B8 represents",
  "expr_elements": ["D6", "B10", "D8", "B8"],
  "expr_ops": ["->", "o", "/T"],
  "aug_type": "original"
}
```

### Long Chain (5 elements)
```json
{
  "story": "Consider the equation B6 -T D2 *T A4 <- C3 o A2^8_d3",
  "expr_elements": ["B6", "D2", "A4", "C3", "A2^8_d3"],
  "expr_ops": ["-T", "*T", "<-", "o"],
  "aug_type": "original"
}
```

### CI-Style Terse
```json
{
  "story": "Working with C9 -> D4",
  "expr_elements": ["C9", "D4"],
  "expr_ops": ["->"],
  "aug_type": "original"
}
```

---

## Key Features

### 1. Diversity
- Covers all 4 worlds (A, B, C, D)
- Uses all 10 noetics (1-10)
- Employs all 9 canonical operators
- Includes extended notation (senses and foundations)

### 2. Balance
- Stratified sampling across chain lengths
- Even distribution of worlds and noetics
- Balanced operator usage

### 3. Quality
- 100% canonical compliance
- No duplicate equations
- Validated element structure
- Proper operator-element count relationships

### 4. Format Variety
- Standard TKS story format
- CI-style terse format
- Multiple story templates for diversity

---

## Execution Details

### Command Used
```bash
python3 scripts/create_mixed_finetune.py --count 1500 --seed 42 --train-ratio 0.85 --output-dir output
```

### Validation Command
```bash
python3 scripts/validate_mixed_finetune.py
```

### Environment
- Python 3.11.9
- System path: /c/Users/wakil/AppData/Local/Microsoft/WindowsApps/python3
- Working directory: C:\Users\wakil\downloads\everthing-tootra-tks

---

## Completion Checklist

- [x] Generate 1500 samples with specified distribution
- [x] 40% short (2-3 elements) = 600 samples ✓
- [x] 30% medium (4 elements) = 450 samples ✓
- [x] 20% long (5-6 elements) = 300 samples ✓
- [x] 10% CI-style terse = 150 samples ✓
- [x] Enforce canon guardrails (worlds, noetics, operators, senses, foundations)
- [x] Split 85/15 train/holdout (1275/225)
- [x] Save to output/mixed_finetune_train.jsonl
- [x] Save to output/mixed_finetune_holdout.jsonl
- [x] Validate all entries against canon
- [x] Generate statistics and reports

---

## Next Steps

The dataset is ready for:
1. Fine-tuning the TKS model
2. Evaluating on mixed-holdout performance
3. Training with diverse equation structures
4. Testing canonical operator understanding

---

## Files Delivered

1. **output/mixed_finetune_train.jsonl** - 1275 training samples
2. **output/mixed_finetune_holdout.jsonl** - 225 holdout samples
3. **scripts/create_mixed_finetune.py** - Generation script
4. **scripts/validate_mixed_finetune.py** - Validation script
5. **AGENT_A_MIXED_FINETUNE_REPORT.md** - This report

---

**Status:** COMPLETE ✓
**Validation:** 100% PASS ✓
**Canon Compliance:** STRICT ✓
