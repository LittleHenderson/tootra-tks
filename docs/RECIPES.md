# TKS Workflows & Recipes

Complete end-to-end workflows for using the TKS (TOOTRA Knowledge System) toolkit. These recipes demonstrate common patterns for data augmentation, training, evaluation, and inference.

## Table of Contents

- [Recipe 1: Complete Training Pipeline](#recipe-1-complete-training-pipeline)
- [Recipe 2: Anti-Attractor Counter-Scenario Generation](#recipe-2-anti-attractor-counter-scenario-generation)
- [Recipe 3: Bulk Inference Processing](#recipe-3-bulk-inference-processing)
- [Recipe 4: Teacher→Train Flow (External LLM Integration)](#recipe-4-teachertrain-flow-external-llm-integration)
- [Quick Reference](#quick-reference)

---

## Recipe 1: Complete Training Pipeline

**Goal:** Generate augmented training data, train a TKS model, evaluate it, and run inference on new scenarios.

### Prerequisites

- Original corpus in JSONL format with stories or equations
- Python environment with dependencies installed

### Step-by-Step Workflow

#### Step 1: Generate Augmented Data

Generate inverted scenarios and anti-attractor pairs from your original corpus:

```bash
python scripts/generate_augmented_data.py \
  --input data/stories.jsonl \
  --output data/augmented.jsonl \
  --axes W N F \
  --use-anti-attractor \
  --validate \
  --save-metrics
```

**What this does:**
- Loads original stories from `data/stories.jsonl`
- Generates inversions across World (W), Noetic (N), and Foundation (F) axes
- Creates anti-attractor counter-scenarios
- Validates all generated expressions against canonical TKS semantics
- Saves augmented corpus to `data/augmented.jsonl`
- Generates metrics file: `data/augmented.metrics.json`

**Expected Output:**
```
TKS Data Augmentation Pipeline
Input:  data/stories.jsonl
Output: data/augmented.jsonl
Axes combinations: [{'W'}, {'N'}, {'F'}]
Anti-attractor: True
Validation: True

Loaded 100 entries from corpus

Processing entry 1/100...
...

AUGMENTATION COMPLETE
================================================================
Original scenarios:       100
Inverted scenarios:       300
Anti-attractor scenarios: 100
Validation failures:      5

Augmentation ratio:       4.00x
Inversion ratio:          3.00x
Anti-attractor ratio:     1.00x

Validation pass rate:     98.75%
World validity:           100.00%
Noetic validity:          100.00%
Operator validity:        97.50%
Structural validity:      100.00%
================================================================
```

#### Step 2: Train Model with Augmented Data

Train the TKS model using the augmented corpus:

```bash
python scripts/train_with_augmented.py \
  --data data/augmented.jsonl \
  --epochs 10 \
  --batch-size 16 \
  --learning-rate 1e-4 \
  --hidden-dim 128 \
  --output-dir output/models \
  --filter-validated
```

**What this does:**
- Loads augmented data from `data/augmented.jsonl`
- Only uses entries that passed canonical validation (due to `--filter-validated`)
- Trains for 10 epochs with batch size 16
- Uses the real TKSLLMCorePipeline model with multi-component loss
- Saves best model checkpoint to `output/models/best_model.pt`
- Saves final model to `output/models/final_model.pt`
- Logs training metrics to `output/models/metrics/training_metrics.json`

**Expected Output:**
```
TKS TRAINING WITH AUGMENTED DATA - Phase 3 (Real Model)
Device: cuda

Tokenizer vocabulary size: 1000
Loading data from: data/augmented.jsonl
Loaded 495 entries

Augmentation distribution:
  anti_attractor: 100
  inversion: 300
  original: 100

Train size: 445
Eval size: 50

Initializing model...
  Model: TKSLLMCorePipeline
  Hidden dim: 128
  Noetic dim: 40
  Loss: TKSLoss (multi-component)
  Parameters: 245,632

Training configuration:
  Epochs: 10
  Batch size: 16
  Learning rate: 0.0001
  Total steps: 278

================================================================
TRAINING LOOP
================================================================

Epoch 1/10
----------------------------------------------------------------------
  Step 0/27: loss=8.2341, lr=1.00e-04
  Step 10/27: loss=6.5432, lr=9.89e-05
  Step 20/27: loss=5.3210, lr=9.56e-05

  Epoch 1 Summary:
    Average loss: 6.1234
    Steps: 27

  Evaluating...
    Eval loss: 5.8765
    Accuracy: 0.2341
    Perplexity: 356.12
    [NEW BEST] Saved best_model.pt

...

Epoch 10/10
----------------------------------------------------------------------
  ...
  Epoch 10 Summary:
    Average loss: 2.3456
    Steps: 27

  Evaluating...
    Eval loss: 2.5678
    Accuracy: 0.6543
    Perplexity: 13.04

Saved final_model.pt
```

#### Step 3: Evaluate the Trained Model

Run comprehensive evaluation on held-out test data:

```bash
python scripts/evaluate_model.py \
  --checkpoint output/models/best_model.pt \
  --data data/augmented.jsonl \
  --test-ratio 0.2 \
  --batch-size 16 \
  --output output/eval_report.json
```

**What this does:**
- Loads the best model checkpoint
- Evaluates on 20% of the data as test set
- Computes accuracy, perplexity, and component-level metrics
- Validates canonical TKS semantics
- Saves detailed report to `output/eval_report.json`

**Expected Output:**
```
Device: cuda
Tokenizer vocabulary: 1000

Loading data from: data/augmented.jsonl
Total entries: 500
Evaluation subset size: 100

Loading model from: output/models/best_model.pt
Model: TKSLLMCorePipeline

======================================================================
EVALUATION RESULTS
======================================================================

Evaluating model performance...

Model Performance:
  Loss: 2.6543
  Accuracy: 0.6421
  Perplexity: 14.21

Per-Augmentation-Type Accuracy:
  original: 0.6789 (2341 tokens)
  inversion: 0.6532 (7023 tokens)
  anti_attractor: 0.6102 (2156 tokens)

Component Losses:
  task: 2.3456
  rpm: 0.1234
  attractor: 0.0987
  involution: 0.0654

Evaluating canonical validity...

Canonical Validity:
  Full validity rate: 0.9850
  World validity: 1.0000
  Noetic validity: 1.0000
  Operator validity: 0.9750
  Structural validity: 1.0000

Per-Augmentation-Type Validity:
  original: 1.0000 (100 entries)
  inversion: 0.9833 (300 entries)
  anti_attractor: 0.9800 (100 entries)

Report saved to: output/eval_report.json

======================================================================
EVALUATION COMPLETE
======================================================================
```

#### Step 4: Run Inference on New Stories

Use the trained model to process new scenarios:

```bash
# Single story inference
python scripts/run_inference.py \
  --story "A teacher inspires a student" \
  --axes W,N \
  --mode soft \
  --format json

# Bulk inference on JSONL file
python scripts/run_inference.py \
  --input-jsonl data/new_stories.jsonl \
  --output-jsonl output/inference_results.jsonl \
  --axes W,N \
  --mode soft \
  --verbose
```

**Single Story Output (JSON):**
```json
{
  "mode": "inversion",
  "inversion_type": "inversion",
  "original": {
    "expression": "B5 -> D3",
    "elements": ["B5", "D3"],
    "ops": ["->"],
    "story": "A teacher causes growth in a student"
  },
  "inverted": {
    "expression": "C6 -> A2",
    "elements": ["C6", "A2"],
    "ops": ["->"],
    "story": "A student causes wisdom in a teacher"
  },
  "explanation": "Applied World (A<->D, B<->C) and Noetic (5<->6, 3<->2) inversions",
  "validator": {
    "is_valid": true,
    "canon_score": 1.0,
    "error_count": 0,
    "warning_count": 0
  }
}
```

**Bulk Processing Summary:**
```
Processing data/new_stories.jsonl -> output/inference_results.jsonl
Axes: {'World', 'Noetic'}, Mode: soft, Anti-attractor: False
Processed 100 items...
Processed 200 items...
...

Bulk processing complete:
  Total items:  250
  Successful:   248
  Errors:       2
  Parse errors: 0
  Output:       output/inference_results.jsonl
```

### Tips for Success

1. **Start Small:** Test with a small corpus (10-20 entries) before processing large datasets
2. **Monitor Metrics:** Check augmentation metrics to ensure quality (>90% validation pass rate recommended)
3. **Validation Filtering:** Use `--filter-validated` during training to exclude invalid augmentations
4. **Checkpoint Selection:** Use `best_model.pt` for inference (lowest eval loss) rather than `final_model.pt`
5. **Batch Size:** Adjust based on GPU memory (reduce if OOM errors occur)

---

## Recipe 2: Anti-Attractor Counter-Scenario Generation

**Goal:** Generate counter-scenarios that oppose the dominant attractor patterns in a given scenario.

### What are Anti-Attractors?

Anti-attractors generate scenarios that:
- Invert dominant world/noetic combinations
- Flip polarity (positive to negative or vice versa)
- Counter foundation patterns
- Create narrative oppositions for balanced training data

### Single Scenario Anti-Attractor

Generate a counter-scenario for a single input:

```bash
python scripts/run_scenario_inversion.py \
  --story "Power corrupts the righteous" \
  --anti-attractor \
  --format json
```

**Output:**
```json
{
  "original": {
    "expr": "C5 +T A7",
    "story": "Power corrupts the righteous"
  },
  "anti_attractor": {
    "expr": "D6 -T B2",
    "story": "Humility purifies the wise"
  },
  "signature": {
    "dominant_world": "C",
    "dominant_noetic": 5,
    "polarity": -1,
    "foundation_tags": [5, 7]
  },
  "explanation": "Anti-attractor inverts C5 dominance to D6, flips negative polarity to positive"
}
```

### Bulk Anti-Attractor Generation

Generate anti-attractors for an entire corpus:

```bash
python scripts/generate_augmented_data.py \
  --input data/original_corpus.jsonl \
  --output data/anti_attractor_corpus.jsonl \
  --use-anti-attractor \
  --validate \
  --verbose
```

**Input Format (data/original_corpus.jsonl):**
```jsonl
{"story": "Power corrupts the righteous", "id": "001"}
{"equation": "B5 -> D3", "id": "002"}
{"story": "Love conquers fear", "id": "003"}
```

**Output Format (data/anti_attractor_corpus.jsonl):**
```jsonl
{"id": "001", "story": "Power corrupts the righteous", "expr": "C5 +T A7", "aug_type": "original", "validator_pass": true}
{"id": "001_anti", "story": "Humility purifies the wise", "expr": "D6 -T B2", "aug_type": "anti_attractor", "validator_pass": true}
{"id": "002", "story": "A person in conflict causes growth", "expr": "B5 -> D3", "aug_type": "original", "validator_pass": true}
{"id": "002_anti", "story": "A person at peace prevents knowledge", "expr": "C6 <- A10", "aug_type": "anti_attractor", "validator_pass": true}
...
```

### CLI Reference for Anti-Attractor Mode

```bash
# Basic anti-attractor synthesis
python scripts/run_inference.py \
  --story "Your story here" \
  --anti-attractor

# With JSON output and validation
python scripts/run_inference.py \
  --equation "B5 +T D3" \
  --anti-attractor \
  --format json

# Bulk processing with anti-attractors
python scripts/run_inference.py \
  --input-jsonl input.jsonl \
  --output-jsonl output.jsonl \
  --anti-attractor \
  --verbose
```

### Understanding Anti-Attractor Signatures

An attractor signature captures:

1. **Element Distribution:** Frequency of each world-noetic combination
2. **Dominant Pattern:** Most frequent world and noetic index
3. **Polarity:** Overall valence (+1 positive, -1 negative, 0 neutral)
4. **Foundation Tags:** Active foundations in the scenario

**Example Signature Analysis:**
```
Original Signature:
  Element Distribution:
    C5: 3 occurrences
    C7: 2 occurrences
    A5: 1 occurrence
  Dominant: C5
  Polarity: -1 (Negative)
  Foundations: F5 (Power), F7 (Lust)

Anti-Attractor Signature:
  Element Distribution:
    D6: 3 occurrences
    D2: 2 occurrences
    B6: 1 occurrence
  Dominant: D6
  Polarity: +1 (Positive)
  Foundations: F6 (Material), F2 (Wisdom)
```

### Use Cases

1. **Balanced Training Data:** Prevent model bias toward specific attractors
2. **Narrative Exploration:** Generate contrasting scenarios for creative writing
3. **Ethical Analysis:** Explore oppositions in moral/philosophical scenarios
4. **Augmentation Diversity:** Increase variety beyond simple axis inversions

---

## Recipe 3: Bulk Inference Processing

**Goal:** Process large batches of scenarios efficiently using JSONL input/output.

### Basic Bulk Inference

Process multiple scenarios with standard inversion:

```bash
python scripts/run_inference.py \
  --input-jsonl input_scenarios.jsonl \
  --output-jsonl output_results.jsonl \
  --axes W,N \
  --mode soft \
  --verbose
```

### Input File Format

Create `input_scenarios.jsonl` with one JSON object per line:

```jsonl
{"story": "A teacher inspires a student"}
{"equation": "B5 -> D3"}
{"story": "Power corrupts", "axes": "F", "mode": "targeted"}
{"story": "Love conquers fear", "anti_attractor": true}
```

### Per-Item Configuration Override

Each item can override default settings:

```jsonl
{"story": "Custom axes example", "axes": "W,F", "mode": "hard"}
{"story": "Targeted example", "mode": "targeted", "from_foundation": 5, "to_foundation": 2}
{"story": "Lenient mode example", "lenient": true}
```

### Output File Format

Each output line contains comprehensive results and metadata:

```jsonl
{
  "success": true,
  "input": {"story": "A teacher inspires a student"},
  "inversion_type": "inversion",
  "result": {
    "original": {
      "expression": "B5 -> D3",
      "elements": ["B5", "D3"],
      "ops": ["->"],
      "story": "A teacher causes growth in a student"
    },
    "inverted": {
      "expression": "C6 -> A2",
      "elements": ["C6", "A2"],
      "ops": ["->"],
      "story": "A student causes wisdom in a teacher"
    }
  },
  "validator": {
    "is_valid": true,
    "canon_score": 1.0,
    "error_count": 0,
    "warning_count": 0
  }
}
```

### Advanced Bulk Processing Options

#### Skip Validation for Speed

```bash
python scripts/run_inference.py \
  --input-jsonl large_corpus.jsonl \
  --output-jsonl results.jsonl \
  --no-validator
```

Skipping validation improves throughput for trusted inputs.

#### Anti-Attractor Bulk Mode

```bash
python scripts/run_inference.py \
  --input-jsonl stories.jsonl \
  --output-jsonl anti_attractors.jsonl \
  --anti-attractor \
  --verbose
```

Generates anti-attractor counter-scenarios for all inputs.

#### Mixed Mode Processing

Use per-item configuration for varied processing:

**Input (mixed_config.jsonl):**
```jsonl
{"story": "Standard inversion", "axes": "W,N"}
{"story": "Anti-attractor", "anti_attractor": true}
{"story": "Targeted remap", "mode": "targeted", "from_world": "B", "to_world": "D"}
```

**Command:**
```bash
python scripts/run_inference.py \
  --input-jsonl mixed_config.jsonl \
  --output-jsonl mixed_results.jsonl \
  --verbose
```

### Processing Large Files

For very large files (10k+ entries):

```bash
# Process in chunks with progress monitoring
python scripts/run_inference.py \
  --input-jsonl large_file.jsonl \
  --output-jsonl results.jsonl \
  --no-validator \
  --verbose 2>&1 | tee processing.log
```

Monitor `processing.log` for:
- Progress updates (every 100 items)
- Error messages
- Final statistics

### Error Handling

Failed items are included in output with error details:

```jsonl
{
  "success": false,
  "input": {"story": "Invalid input with unknown tokens @#$%"},
  "error": "Validation error: Unknown token '@' at position 30"
}
```

Filter successful results:

```bash
# Extract only successful results
jq 'select(.success == true)' output_results.jsonl > successful_only.jsonl

# Count errors
jq 'select(.success == false)' output_results.jsonl | wc -l
```

### Batch Statistics

View processing summary:

```bash
python scripts/run_inference.py \
  --input-jsonl input.jsonl \
  --output-jsonl output.jsonl \
  --verbose
```

**Output:**
```
Bulk processing complete:
  Total items:  1000
  Successful:   987
  Errors:       13
  Parse errors: 0
  Output:       output.jsonl
```

---

## Recipe 4: Teacher→Train Flow (External LLM Integration)

**Goal:** Use external LLMs (e.g., Gemini, GPT-4) as "teachers" to generate TKS interpretations, validate them, optionally augment, then train a smaller student model.

### What is the Teacher→Train Flow?

This workflow leverages powerful external LLMs to bootstrap high-quality TKS training data:
1. **Teacher Generation:** External LLMs interpret stories/equations as TKS expressions
2. **Canonical Validation:** Filter outputs to ensure TKS semantic compliance
3. **Augmentation (Optional):** Apply inversions and anti-attractors to validated data
4. **Student Training:** Train a smaller, faster model on teacher-generated data
5. **Evaluation:** Assess student model performance

### Prerequisites

- API keys for external LLM providers (e.g., Gemini, OpenAI)
- Input corpus: stories or equations in JSONL format
- Python environment with dependencies installed

### Step-by-Step Workflow

#### Step 1: Prepare Input Equations

Create an input file with stories or equations to interpret:

**Example: data/equations.jsonl**
```jsonl
{"story": "A teacher inspires a student", "id": "eq_001"}
{"equation": "B5 -> D3", "id": "eq_002"}
{"story": "Power corrupts the righteous", "id": "eq_003"}
```

#### Step 2: Generate Teacher Interpretations

Use external LLMs to generate TKS interpretations:

```bash
python scripts/run_teacher.py generate data/equations.jsonl \
  --output data/teacher_outputs.jsonl \
  --providers gemini:gemini-1.5-pro \
  --min-canon 0.8 \
  --temperature 0.7
```

**What this does:**
- Sends each input to Gemini 1.5 Pro for TKS interpretation
- Filters outputs with canonical validity score >= 0.8
- Saves teacher-generated expressions to `data/teacher_outputs.jsonl`
- Includes validation metadata for each output

**Expected Output:**
```
Teacher Generation Pipeline
Provider: gemini:gemini-1.5-pro
Input: data/equations.jsonl
Output: data/teacher_outputs.jsonl
Min canonical score: 0.8

Processing 3 equations...
  [1/3] eq_001: Success (canon=0.95)
  [2/3] eq_002: Success (canon=1.0)
  [3/3] eq_003: Success (canon=0.88)

Generation complete:
  Total: 3
  Success: 3
  Failed: 0
  Avg canonical score: 0.943
```

**Output Format (data/teacher_outputs.jsonl):**
```jsonl
{"id": "eq_001", "story": "A teacher inspires a student", "expr": "B5 -> D3", "provider": "gemini", "canon_score": 0.95, "validator_pass": true}
{"id": "eq_002", "story": "A person in conflict causes growth", "expr": "B5 -> D3", "provider": "gemini", "canon_score": 1.0, "validator_pass": true}
{"id": "eq_003", "story": "Power corrupts the righteous", "expr": "C5 +T A7", "provider": "gemini", "canon_score": 0.88, "validator_pass": true}
```

#### Step 3: Validate Teacher Outputs

Run strict canonical validation on teacher outputs:

```bash
python scripts/canonical_validator.py \
  --input data/teacher_outputs.jsonl \
  --output data/teacher_outputs_valid.jsonl \
  --min-score 0.9 \
  --verbose
```

**What this does:**
- Applies comprehensive canonical TKS validation
- Filters entries with validation score >= 0.9
- Checks worlds (A-D), noetics (1-10), operators (9 allowed), structure
- Saves only validated entries to `data/teacher_outputs_valid.jsonl`

**Expected Output:**
```
Canonical Validation Pipeline
Input: data/teacher_outputs.jsonl
Output: data/teacher_outputs_valid.jsonl
Min score: 0.9

Validating 3 entries...

Validation Results:
  Total entries: 3
  Passed: 3
  Failed: 0
  Pass rate: 100.00%

Component Validity:
  World validity: 100.00%
  Noetic validity: 100.00%
  Operator validity: 100.00%
  Structural validity: 100.00%

Saved 3 validated entries to: data/teacher_outputs_valid.jsonl
```

#### Step 4: Augment Teacher Data (Optional)

Apply inversions and anti-attractors to expand the validated dataset:

```bash
python scripts/generate_augmented_data.py \
  --input data/teacher_outputs_valid.jsonl \
  --output data/teacher_augmented.jsonl \
  --axes W N F \
  --use-anti-attractor \
  --validate \
  --save-metrics
```

**What this does:**
- Generates inversions across World, Noetic, and Foundation axes
- Creates anti-attractor counter-scenarios
- Validates all augmented expressions
- Produces augmentation metrics

**Expected Output:**
```
TKS Data Augmentation Pipeline
Input:  data/teacher_outputs_valid.jsonl
Output: data/teacher_augmented.jsonl

Loaded 3 entries from corpus

Processing augmentations...

AUGMENTATION COMPLETE
================================================================
Original scenarios:       3
Inverted scenarios:       9
Anti-attractor scenarios: 3
Validation failures:      0

Augmentation ratio:       5.00x
Validation pass rate:     100.00%
================================================================

Saved to: data/teacher_augmented.jsonl
Metrics: data/teacher_augmented.metrics.json
```

#### Step 5: Train Student Model

Train a smaller student model on teacher-generated data:

```bash
python scripts/train_with_augmented.py \
  --data data/teacher_augmented.jsonl \
  --epochs 10 \
  --batch-size 16 \
  --learning-rate 1e-4 \
  --output-dir output/models \
  --use-augmented \
  --filter-validated
```

**What this does:**
- Loads teacher-generated augmented data
- Trains TKSLLMCorePipeline model with multi-component loss
- Uses only validated entries (`--filter-validated`)
- Saves best checkpoint based on eval loss
- Logs training metrics

**Expected Output:**
```
TKS TRAINING WITH AUGMENTED DATA
Device: cuda

Loading data from: data/teacher_augmented.jsonl
Loaded 15 entries

Augmentation distribution:
  original: 3
  inversion: 9
  anti_attractor: 3

Train size: 13
Eval size: 2

Training for 10 epochs...
Epoch 1/10: loss=5.234, eval_loss=4.876, accuracy=0.312
...
Epoch 10/10: loss=1.543, eval_loss=1.678, accuracy=0.745

Saved best_model.pt (eval_loss: 1.654)
Saved final_model.pt
```

#### Step 6: Evaluate Student Model

Assess the trained student model's performance:

```bash
python scripts/evaluate_model.py \
  --checkpoint output/models/best_model.pt \
  --data data/teacher_augmented.jsonl \
  --output output/eval_report.json \
  --test-ratio 0.2
```

**What this does:**
- Loads best student model checkpoint
- Evaluates on held-out test data
- Computes accuracy, perplexity, component losses
- Validates canonical compliance
- Saves detailed evaluation report

**Expected Output:**
```
EVALUATION RESULTS
======================================================================

Model Performance:
  Loss: 1.723
  Accuracy: 0.732
  Perplexity: 5.61

Canonical Validity:
  Full validity rate: 0.967
  World validity: 1.000
  Noetic validity: 1.000
  Operator validity: 0.933

Report saved to: output/eval_report.json
======================================================================
```

### Alternative: Direct Training Without Augmentation

For faster iteration, skip augmentation and train directly on validated teacher outputs:

```bash
# Validate teacher outputs
python scripts/canonical_validator.py \
  --input data/teacher_outputs.jsonl \
  --output data/teacher_valid.jsonl \
  --min-score 0.9

# Train directly (no augmentation)
python scripts/train_with_augmented.py \
  --data data/teacher_valid.jsonl \
  --epochs 15 \
  --batch-size 16 \
  --learning-rate 1e-4 \
  --output-dir output/models \
  --filter-validated
```

### Multi-Provider Teacher Ensemble

Use multiple LLM providers for diverse interpretations:

```bash
python scripts/run_teacher.py generate data/equations.jsonl \
  --output data/teacher_multi.jsonl \
  --providers gemini:gemini-1.5-pro openai:gpt-4 \
  --min-canon 0.85 \
  --ensemble-mode vote
```

**Ensemble modes:**
- `vote`: Use majority vote across providers
- `highest`: Select interpretation with highest canonical score
- `all`: Include all provider outputs (most data)

### Canonical Constraints Reminder

All teacher outputs must comply with TKS canonical semantics:

- **Worlds:** A, B, C, D only
- **Noetics:** 1-10 only (pairs: 2↔3, 5↔6, 8↔9; self-duals: 1,4,7,10)
- **Foundations:** 1-7 only
- **Operators:** +, -, +T, -T, ->, <-, *T, /T, o (9 total)

Validation ensures teacher outputs adhere to these constraints before training.

### Tips for Teacher→Train Success

1. **Start with High-Quality Inputs:** Clear, unambiguous stories yield better teacher interpretations
2. **Set Appropriate Canon Thresholds:** Use `--min-canon 0.85-0.95` for teacher generation
3. **Validate Strictly:** Use `--min-score 0.9` for validation to ensure training data quality
4. **Monitor Teacher Diversity:** Check that teachers generate varied expressions, not repetitive patterns
5. **Augment Strategically:** Use augmentation for smaller datasets; may skip for large teacher corpora
6. **Evaluate Teacher-Student Gap:** Compare teacher canonical scores with student model outputs

### Common Pitfalls

1. **Low Teacher Canon Scores:** Review prompts, adjust temperature, or try different providers
2. **Overfitting to Teacher Style:** Augmentation helps diversify beyond teacher's specific patterns
3. **Validation Bottleneck:** If too many teacher outputs fail validation, adjust `--min-canon` threshold
4. **Small Dataset:** Generate more teacher examples or apply aggressive augmentation

---

## Quick Reference

### Common Commands

```bash
# Teacher generation (external LLM)
python scripts/run_teacher.py generate INPUT.jsonl \
  --output TEACHER_OUT.jsonl --providers gemini:gemini-1.5-pro \
  --min-canon 0.8

# Canonical validation
python scripts/canonical_validator.py \
  --input TEACHER_OUT.jsonl --output VALIDATED.jsonl \
  --min-score 0.9

# Data augmentation
python scripts/generate_augmented_data.py \
  --input INPUT.jsonl --output OUTPUT.jsonl \
  --axes W N F --use-anti-attractor --validate

# Training
python scripts/train_with_augmented.py \
  --data AUGMENTED.jsonl --epochs 10 --batch-size 16 \
  --output-dir models/ --filter-validated

# Evaluation
python scripts/evaluate_model.py \
  --checkpoint models/best_model.pt --data AUGMENTED.jsonl \
  --output eval_report.json

# Single inference
python scripts/run_inference.py \
  --story "Your story" --axes W,N --format json

# Bulk inference
python scripts/run_inference.py \
  --input-jsonl INPUT.jsonl --output-jsonl OUTPUT.jsonl \
  --axes W,N --verbose

# Anti-attractor synthesis
python scripts/run_scenario_inversion.py \
  --story "Your story" --anti-attractor
```

### Axis Codes

- **N** = Noetic (involution pairs: 2<->3, 5<->6, 8<->9; self-duals: 1,4,7,10)
- **E** = Element (full element inversion: world + noetic)
- **W** = World (world mirror: A<->D, B<->C)
- **F** = Foundation (1<->7, 2<->6, 3<->5, 4 self-dual)
- **S** = SubFoundation (foundation + world compound)
- **A** = Acquisition (negation toggle)
- **P** = Polarity (valence flip)

### Modes

- **soft** - Invert only where canonical dual/opposite exists (recommended default)
- **hard** - Apply inversion unconditionally on all selected axes
- **targeted** - Use TargetProfile remaps; leave others unchanged

### Canonical Constraints

- **Worlds:** A, B, C, D only
- **Noetics:** 1-10 only
- **Foundations:** 1-7 only
- **Operators:** +, -, +T, -T, ->, <-, *T, /T, o (9 total)

### Validation Thresholds

- **Training Data:** >90% canonical pass rate recommended
- **Production Use:** 100% validation for critical applications
- **Exploratory:** Lenient mode acceptable with `--lenient` flag

### Performance Tips

1. **Batch Size:** Start with 16, reduce if OOM errors occur
2. **Validation:** Skip with `--no-validator` for large bulk jobs (trusted inputs only)
3. **GPU Usage:** CUDA enabled automatically if available
4. **Checkpointing:** Save frequently during long training runs
5. **Data Quality:** Filter low-quality augmentations with `--filter-validated`

---

## Troubleshooting

### Issue: "Unknown token" errors

**Solution:** Use `--lenient` flag to allow unknown tokens with warnings:
```bash
python scripts/run_inference.py --story "Text with unknowns" --lenient
```

### Issue: Low validation pass rate (<90%)

**Solutions:**
1. Review input corpus for canonical compliance
2. Reduce number of inversion axes
3. Use `soft` mode instead of `hard`
4. Inspect failed entries in metrics file

### Issue: Training loss not decreasing

**Solutions:**
1. Increase learning rate: `--learning-rate 5e-4`
2. Reduce batch size for more frequent updates
3. Train longer: `--epochs 20`
4. Check data quality in augmented corpus

### Issue: Bulk processing errors

**Solutions:**
1. Validate JSONL format: one JSON object per line
2. Check for special characters needing escaping
3. Use `--verbose` to identify problematic entries
4. Review error entries in output file

---

## Next Steps

- **Deep Dive:** See `docs/SCENARIO_INVERSION.md` for complete inversion semantics
- **Anti-Attractor Details:** See `docs/anti_attractor_guide.md` for attractor theory
- **Validation:** See `docs/Agent3_Validation_Implementation.md` for canonical rules
- **Telemetry:** See `docs/TELEMETRY_GUIDE.md` for metrics tracking

For questions or issues, consult the main README.md or project documentation.
