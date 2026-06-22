# TKS 4-Operation Block V2: Negation Scope Experiment

## Overview

This experimental framework tests the core TKS subtractive hypothesis: **models with separate inhibition attention should outperform standard transformers on negation scope resolution tasks.**

## What's Been Built

### 1. Core Architecture (`tks_4op_block_v2.py`)

**TKS4OpBlockV2** - Complete 4-operation block with:
- ✅ Separate attention contexts (ADD uses `attn_add`, SUB uses `attn_inh`)
- ✅ Independent sigmoid gates (no softmax competition)
- ✅ Interpretable inhibition attention (returns attention patterns)
- ✅ Stability mechanisms (alpha_sub scaling, tanh bounding)
- ✅ Comprehensive instrumentation

**Three Model Variants:**
- **Condition A**: `SimpleTransformerLM` - Baseline transformer (control)
- **Condition B**: `SubtractiveLM` - ADD + SUB only (core hypothesis test)
- **Condition C**: `Full4OpLM` - All 4 operations (full system test)

### 2. Dataset Generator (`negation_scope_dataset.py`)

**NegationScopeDataset** - Synthetic negation scope examples:
```
"The wizard is wise" → wise=1
"The wizard is not wise" → wise=0
"The warrior is brave and not cowardly" → brave=1, cowardly=0
```

**Features:**
- Configurable negation probability
- Multiple properties per example (1-3)
- Tracks NOT positions and property positions for mechanistic probes
- QA variant available

### 3. Experimental Framework (`experiment_negation_scope.py`)

**Complete A/B/C experimental pipeline:**
- Training with AdamW + OneCycle LR
- Validation-based early stopping
- Test set evaluation with mechanistic probe
- Automated comparison table generation
- GPU-only (enforces project policy)

### 4. Mechanistic Probe (`mechanistic_probe.py`)

**The "Smoking Gun" Test:**

Analyzes whether inhibition attention learns to focus on negated predicates:
```
For each "not" token:
  - Measure attention to predicates vs other tokens
  - Ratio > 2-3x = meaningful inhibition targeting
```

Includes visualization of attention patterns.

## Installation

```bash
# Ensure CUDA GPU is available (REQUIRED by project policy)
nvidia-smi

# Install dependencies
pip install torch numpy matplotlib seaborn tqdm

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Quick Start

### Run Full Experiment (All Conditions)

```bash
# Default: 20 epochs, 10k training samples
python experiment_negation_scope.py

# Custom configuration
python experiment_negation_scope.py \
  --conditions A B C \
  --epochs 30 \
  --batch-size 64 \
  --lr 2e-4 \
  --train-samples 20000 \
  --save-dir experiments/my_experiment
```

### Run Single Condition

```bash
# Test only the subtractive hypothesis (Condition B)
python experiment_negation_scope.py --conditions B
```

### Analyze Trained Model

```bash
# Run mechanistic probe on trained model
python mechanistic_probe.py \
  --model-path experiments/negation_scope/model_B.pt \
  --model-type B \
  --num-examples 200 \
  --visualize \
  --output-dir probe_results_B
```

## Expected Results

### If Hypothesis is Correct:

**Performance:**
```
Condition A (Baseline):    Loss ~2.5, Accuracy ~0.65
Condition B (Subtractive): Loss ~2.2, Accuracy ~0.72  (↑ 10%+ improvement)
Condition C (Full 4-Op):   Loss ~2.1, Accuracy ~0.73  (marginal vs B)
```

**Mechanistic Probe (Conditions B/C):**
```
Attention: NOT → Predicate:     ~0.25
Attention: NOT → Other tokens:  ~0.08
Ratio:                          ~3.0x

Interpretation: ✓✓ STRONG EVIDENCE - Inhibition attention focuses on predicates
```

### If Hypothesis is Incorrect:

```
All conditions perform similarly (< 5% difference)
Probe ratio < 1.5x (no meaningful predicate focusing)
```

## Output Structure

```
experiments/negation_scope/
├── config.json              # Experiment configuration
├── vocab.txt                # Vocabulary mapping
├── model_A.pt              # Baseline model checkpoint
├── model_B.pt              # Subtractive model checkpoint
├── model_C.pt              # Full 4-Op model checkpoint
└── results.json            # Comparison results

probe_results_B/
├── probe_results.json      # Mechanistic probe statistics
└── visualizations/
    ├── attention_example_1.png
    ├── attention_example_2.png
    └── ...
```

## Architecture Details

### Separate Inhibition Attention (Key Innovation)

```python
# V1 Problem: Shared context
c = self.attn(x)  # Same context for all operations

# V2 Solution: Separate contexts
c_add = self.attn_add(x)  # For accumulation
c_inh = self.attn_inh(x)  # For inhibition (can learn orthogonal patterns)
```

### Independent Gates (No Collapse)

```python
# V1 Problem: Softmax competition
gates = softmax([g_add, g_sub, g_mul, g_div])  # Sum to 1, suppress each other

# V2 Solution: Independent sigmoid
w_add = sigmoid(g_add)  # Can all be active simultaneously
w_sub = sigmoid(g_sub)
w_mul = sigmoid(g_mul)
w_div = sigmoid(g_div)
```

### Operation Execution

```python
# ADD: Accumulation
term_add = w_add * c_add

# SUB: Inhibition with stability
term_sub = w_sub * alpha_sub * tanh(c_inh)  # Bounded to [-alpha, +alpha]

# MUL: Gating/binding
g_mul = sigmoid(mul_gate_proj(c_add))
term_mul = x * (1.0 + w_mul * g_mul)

# DIV: Normalization
denom = eps + softplus(div_denom_proj(c_add))
term_div = x / (eps + w_div * denom)

# Combine
out = x + term_add - term_sub  # Residual + ADD - SUB
out = term_mul                 # Apply MUL gating
out = term_div                 # Apply DIV normalization
```

## Known Limitations

### 1. MUL/DIV Use ADD Context

Currently `c_add` is used for both MUL and DIV projections. This is probably fine for initial testing (the key hypothesis is about inhibition), but if you later want to test whether "what to bind" differs from "what to accumulate," you'd need:

```python
# Future enhancement
c_mul = self.attn_mul(z_mul)  # Separate binding context
```

### 2. Gate Statistics Are Detached

```python
stats["w_add"] = w_add.mean().item()  # Breaks gradient
```

This means you can't backprop through gate stats for regularization. For initial experiments this is fine. If you see collapse during training (e.g., `w_sub → 0`), add auxiliary loss:

```python
# Inside forward, before detaching
gate_entropy = -(w_sub * log(w_sub + 1e-8) +
                 (1-w_sub) * log(1-w_sub + 1e-8)).mean()
# Add to loss: loss = task_loss - 0.01 * gate_entropy
```

### 3. No Auxiliary Loss by Default

Independent gates reduce but don't eliminate collapse risk. Monitor gate statistics during training. If gates collapse, add entropy regularization.

## Troubleshooting

### "No CUDA GPU detected"

Project policy requires GPU training. Solutions:
- Run on GPU-enabled machine
- Use cloud GPU (Colab, AWS, etc.)
- For testing only, comment out GPU check in `experiment_negation_scope.py` (line 33)

### Poor Performance on All Conditions

- Increase training samples (try 50k-100k)
- Increase epochs (try 50+)
- Adjust learning rate (try 5e-5 to 5e-4)
- Check dataset statistics (ensure ~50% negation)

### Probe Ratio < 2.0

This suggests inhibition attention isn't learning negation-specific patterns:
- Visualize attention to see what it's learning
- Try increasing `alpha_sub` (0.1 → 0.3)
- Try more inhib_heads (4 → 8)
- Check if negation examples are sufficiently represented in training data

## Experimental Ladder

```
Phase 1: A vs B
├── A = Baseline transformer
├── B = Subtractive block (separate inhibition attention)
└── Test on negation scope task

    If B > A with probe ratio > 2.5x:
      → Hypothesis SUPPORTED
      → Proceed to Phase 2

    If B ≈ A or probe ratio < 2.0:
      → Hypothesis NOT SUPPORTED
      → Diagnose: visualize attention, check gates

Phase 2: B vs C
├── B = Subtractive only
├── C = Full 4-Op
└── Test whether MUL/DIV add value beyond SUB alone

    If C ≈ B:
      → SUB is sufficient for this task

    If C > B:
      → Full operations provide additional benefit
      → Investigate what MUL/DIV contribute
```

## Next Steps

1. **Run the experiment:**
   ```bash
   python experiment_negation_scope.py --conditions A B
   ```

2. **Analyze results:**
   - Compare loss/accuracy
   - Check probe ratio
   - Visualize attention patterns

3. **If successful (B > A, probe > 2.5x):**
   - Write up findings
   - Test on harder negation tasks (nested scope, double negation)
   - Scale to real language data

4. **If unsuccessful:**
   - Diagnose attention patterns
   - Try architectural tweaks
   - Test on simpler tasks first

## Citation

Based on assessment of TKS 4-Operation Block V2 design improvements addressing:
- Shared context issue → Separate attention mechanisms
- Softmax collapse → Independent gates
- Interpretability → Attention visualization
- Stability → Bounded operations

## Questions?

See `tks_4op_block_v2.py` for implementation details.
Run with `--help` flag for all command-line options.
