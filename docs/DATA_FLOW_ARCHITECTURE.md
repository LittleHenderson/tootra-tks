# TKS v7 Data Flow Architecture

## Complete Step-by-Step: Training to Production

This document explains exactly how data flows through the TKS system at every stage.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Training Flow](#training-flow)
3. [Validation Flow](#validation-flow)
4. [Inference Flow](#inference-flow)
5. [Production Runtime Flow](#production-runtime-flow)
6. [Component Integration Map](#component-integration-map)

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TKS v7 SYSTEM                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  TRAINING                    INFERENCE                              │
│  ─────────                   ─────────                              │
│  Data → Model → Weights      Input → Model → Output                 │
│                                      ↓                              │
│                              Coherence Gate                         │
│                                      ↓                              │
│                              Final Output                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Purpose | File |
|-----------|---------|------|
| TKS v7 Model | Core transformer + DPS + Regulators | `tks_llm_core_v7.py` |
| RTTA-C | Deterministic reasoning (15+ operators) | `tks_features/rtta_canonical.py` |
| RTTA-R | Failure rumination & learning | `tks_features/rtta_rumination.py` |
| Lacunary Flow | Pattern coherence tracking | `tks_features/lacunary_pattern_flow.py` |
| Coherence System | Gibberish detection (99.91% acc) | `tks_features/noetic_fractal_coherence.py` |
| DPS Gating | Depth permission (earned recursion) | `tks_features/dps_gating.py` |
| Episode Runner | Orchestrates all components | `src/tks/runner.py` |

---

## Training Flow

### Step 1: Data Preparation

```
Raw Data Sources
────────────────
data/v6_cot_training.jsonl      (50K examples, 197M) - Chain-of-thought
data/v7_canon_training.jsonl    (10K examples) - Canon patterns
data/v7_discovery_training.jsonl (10K examples) - Discovery patterns
data/v7_diversified_training.jsonl (20K examples) - Diverse examples
data/v7_math_code_training.jsonl (10K examples) - Math/code reasoning
data/coherence_training_nl.jsonl (22K examples) - Coherence examples
data/dps_training_data.jsonl    (3K examples) - DPS gating
data/rtta_training.jsonl        (2K examples) - RTTA reasoning
                                ─────────────────────
                                ↓ Combined into ↓
                                ─────────────────────
data/v7_combined_training.jsonl (171K examples, 268M)
```

### Step 2: Tokenization

```python
# File: train_v7.py, lines 45-70

Input: "Goal: Find wisdom <SEP> Reasoning: CHECK 4 D = 0.9 [MASTERING]"
                    ↓
            Tokenizer (tokenizer_v5.json)
                    ↓
Output: [1024, 58, 2901, 7744, 3, 892, 58, 401, 892, ...]
        (token IDs, max_len=512)
```

### Step 3: Model Forward Pass

```
Input Token IDs [batch, seq_len]
        ↓
┌───────────────────────────────────────┐
│ 1. Token Embedding                    │
│    token_emb.weight [vocab, hidden]   │
│    → [batch, seq, 384]                │
├───────────────────────────────────────┤
│ 2. Position Embedding                 │
│    pos_emb.weight [max_seq, hidden]   │
│    → Added to token embeddings        │
├───────────────────────────────────────┤
│ 3. V6 Transformer Blocks (x12)        │
│    For each block:                    │
│    ├─ Layer Norm                      │
│    ├─ Self-Attention (16 heads)       │
│    ├─ Residual Connection             │
│    ├─ Layer Norm                      │
│    ├─ Feed-Forward (384 → 1536 → 384) │
│    └─ Residual Connection             │
├───────────────────────────────────────┤
│ 4. DPS Gating Layer                   │
│    ├─ Compute novelty score           │
│    ├─ Check depth permission          │
│    └─ Gate hidden states              │
├───────────────────────────────────────┤
│ 5. Regulators                         │
│    ├─ Entropy regulator               │
│    ├─ Confidence regulator            │
│    └─ Apply constraints               │
├───────────────────────────────────────┤
│ 6. LM Head                            │
│    lm_head [hidden, vocab]            │
│    → [batch, seq, 16384] logits       │
└───────────────────────────────────────┘
        ↓
Output Logits [batch, seq, vocab_size]
```

### Step 4: Loss Computation

```python
# File: train_v7.py, lines 220-250

# Standard cross-entropy loss
loss = CrossEntropyLoss(logits, labels, ignore_index=pad_id)

# DPS auxiliary loss (depth prediction)
dps_loss = model.get_dps_loss()

# Combined loss
total_loss = loss + 0.1 * dps_loss
```

### Step 5: Backpropagation & Weight Update

```
total_loss.backward()
        ↓
    Gradients computed for all parameters
        ↓
    optimizer.step() (AdamW, lr=5e-5)
        ↓
    Weights updated
        ↓
    Save checkpoint every N steps
```

### Step 6: Checkpoint Saving

```
checkpoints/
├── v7_step_1000.pt   (periodic)
├── v7_step_2000.pt
├── v7_best.pt        (best validation loss)
└── v7_final.pt       (end of training)
```

---

## Validation Flow

### During Training (Every N Steps)

```
Validation Data (data/v7_discovery_validation.jsonl)
        ↓
┌─────────────────────────────────────┐
│ 1. Load validation batch            │
│ 2. model.eval() (no gradients)      │
│ 3. Forward pass                     │
│ 4. Compute validation loss          │
│ 5. Track metrics:                   │
│    ├─ Perplexity                    │
│    ├─ Accuracy                      │
│    └─ DPS depth distribution        │
│ 6. If best_val_loss → save model    │
└─────────────────────────────────────┘
```

### Validation Metrics

| Metric | What It Measures | Target |
|--------|------------------|--------|
| Loss | Cross-entropy prediction error | Lower is better |
| Perplexity | exp(loss), model confidence | < 50 for good models |
| Accuracy | Token prediction accuracy | > 80% |
| DPS Depth | Average earned depth | 2-4 for complex tasks |

---

## Inference Flow

### Step 1: Input Processing

```
User Input: "What is 2 + 2?"
        ↓
┌─────────────────────────────────────┐
│ Input Classification                │
│ ├─ Is this structured reasoning?    │
│ │   → Route to RTTA-C               │
│ ├─ Is this TKS symbolic?            │
│ │   → Route to v7 Model             │
│ └─ Mixed/unclear?                   │
│     → Try RTTA-C first, fallback v7 │
└─────────────────────────────────────┘
```

### Step 2a: RTTA-C Path (Deterministic Reasoning)

```
Question: "What is 2 + 2?"
        ↓
┌─────────────────────────────────────┐
│ RTTA-C Canonical Solver             │
│ (tks_features/rtta_canonical.py)    │
├─────────────────────────────────────┤
│ 1. Detect problem type              │
│    → arithmetic                     │
│                                     │
│ 2. Check 12-slot prerequisites      │
│    Slot 1D: Mental Desire ✓         │
│    Slot 2W: Mental Wisdom ✓         │
│    Slot 3P: Mental Power ✓          │
│    ...                              │
│                                     │
│ 3. Apply operator                   │
│    _solve_arithmetic("2 + 2")       │
│                                     │
│ 4. Return structured result         │
│    {                                │
│      "answer": "4",                 │
│      "confidence": 1.0,             │
│      "operator": "arithmetic",      │
│      "iterations": 1                │
│    }                                │
└─────────────────────────────────────┘
        ↓
Answer: "4" (deterministic, 100% confidence)
```

### Step 2b: v7 Model Path (Neural Generation)

```
Input: "Goal: Cultivate wisdom <SEP> Reasoning:"
        ↓
┌─────────────────────────────────────┐
│ v7 Model Autoregressive Generation  │
├─────────────────────────────────────┤
│ For each token:                     │
│ 1. Forward pass → logits            │
│ 2. DPS check depth permission       │
│ 3. Sample next token (top-k=40)     │
│ 4. Append to sequence               │
│ 5. Check stop conditions:           │
│    ├─ <EOS> token                   │
│    ├─ max_length reached            │
│    └─ coherence dropped too low     │
└─────────────────────────────────────┘
        ↓
Generated: "CHECK 4 D (Mental Desire) = 0.9 [MASTERING] -> PASS..."
```

### Step 3: Coherence Gating

```
Generated Output
        ↓
┌─────────────────────────────────────┐
│ Coherence Check                     │
│ (noetic_fractal_coherence.py)       │
├─────────────────────────────────────┤
│ 1. Compute lacunarity               │
│    Λ = 1 + Var(μ) / E[μ]²          │
│    → Coherent: Λ ≈ 1.0-1.3         │
│    → Gibberish: Λ > 1.5            │
│                                     │
│ 2. Compute texture tuple            │
│    (dimension, lacunarity, entropy) │
│                                     │
│ 3. Run trained classifier           │
│    → 99.91% accuracy                │
│                                     │
│ 4. Decision:                        │
│    score > 0.5 → PASS               │
│    score < 0.5 → REJECT/RETRY       │
└─────────────────────────────────────┘
        ↓
Coherent Output (or retry)
```

### Step 4: Final Output

```
┌─────────────────────────────────────┐
│ Output Package                      │
├─────────────────────────────────────┤
│ {                                   │
│   "answer": "...",                  │
│   "source": "rtta-c" | "v7-model",  │
│   "confidence": 0.95,               │
│   "coherence_score": 0.87,          │
│   "lacunarity": 1.12,               │
│   "nf_coords": "3:7:2",             │
│   "dps_depth": 3                    │
│ }                                   │
└─────────────────────────────────────┘
```

---

## Production Runtime Flow

### Full Episode Flow (EpisodeRunner)

```
User Request
        ↓
┌───────────────────────────────────────────────────────────────┐
│ EPISODE RUNNER (src/tks/runner.py)                            │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│ 1. WAKE FROM RUMINATION                                       │
│    └─ RTTA-R stops background processing                      │
│                                                               │
│ 2. START LACUNARY TRACE                                       │
│    └─ Begin pattern tracking                                  │
│                                                               │
│ 3. GOVERNANCE PRE-CHECK                                       │
│    ├─ Compute High-Stakes score: HS = (U × K) × A²           │
│    ├─ U = uncertainty, K = stakes, A = alignment              │
│    └─ Determine mode: NORMAL | HIGH_STAKES | CRITICAL         │
│                                                               │
│ 4. GENERATE RPM PLAN                                          │
│    └─ Break goal into prerequisite steps                      │
│                                                               │
│ 5. ROUTE TO TOOLS                                             │
│    ├─ RTTA-C for reasoning questions                          │
│    ├─ v7 Model for TKS generation                             │
│    └─ External tools if needed                                │
│                                                               │
│ 6. EXECUTE                                                    │
│    └─ Run selected tool/model                                 │
│                                                               │
│ 7. VERIFY                                                     │
│    └─ Check output validity                                   │
│                                                               │
│ 8. COHERENCE GATE ← NEW                                       │
│    ├─ Compute coherence score                                 │
│    ├─ Compute lacunarity                                      │
│    ├─ Get NF coordinates                                      │
│    └─ Gate/warn if incoherent                                 │
│                                                               │
│ 9. FINALIZE LACUNARY TRACE                                    │
│    └─ Store pattern statistics                                │
│                                                               │
│ 10. WRITE TO MEMORY                                           │
│     └─ Store successful results                               │
│                                                               │
│ 11. COMPUTE REWARD                                            │
│     └─ Score for RL training                                  │
│                                                               │
│ 12. LOG FAILURES TO RUMINATION                                │
│     └─ RTTA-R queues failures for background analysis         │
│                                                               │
└───────────────────────────────────────────────────────────────┘
        ↓
Episode Result
├─ success: bool
├─ output: Any
├─ coherence_score: float
├─ lacunarity: float
├─ nf_coords: "X:Y:Z"
├─ pattern_category: str
└─ violations: List[str]
```

### Background Rumination (Idle Time)

```
When system is idle (no active requests):
        ↓
┌─────────────────────────────────────┐
│ RTTA-R RUMINATION                   │
│ (tks_features/rtta_rumination.py)   │
├─────────────────────────────────────┤
│ 1. Load failure ledger              │
│    └─ Past questions that failed    │
│                                     │
│ 2. Sort by desire level             │
│    └─ Higher desire = more retries  │
│                                     │
│ 3. Attempt resolution               │
│    ├─ Try different operators       │
│    ├─ Decompose problem             │
│    └─ Learn new patterns            │
│                                     │
│ 4. Update ledger                    │
│    ├─ Mark solved                   │
│    ├─ Increase desire if stuck      │
│    └─ Retire if hopeless            │
│                                     │
│ 5. Wake on new request              │
│    └─ Pause rumination              │
└─────────────────────────────────────┘
```

---

## Component Integration Map

### What Fires When

| Stage | Components Active |
|-------|-------------------|
| Training | v7 Model, DPS Gating, Tokenizer |
| Validation | v7 Model, DPS Gating, Metrics |
| Inference (reasoning) | RTTA-C, Coherence Check |
| Inference (generation) | v7 Model, DPS, Coherence Check |
| Production Episode | ALL: Runner + RTTA-C + v7 + Lacunary + Coherence + DPS + Governance + RTTA-R |

### Data Flow Summary

```
TRAINING:
  JSONL → Tokenizer → Model → Loss → Backprop → Weights → Checkpoint

INFERENCE:
  Input → Classify → [RTTA-C | v7 Model] → Coherence Gate → Output

PRODUCTION:
  Request → Runner → [Governance → Route → Execute → Verify → Coherence → Memory] → Response
                                                    ↑
                                            Rumination (background)
```

---

## File Reference

| File | Role in Data Flow |
|------|-------------------|
| `train_v7.py` | Training loop, data loading, weight updates |
| `tks_llm_core_v7.py` | Model architecture, forward pass |
| `tks_llm_core_v6.py` | Base transformer blocks |
| `tks_features/dps_gating.py` | Depth permission during forward pass |
| `tks_features/rtta_canonical.py` | Deterministic reasoning at inference |
| `tks_features/rtta_rumination.py` | Background failure analysis |
| `tks_features/lacunary_pattern_flow.py` | Pattern tracking during episodes |
| `tks_features/noetic_fractal_coherence.py` | Output coherence checking |
| `tks_features/coherent_tks_wrapper.py` | Wrapper for coherence-gated generation |
| `src/tks/runner.py` | Production episode orchestration |
| `tokenizer_v5.json` | Token vocabulary (16384 tokens) |

---

## Quick Commands

```bash
# Training (uses combined data by default)
py -3.12 train_v7.py --epochs 10 --batch_size 8 --lr 5e-5

# Test integrated system
py -3.12 scripts/test_integrated_runner.py

# Test coherence system
py -3.12 scripts/test_lacunary_coherence.py --integrated

# Test RTTA-C reasoning
py -3.12 -c "from tks_features.rtta_canonical import RTTACanonical; r=RTTACanonical(); print(r.reason('What is 2+2?'))"
```

---

*Document Version: 1.0*
*Last Updated: 2026-01-13*
*Author: TKS Research Pipeline + Claude Opus 4.5*
