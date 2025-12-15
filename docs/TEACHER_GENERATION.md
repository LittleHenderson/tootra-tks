# Multi-Provider Teacher Generation Guide

This guide explains how to generate TKS training data using multiple LLM providers (Gemini, Anthropic, OpenAI) in a fan-out parallel architecture.

## Overview

The teacher generation pipeline:
1. Splits seed equations into chunks (default: 5)
2. Runs teacher generation per chunk per provider
3. Validates outputs against canonical TKS rules
4. Combines validated outputs
5. Optionally augments with inversion + anti-attractor

## Canon Guardrails

All generated data must conform to:

| Dimension | Values |
|-----------|--------|
| **Worlds** | A, B, C, D (4 total) |
| **Noetics** | 1-10 (pairs: 2↔3, 5↔6, 8↔9; self-duals: 1, 4, 7, 10) |
| **Foundations** | 1-7 |
| **Sub-foundations** | 28 (7 × 4) |
| **Operators** | +, -, +T, -T, ->, <-, *T, /T, o (9 total) |

## Prerequisites

### 1. Seed Data
Ensure `data/equations.jsonl` exists with canonical equations:
```json
{"elements": ["A1"], "equation": "A1"}
{"elements": ["B5", "C6"], "equation": "B5 o C6"}
{"elements": ["A2", "B3"], "equation": "A2 +T B3"}
```

### 2. API Keys
Set environment variables for your providers:

**PowerShell:**
```powershell
$env:GEMINI_API_KEY = "your-gemini-key"
$env:ANTHROPIC_API_KEY = "your-anthropic-key"
$env:OPENAI_API_KEY = "your-openai-key"
```

**CMD:**
```cmd
set GEMINI_API_KEY=your-gemini-key
set ANTHROPIC_API_KEY=your-anthropic-key
set OPENAI_API_KEY=your-openai-key
```

**Bash:**
```bash
export GEMINI_API_KEY="your-gemini-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key"
```

## Usage

### PowerShell (Recommended)

```powershell
# Default: 5 chunks, all 3 providers
.\scripts\run_teacher_agents.ps1

# Custom chunks
.\scripts\run_teacher_agents.ps1 -Chunks 10

# Single provider
.\scripts\run_teacher_agents.ps1 -Providers "gemini:gemini-1.5-pro"

# Skip validation (not recommended)
.\scripts\run_teacher_agents.ps1 -SkipValidation

# Skip augmentation
.\scripts\run_teacher_agents.ps1 -SkipAugmentation

# Custom min canon score
.\scripts\run_teacher_agents.ps1 -MinCanon 0.9
```

### Windows CMD

```cmd
REM Default: 5 chunks
scripts\run_teacher_agents.bat

REM Custom chunks
scripts\run_teacher_agents.bat 10
```

## Output Files

| File | Description |
|------|-------------|
| `output/chunks/equations_N.jsonl` | Split input chunks |
| `output/teacher_gemini_N.jsonl` | Raw Gemini outputs |
| `output/teacher_anthropic_N.jsonl` | Raw Anthropic outputs |
| `output/teacher_openai_N.jsonl` | Raw OpenAI outputs |
| `output/teacher_valid/*.valid.jsonl` | Validated outputs |
| `output/teacher_all.jsonl` | Combined validated data |
| `output/teacher_augmented.jsonl` | Augmented (final training data) |

## Pipeline Steps

### Step 1: Chunk Splitting
Splits `data/equations.jsonl` into N chunks for parallel processing:
```
equations_0.jsonl (3 equations)
equations_1.jsonl (3 equations)
equations_2.jsonl (3 equations)
...
```

### Step 2: Teacher Generation
For each chunk, runs teacher generation per provider:
```
provider:model → chunk → teacher output
gemini:gemini-1.5-pro → equations_0.jsonl → teacher_gemini_0.jsonl
anthropic:claude-3-sonnet → equations_0.jsonl → teacher_anthropic_0.jsonl
openai:gpt-4o → equations_0.jsonl → teacher_openai_0.jsonl
```

### Step 3: Validation
Validates each output against canonical rules:
- Worlds must be A/B/C/D
- Noetics must be 1-10
- Operators must be in ALLOWED_OPS
- Canon score ≥ min threshold (default: 0.8)

### Step 4: Combination
Concatenates all validated outputs into `teacher_all.jsonl`.

### Step 5: Augmentation (Optional)
Applies:
- **Inversion**: World (A↔B, C↔D), Noetic (2↔3, 5↔6, 8↔9)
- **Anti-attractor**: Generates complementary expressions

## Training on Generated Data

After generation, train the model:

```bash
python scripts/train_with_augmented.py \
    --data output/teacher_augmented.jsonl \
    --epochs 3 \
    --batch-size 8 \
    --learning-rate 5e-4
```

Or use quick_train.py for simpler training:

```bash
python scripts/quick_train.py \
    --data output/teacher_augmented.jsonl \
    --epochs 5 \
    --output output/teacher_model
```

## Evaluation

Evaluate the trained model with validator pass-rate:

```bash
python scripts/phase6_eval.py \
    --checkpoint output/teacher_model/final_model.pt \
    --data output/teacher_augmented.jsonl \
    --output output/eval_metrics.json
```

## Troubleshooting

### API Key Errors
- Ensure environment variables are set before running
- Check key validity with provider's API console

### Low Canon Scores
- Review generated expressions for invalid worlds/noetics
- Increase `--min-canon` threshold for stricter filtering
- Check provider prompt templates in `run_teacher.py`

### Empty Outputs
- Verify `data/equations.jsonl` has valid seed data
- Check provider API quotas/limits
- Review error logs in console output

## Provider Models

| Provider | Default Model | Alternatives |
|----------|--------------|--------------|
| Gemini | gemini-1.5-pro | gemini-1.5-flash, gemini-pro |
| Anthropic | claude-3-sonnet-20240229 | claude-3-opus, claude-3-haiku |
| OpenAI | gpt-4o | gpt-4-turbo, gpt-3.5-turbo |

## Parallel Architecture

```
                    ┌─────────────────────────────────────┐
                    │       data/equations.jsonl          │
                    │         (15 seed equations)         │
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
              ▼                       ▼                       ▼
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │   Chunk 0-4     │     │   Chunk 0-4     │     │   Chunk 0-4     │
    │    Gemini       │     │   Anthropic     │     │    OpenAI       │
    └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
             │                       │                       │
             ▼                       ▼                       ▼
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │  teacher_gemini │     │teacher_anthropic│     │  teacher_openai │
    │   _0..4.jsonl   │     │   _0..4.jsonl   │     │   _0..4.jsonl   │
    └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
             │                       │                       │
             └───────────────────────┼───────────────────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │     Validation      │
                          │  (canonical_validator)│
                          └──────────┬──────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │  teacher_all.jsonl  │
                          │  (combined valid)   │
                          └──────────┬──────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │    Augmentation     │
                          │ (invert + anti-attr)│
                          └──────────┬──────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │teacher_augmented.jsonl│
                          │ (final training data) │
                          └─────────────────────┘
```

## See Also

- [DATA_SANITIZER_GUIDE.md](DATA_SANITIZER_GUIDE.md) - Data validation and sanitization
- [INFERENCE_API.md](INFERENCE_API.md) - HTTP API for inference
- [../scripts/run_teacher.py](../scripts/run_teacher.py) - Core teacher generation script
