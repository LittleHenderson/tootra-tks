# AI Assistant Instructions

**Read this file first when working in this codebase.**

---

## Project Overview

This is the TKS (Traceable Knowledge System) LLM project - a novel neural architecture with:
- Stable routing to 10 noetic operators
- Attractor computation for reasoning
- DPS (Depth Permission System) for adaptive computation
- Governance rails for safety

---

## Special Formats

### When User Asks for Explanations

If the user asks you to **explain, define, or break down** technical concepts:

**USE THIS FORMAT:** `docs/EXPLAIN_FORMAT_STANDARD.md`

This format includes:
- Three-column table: Metaphor | Technical | 6th Grade
- "Like this:" everyday analogies
- Bullet points with numbers and ranges
- Quick Reference Card at the end

**Trigger phrases to watch for:**
- "explain this"
- "what does ___ mean"
- "define"
- "break down"
- "simple terms"
- "metaphor"
- "6th grade" / "ELI5"
- "help me understand"

---

## Training Rules

### GPU Only - No CPU Training

All training scripts enforce GPU usage. If CUDA is not available:
- **DO NOT** modify scripts to allow CPU
- **DO NOT** create workarounds
- **STOP** and report the issue

### Use Provided Scripts

- `train_v5.py` - Main model training
- `scripts/train_dps_layer.py` - DPS layer training
- `scripts/verify_v5_setup.py` - Setup verification

**DO NOT** create your own training scripts.

---

## Key Files

| File | Purpose |
|------|---------|
| `tks_llm_core_v5.py` | Main v5 model |
| `tks_llm_core_v4.py` | Previous version |
| `configs/v5_recommended.py` | Config factory |
| `tks_features/routing_stability.py` | Stable routing |
| `tks_features/dps_gating.py` | DPS layer |
| `tks_features/governance_rails.py` | Safety gates |
| `GEMINI_TRAINING_INSTRUCTIONS.md` | Training guide |
| `docs/EXPLAIN_FORMAT_STANDARD.md` | Explanation format |

---

## User Preferences

1. **No emojis** unless explicitly requested
2. **Concise responses** - this is CLI output
3. **Verify before modifying** - read files before editing
4. **GPU only** for training - user's CPU overheats

---

## Quick Commands

```bash
# Verify setup
python scripts/verify_v5_setup.py

# Train v5 (GPU required)
python train_v5.py --epochs 150

# Run tests
python -m pytest tests/test_v5_e2e.py -v
```
