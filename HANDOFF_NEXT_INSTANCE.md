# TKS LLM HANDOFF - NEXT INSTANCE MUST READ

**CRITICAL: READ THIS ENTIRE FILE BEFORE DOING ANYTHING**

**Last Updated:** 2026-01-01 20:15 UTC
**Last Agent:** Claude Opus 4.5 (claude-opus-4-5-20251101)
**Session Type:** Multi-Agent Parallel Training (300k Dataset with v4 Model)

---

## CURRENT TRAINING STATUS

### Training In Progress
- **Model:** TKSNoeticLM v4 (105,267 parameters)
- **Dataset:** 300,140 examples (`data/combined_all_v5.jsonl`)
- **Epochs:** 10 (with early stopping patience=3)
- **Batch size:** 64
- **GPU:** NVIDIA GeForce RTX 4070 (8GB)
- **Mixed Precision:** Disabled (FP32) - FP16 has overflow issues in v4

### Loss Progress
| Epoch | Step | Loss |
|-------|------|------|
| 1 | 0 | 4.90 |
| 1 | 1000 | 3.78 |
| 2 | 2000 | 1.47 |
| 2 | 3000 | 1.00 |

**Expected completion:** ~15-20 minutes from start

---

## CRITICAL FIXES APPLIED THIS SESSION

### 1. Model Architecture Clarification
- **Issue:** `train_cuda.py` was using `TKSLLMCorePipeline` (92k params) NOT `TKSNoeticLM` v4
- **Fix:** Added `--model-type noetic_v4` option (now default)
- **Command updated:** Uses `--model-type noetic_v4` by default

### 2. FP16 Overflow Fix
- **Issue:** `-1e9` masking value overflows FP16
- **Fix:** Use `--no-amp` flag to disable mixed precision for v4 model
- **Future fix:** Update v4 model to use `-65504` for FP16 compatibility

### 3. Forward Signature Mismatch
- **Issue:** v4 model doesn't support `return_tensor_deltas`
- **Fix:** Added model type detection in training loop

---

## ARCHITECTURE: v4 Model vs Symbolic Compiler

| Layer | Purpose | File |
|-------|---------|------|
| **TKSNoeticLM v4** | Neural NL↔equation translation | `tks_llm_core_v4.py` |
| **tks-rs** | Symbolic DSL compiler, OOP, VM | `tks-rs/` |

### Neural v4 Model Features:
- NoeticTokenEmbedding with world-wise transforms
- NoeticBlocks with NoeticRouter and CausalFractalAttention
- StableAttractorLayer (guaranteed convergence)
- RPMGatingMechanism (D/W/P filtering)
- Optional: World Bridge, Operator Core, NL Retriever

### Symbolic tks-rs Features:
- Full TKS DSL (classes, blueprints, methods)
- Type inference and checking
- IR lowering and bytecode compilation
- VM with OOP support (classes, fields, methods)
- 232/232 tests passing

**Integration Status:** Not yet connected. v4 generates equations, tks-rs could compile them.

---

## IMMEDIATE CONTEXT

1. **CUDA Environment** (USE THIS FOR ALL PYTHON):
   ```
   C:\Users\wakil\downloads\everthing-tootra-tks\.venv-cuda\Scripts\python.exe
   ```
   - PyTorch 2.6.0+cu124
   - GPU: NVIDIA GeForce RTX 4070 (CUDA 12.4, 8GB)

2. **Training Commands:**
   ```bash
   # Train with v4 model (DEFAULT - 105k params)
   "C:\Users\wakil\downloads\everthing-tootra-tks\.venv-cuda\Scripts\python.exe" scripts/train_cuda.py \
     --data data/combined_all_v5.jsonl \
     --output-dir output/combined_model_v5_noetic \
     --epochs 10 --batch-size 64 --model-type noetic_v4 --no-amp

   # IMPORTANT: Use --no-amp for v4 model to avoid FP16 overflow
   ```

3. **Data Files:**
   | File | Examples |
   |------|----------|
   | `data/combined_all_v5.jsonl` | 300,140 |
   | `data/combined_all_v4.jsonl` | 20,140 |
   | `data/generated_training_batch_4.jsonl` | 70,000 |
   | `data/generated_training_batch_5.jsonl` | 70,000 |
   | `data/generated_training_batch_6.jsonl` | 70,000 |
   | `data/generated_training_batch_7.jsonl` | 70,000 |

---

## NEXT STEPS

1. **Wait for training to complete** (if still running)
2. **Evaluate model quality:**
   ```bash
   "C:\Users\wakil\downloads\everthing-tootra-tks\.venv-cuda\Scripts\python.exe" scripts/test_roundtrip.py
   ```
3. **If loss not converging, try:**
   - Increase layers: `--num-layers 6`
   - Enable Operator Core in v4 config
   - Add more training data

---

## CANONICAL RULES (NEVER MODIFY)

### Worlds (4 only)
- A = Atziluth (Spiritual)
- B = Briah (Mental)
- C = Yetzirah (Emotional)
- D = Assiah (Physical)

### Noetics (1-10)
- 1=Mind, 2=Positive, 3=Negative, 4=Vibration, 5=Female
- 6=Male, 7=Rhythm, 8=Cause, 9=Effect, 10=Idea
- Involution pairs: 2<->3, 5<->6, 8<->9
- Self-duals: 1, 4, 7, 10

### RPM Mapping (FROZEN)
- Desire: {1, 4, 7}
- Wisdom: {5, 6}
- Power: {8, 9}

---

## TEST STATUS

- **tks-rs Compiler:** 230/230 passed, 6 ignored
- **DWP Canonical:** 17/17
- **Anti-Attractor:** 34/34
- **Semantic Roundtrip:** 5/5

---

## KEY FILES

| File | Purpose |
|------|---------|
| `HANDOFF_NEXT_INSTANCE.md` | THIS FILE - read first |
| `scripts/train_cuda.py` | GPU training (supports v4 and tks_pipeline) |
| `scripts/generate_training_batch.py` | Data generation script |
| `tks_llm_core_v4.py` | TKSNoeticLM v4 (neural model) |
| `tks_llm_core_v2.py` | TKSLLMCorePipeline (legacy) |
| `tks-rs/` | Rust symbolic compiler (DSL, OOP, VM) |
| `.venv-cuda/` | CUDA Python environment |

---

**REMEMBER: Use --model-type noetic_v4 --no-amp for v4 training!**
