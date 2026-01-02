# TKS Project Handoff Document
**Last Updated:** 2026-01-01 17:40 UTC
**Last Agent:** Claude Opus 4.5 (claude-opus-4-5-20251101)

---

## MISSION STATEMENT

Build a **canon-faithful TKS reasoning engine** that:
- Translates between natural language and canonical TKS equations (bidirectional)
- Validates every output against TKS ontology (hard constraints, not soft)
- Uses RPM gating (Desire/Wisdom/Power) to filter reasoning trajectories
- Models attractor/anti-attractor dynamics for stable noetic states
- Operates in fixed 40D noetic space with involutions and algebraic constraints
- Provides full auditability (validators, round-trip tests, per-component metrics)

**NOT a generic chatbot** - enforces canonical constraints as hard requirements.

---

## CURRENT STATE (2024-12-22)

### Completed This Session
1. **Pivoted from 600k generation** - Was taking ~26 days, switched to validation-first
2. **Fixed provider bugs:**
   - Gemini model: `gemini-3-pro-preview` (invalid) → `gemini-2.5-pro`
   - Gemini max_tokens: 1024 → 8192 (was causing empty responses)
   - Query timeout: 60s → 180s in `teacher/ensemble.py:49`
3. **Trained validation model:**
   - Data: `output/validation_run/train.jsonl` (2000 noetic fractal examples)
   - Model: `output/validation_run/model/best_model.pt`
   - Eval loss: 1.12 (converged in 10 epochs, 35 seconds)
4. **Audited codebase** - All 5 mission components assessed:

| Component | Status | Key File |
|-----------|--------|----------|
| RPM Gating | ✅ Ready | `tks_rules/rpm.py` |
| Attractor/Anti-Attractor | ✅ Ready | `anti_attractor.py` (1050 lines) |
| Scenario Inversion | ✅ Ready | `inversion/engine.py`, `scenario_inversion.py` |
| Canon Validation | ✅ Ready | `teacher/validator.py` |
| Round-Trip | ⚠️ Partial | `scripts/test_roundtrip.py` (semantic fidelity missing) |

5. **Ran tests:** 52/53 passed (98%)

### In Progress (Agents Launched 2024-12-22 ~19:00 UTC)
1. **Generating equation interpretations** (200 examples via Claude API)
   - Agent ID: a2dbb0f
   - Output: `output/equation_interpretations/`
   - Command: `scripts/run_teacher_agents.py --limit 200 --providers anthropic:claude-opus-4-5`

2. **Implementing semantic round-trip fidelity metrics**
   - Agent ID: a77a8ad
   - Output: `scripts/semantic_roundtrip_metrics.py`
   - Features: embedding similarity, element overlap, fidelity score

---

## KEY FILES & LOCATIONS

### Training Data
```
data/noetic_fractals_complete_teacher.jsonl  # 2000 examples (fractal↔language)
data/noetic_fractals_complete_teacher_holdout.jsonl  # 200 holdout
data/equations_600k.jsonl  # 600k seed equations (218MB)
output/teacher_600k/chunks/  # 80 chunks of 7500 equations each
```

### Models
```
output/validation_run/model/best_model.pt  # Trained on noetic fractals
output/validation_run/model/training_metrics.json
```

### Scripts
```
scripts/train_cuda.py  # GPU training (--model-type tks_pipeline)
scripts/run_teacher_supervisor.py  # Multi-worker teacher generation
scripts/run_teacher_agents.py  # Worker script
scripts/run_teacher_resume.py  # Resume failed chunks
scripts/test_roundtrip.py  # Round-trip validation
```

### Core Components
```
tks_rules/rpm.py  # RPM D/W/P = {1,4,7}/{5,6}/{8,9}
anti_attractor.py  # Attractor dynamics, counter-scenario synthesis
inversion/engine.py  # 42 inversion modes
teacher/validator.py  # CanonicalValidator class
teacher/ensemble.py  # MultiLLMTeacher (timeout=180s now)
teacher/providers.py  # GeminiProvider (max_tokens=8192 now)
```

### API Keys
```
~/.config/tks/keys.env  # Contains ANTHROPIC_API_KEY, GEMINI_API_KEY
```

---

## WHAT NEEDS TO BE DONE

### Immediate (Next Session)
1. **Check agent outputs:**
   - Equation interpretations: `output/equation_interpretations/`
   - Semantic metrics: Check if implemented in `scripts/` or `tests/`

2. **Combine training data:**
   ```bash
   cat data/noetic_fractals_complete_teacher.jsonl \
       output/equation_interpretations/*.jsonl > data/combined_teacher.jsonl
   ```

3. **Train on combined data:**
   ```bash
   python3 scripts/train_cuda.py \
     --data data/combined_teacher.jsonl \
     --output-dir output/combined_model \
     --epochs 20 --batch-size 16 --model-type tks_pipeline
   ```

4. **Run semantic round-trip tests** (if metrics implemented)

### Medium Term
1. Generate more training data (10k-50k examples) if model quality insufficient
2. Implement spectral radius analysis for attractor stability
3. Add partial inversion modes to scenario_inversion.py

### Long Term
1. Scale to 100k+ examples once pipeline validated
2. Calibration pipeline (Brier score/ECE) for probability outputs
3. Hyperstition/Overton window tracking

---

## CANONICAL RULES (DO NOT MODIFY)

### Worlds (4 only)
- A = Atziluth (Spiritual)
- B = Briah (Mental)
- C = Yetzirah (Emotional)
- D = Assiah (Physical)

### Noetics (1-10)
- 1=Mind, 2=Positive, 3=Negative, 4=Vibration, 5=Female
- 6=Male, 7=Rhythm, 8=Cause, 9=Effect, 10=Idea
- Involution pairs: 2↔3, 5↔6, 8↔9
- Self-duals: 1, 4, 7, 10

### RPM Mapping (MVR Protocol - FROZEN)
- Desire: {1, 4, 7} = Mind, Vibration, Rhythm
- Wisdom: {5, 6} = Female, Male
- Power: {8, 9} = Cause, Effect

### Foundations (1-7)
- 1=Association, 2=Wisdom, 3=Life, 4=Companionship
- 5=Power, 6=Wealth, 7=Continuation
- Opposites: 1↔7, 2↔6, 3↔5, 4↔4

### Operators (9 canonical)
```
+, -, +T, -T, *T, /T, ->, <-, o
```

---

## COMMANDS REFERENCE

### Generate Teacher Data (small batch)
```bash
python3 scripts/run_teacher_supervisor.py \
  --data data/equations_600k.jsonl \
  --output-dir output/teacher_small \
  --workers 4 --chunks 10 --limit 100 \
  --providers anthropic:claude-opus-4-5 \
  --keys-from-config ~/.config/tks/keys.env
```

### Train Model
```bash
python3 scripts/train_cuda.py \
  --data <data.jsonl> \
  --output-dir output/models \
  --epochs 10 --batch-size 16 \
  --model-type tks_pipeline \
  --early-stopping 3
```

### Run Tests
```bash
python3 -m pytest tests/test_regression_gate.py tests/test_dwp_canonical.py tests/test_anti_attractor.py -v
```

### Check GPU
```bash
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

---

## AGENT INSTRUCTIONS

**Before starting work:**
1. Read this file (`HANDOFF_CLAUDE_CLI.md`)
2. Check `output/` for any new files from previous agents
3. Run `ps aux | grep python` to check for running processes
4. Review recent git changes: `git status && git log --oneline -5`

**After completing work:**
1. Update this file with what was done
2. Update the "Last Updated" timestamp
3. List any new files created
4. Note any issues or blockers

**Key constraints:**
- DO NOT modify canonical rules (worlds, noetics, RPM, foundations, operators)
- All outputs must pass canon validation
- Use `--model-type tks_pipeline` (Transformer disabled)
- API keys in `~/.config/tks/keys.env`

---

## RECENT CHANGES LOG

### 2024-12-22
- Fixed Gemini provider (model name + max_tokens)
- Fixed query timeout (60→180s)
- Trained validation model on 2000 examples
- Audited all 5 mission components
- Launched agents for: equation interpretations + semantic metrics

### 2026-01-01 (Parallel Agent Session)
- **Combined training data:** Created  (2,640 examples)
  - 2,000 noetic fractals + 640 equation interpretations
  - Task types: fractal_to_language, language_to_fractal, equation_to_interpretation, etc.
- **Semantic round-trip metrics:** FULLY IMPLEMENTED
  -  (825 lines)
  -  (279 lines)
  -  (257 lines)
  - Fidelity formula: 0.6 * semantic_similarity + 0.4 * element_overlap
- **Data quality verified:** All datasets pass canonical rules
  - 600k equations: Valid worlds (A-D), valid noetics (1-10)
  - 2k noetic fractals: Valid fractal notation (0-9)
- **PyTorch reinstalled:** torch 2.9.1 (CPU) - Python 3.14 has no CUDA wheels yet
- **Tests:** 53/53 passed (100%)
- **Ready for training** on combined dataset
