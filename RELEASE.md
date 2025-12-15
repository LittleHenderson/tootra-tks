# TKS v0.2.4 Release Notes

Release Date: 2025-12-14

## Highlights

- Teacher-driven pipeline: 15 equations → 60 validated teacher examples → 3x augmentation (inversion + anti-attractor) → training run.
- Training: real model; loss reduced 4.17 → 3.24 over 2 epochs (smoke-scale run).
- Evaluation: 100% canonical validity on eval set; smoke-level accuracy ~1% as expected.
- New scripts/tools: `scripts/validate_teacher_output.py`, `scripts/quick_train.py`.
- New data/artifacts: `data/equations.jsonl`, `output/teacher_outputs.jsonl`, `output/teacher_augmented.jsonl`, `output/phase5_models/final_model.pt`, training/eval reports.
- Docs: `docs/RECIPES.md` updated (Recipe 4: Teacher→Train Flow); PHASE5 summaries/guides added.
- Canon preserved: ALLOWED_OPS=9; worlds A/B/C/D; noetics/foundations fixed; SUBFOUND_MAP=28; involution pairs N2↔N3, N5↔N6, N8↔N9.

## Getting Started

```bash
# Generate teacher data (example)
python scripts/run_teacher.py generate equations.jsonl --output teacher_outputs.jsonl --providers gemini:gemini-1.5-pro --min-canon 0.8

# Validate teacher outputs
python scripts/validate_teacher_output.py --input teacher_outputs.jsonl --output teacher_outputs_valid.jsonl

# Augment teacher data (inversion + anti-attractor)
python scripts/generate_augmented_data.py --input teacher_outputs_valid.jsonl --output teacher_augmented.jsonl --use-anti-attractor --validate

# Train with augmented teacher data
python scripts/train_with_augmented.py --data teacher_augmented.jsonl --epochs 2 --batch-size <B> --learning-rate <LR> --use-augmented

# Evaluate
python scripts/evaluate_model.py --model output/phase5_models/final_model.pt --data <heldout.jsonl>

# Inference (bulk)
python scripts/run_inference.py --input-jsonl data/stories.jsonl --output-jsonl data/inferred.jsonl

# Run tests
python -m pytest tests -v
```

## Canon Checks

- Operators: `+`, `-`, `+T`, `-T`, `->`, `<-`, `*T`, `/T`, `o`
- Worlds: A/B/C/D; Noetics: 1-10; Foundations: 1-7; SUBFOUND_MAP=28
- Involution pairs: N2↔N3, N5↔N6, N8↔N9; self-duals: N1, N4, N7, N10
