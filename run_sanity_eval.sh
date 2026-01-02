#!/bin/bash
# Sanity evaluation script for long_v4 and v3 models on standard holdout

cd /c/Users/wakil/downloads/everthing-tootra-tks

echo "=== Evaluating long_v4 on standard holdout ==="
.venv/Scripts/python.exe scripts/phase6_eval.py \
    --checkpoint output/teacher_model_long_v4/final_model.pt \
    --data output/teacher_rich_v2_holdout.jsonl \
    --output output/teacher_model_long_v4/eval_vs_standard_holdout.json \
    --test-ratio 1.0

echo ""
echo "=== Evaluating v3 on standard holdout ==="
.venv/Scripts/python.exe scripts/phase6_eval.py \
    --checkpoint output/teacher_model_v3/final_model.pt \
    --data output/teacher_rich_v2_holdout.jsonl \
    --output output/teacher_model_v3/eval_vs_standard_holdout.json \
    --test-ratio 1.0

echo ""
echo "=== Done ==="
