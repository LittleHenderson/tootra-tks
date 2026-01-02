@echo off
cd /d "C:\Users\wakil\downloads\everthing-tootra-tks"
python scripts/train_story_equation.py --data output/teacher_story_eq_train.jsonl --output-dir output/teacher_model_story --epochs 10 --batch-size 8 --learning-rate 1e-3
pause
