@echo off
cd /d "C:\Users\wakil\downloads\everthing-tootra-tks"
.\.venv311\Scripts\python.exe scripts\quick_train.py --data output\teacher_mvr_converted.jsonl --output-dir output\teacher_model_mvr_v1 --epochs 10 --batch-size 4 --learning-rate 0.001
pause
