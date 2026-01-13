@echo off
REM TKS v5 STAGE 3: Add Multilingual Training
REM Resume from Stage 2 checkpoint, train on multilingual data

echo ============================================
echo TKS v5 - STAGE 3: MULTILINGUAL
echo ============================================
echo.
echo Prerequisite: Stage 2 checkpoint must exist
echo Resume from: checkpoints/v5_final.pt
echo.
echo Data: output/train_multilingual.jsonl
echo Samples: ~2.2M multilingual + translation pairs
echo.
echo 15 Languages:
echo   Spanish, French, Portuguese, Italian, German,
echo   Dutch, Swedish, Russian, Polish, Japanese,
echo   Chinese, Korean, Hindi, Swahili, Arabic
echo.
echo Est. time: ~3 days
echo.
echo ============================================
echo Starting Stage 3 training...
echo ============================================
echo.

cd /d C:\Users\wakil\downloads\everthing-tootra-tks

call .venv-cuda\Scripts\activate.bat

REM Check Stage 2 checkpoint exists
if not exist checkpoints\v5_final.pt (
    echo ERROR: Stage 2 checkpoint not found!
    echo Run run_stage2_training.bat first.
    pause
    exit /b 1
)

REM Backup Stage 2 checkpoint
copy checkpoints\v5_final.pt checkpoints\v5_stage2_final.pt

REM Run Stage 3 with resume
python train_v5.py --epochs 1 --data output/train_multilingual.jsonl --resume checkpoints/v5_final.pt

echo.
echo ============================================
echo STAGE 3 COMPLETE - FULL CURRICULUM DONE!
echo ============================================
echo.
echo Checkpoints available:
echo   - checkpoints/v5_stage1_final.pt (graded vocab)
echo   - checkpoints/v5_stage2_final.pt (+ professions)
echo   - checkpoints/v5_final.pt (full: + multilingual)
echo.
echo Run evaluation with: python scripts/full_eval.py
echo.
pause
