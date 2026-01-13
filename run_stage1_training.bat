@echo off
REM TKS v5 STAGE 1: Graded Vocabulary Training
REM 1.35M samples - Foundation for generalization

echo ============================================
echo TKS v5 - STAGE 1: GRADED VOCABULARY
echo ============================================
echo.
echo Data: 1,346,685 samples (graded vocab only)
echo File: output/train_graded.jsonl
echo.
echo Vocabulary Levels (15):
echo   - Grade 5-12 (8 levels)
echo   - College Freshman-Senior (4 levels)
echo   - Master's, PhD
echo   - Street/AAVE
echo.
echo Training Config:
echo   - Batch size: 4
echo   - Steps: ~336,671 (1 epoch)
echo   - Est. time: ~2 days
echo.
echo Purpose: Build foundation vocabulary understanding
echo          before adding professions and languages.
echo.
echo ============================================
echo Starting Stage 1 training...
echo ============================================
echo.

cd /d C:\Users\wakil\downloads\everthing-tootra-tks

REM Activate CUDA venv
call .venv-cuda\Scripts\activate.bat

REM Run Stage 1 training
python train_v5.py --epochs 1 --data output/train_graded.jsonl

echo.
echo ============================================
echo STAGE 1 COMPLETE!
echo ============================================
echo.
echo Next: Run Stage 2 (add professions) with:
echo   run_stage2_training.bat
echo.
echo Checkpoint: checkpoints/v5_final.pt
echo.
pause
