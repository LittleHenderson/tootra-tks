@echo off
REM TKS v5 STAGE 1 VALIDATION: Quick 5K step test
REM Verify training works before committing to full run

echo ============================================
echo TKS v5 - STAGE 1 VALIDATION (5K steps)
echo ============================================
echo.
echo Purpose: Verify training pipeline works
echo Data: output/train_graded.jsonl
echo Steps: 5,000 (~40 minutes)
echo.
echo What to check:
echo   - Loss decreasing
echo   - No CUDA errors
echo   - Checkpoints saving
echo   - Memory stable
echo.
echo ============================================
echo Starting validation run...
echo ============================================
echo.

cd /d C:\Users\wakil\downloads\everthing-tootra-tks

call .venv-cuda\Scripts\activate.bat

python train_v5.py --steps 5000 --data output/train_graded.jsonl

echo.
echo ============================================
echo VALIDATION COMPLETE!
echo ============================================
echo.
echo If loss decreased and no errors, run full Stage 1:
echo   run_stage1_training.bat
echo.
pause
