@echo off
echo ============================================
echo TKS Training with Real-Time Leaderboard
echo ============================================
echo.

REM Check if Python is available
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python not found in PATH
    pause
    exit /b 1
)

REM Run training with 10 epochs
python scripts/train_leaderboard.py --epochs 10 --batch-size 16 --lr 3e-4 --output-dir checkpoints/v3_leaderboard

echo.
echo ============================================
echo Training Complete!
echo ============================================
pause
