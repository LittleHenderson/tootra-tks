@echo off
REM TKS v5 Generalization Training - 12 HOUR RUN
REM Uses 11,320 samples for better generalization

echo ============================================
echo TKS v5 - 12 HOUR GENERALIZATION TRAINING
echo ============================================
echo.
echo Data: 11,320 samples (11x more than before)
echo Epochs: 75 (optimized for 12 hour run)
echo Goal: GENERALIZE, not memorize
echo.
echo Expected Results:
echo   - Loss: 1.5-2.5 (higher is BETTER for generalization)
echo   - NJT Gain: Should vary from 2.0
echo   - Entropy: Should stay above 0.5
echo.
echo Checkpoints saved every 5000 steps to checkpoints/
echo.
echo ============================================
echo Starting training... (Press Ctrl+C to stop)
echo ============================================
echo.

cd /d C:\Users\wakil\downloads\everthing-tootra-tks

REM Activate CUDA venv
call .venv-cuda\Scripts\activate.bat

REM Run training - 75 epochs for 12 hour run
REM With 11,320 samples @ batch 4 = 2,830 steps/epoch
REM 75 epochs = ~212,250 steps total
python train_v5.py --epochs 75

echo.
echo ============================================
echo TRAINING COMPLETE!
echo ============================================
echo.
echo Check results:
echo   - Final checkpoint: checkpoints/v5_final.pt
echo   - Training log: training_log.txt
echo.
pause
