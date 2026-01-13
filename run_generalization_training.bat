@echo off
REM TKS v5 Generalization Training - 6.4 MILLION SAMPLES (NO HEBREW)
REM 100%% unique, massive diversity

echo ============================================
echo TKS v5 - GENERALIZATION TRAINING (6.4M DATA)
echo ============================================
echo.
echo Data: 6,392,140 samples (100%% UNIQUE, NO HEBREW)
echo.
echo Dataset Breakdown:
echo   - Graded Vocabulary: 1.35M (15 levels: 5th grade to PhD + Street)
echo   - Profession Analogies: 2.49M (25 professions)
echo   - Multilingual: 2.24M (15 languages)
echo   - Enriched/Augmented: 313k
echo.
echo Professions: Doctor, Nurse, Surgeon, Electrician, Plumber,
echo   Mechanic, Carpenter, Welder, HVAC, Musician, Chef, Artist,
echo   DJ, Writer, Programmer, Engineer, Architect, Data Scientist,
echo   Teacher, Coach, Firefighter, Detective, Banker, Lawyer, Sales
echo.
echo Languages: Spanish, French, Portuguese, Italian, German, Dutch,
echo   Swedish, Russian, Polish, Japanese, Chinese, Korean, Hindi,
echo   Swahili, Arabic
echo.
echo Epochs: 1 (6.4M samples = ~1.6M steps per epoch)
echo Goal: GENERALIZE across vocab, professions, and languages
echo.
echo Key Settings:
echo   - Dropout: 0.2
echo   - Weight decay: 0.1
echo   - TKS canon-compliant (NO Hebrew)
echo.
echo ============================================
echo Starting training... (Press Ctrl+C to stop)
echo ============================================
echo.

cd /d C:\Users\wakil\downloads\everthing-tootra-tks

REM Activate CUDA venv
call .venv-cuda\Scripts\activate.bat

REM Run training - 1 epoch with 2.1M samples
REM With 2,128,978 samples @ batch 4 = ~532,244 steps/epoch
python train_v5.py --epochs 1

echo.
echo ============================================
echo TRAINING COMPLETE!
echo ============================================
echo.
echo Check results:
echo   - Final checkpoint: checkpoints/v5_final.pt
echo.
pause
