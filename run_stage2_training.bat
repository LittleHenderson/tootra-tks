@echo off
REM TKS v5 STAGE 2: Add Professions Training
REM Resume from Stage 1 checkpoint, train on professions data

echo ============================================
echo TKS v5 - STAGE 2: PROFESSIONS
echo ============================================
echo.
echo Prerequisite: Stage 1 checkpoint must exist
echo Resume from: checkpoints/v5_final.pt
echo.
echo Data: output/train_professions.jsonl
echo Samples: ~2.5M profession analogies/mappings
echo.
echo 25 Professions:
echo   Doctor, Nurse, Surgeon, Electrician, Plumber,
echo   Mechanic, Carpenter, Welder, HVAC Tech, Musician,
echo   Chef, Artist, DJ, Writer, Programmer, Engineer,
echo   Architect, Data Scientist, Teacher, Coach,
echo   Firefighter, Detective, Banker, Lawyer, Salesperson
echo.
echo Est. time: ~4 days
echo.
echo ============================================
echo Starting Stage 2 training...
echo ============================================
echo.

cd /d C:\Users\wakil\downloads\everthing-tootra-tks

call .venv-cuda\Scripts\activate.bat

REM Check Stage 1 checkpoint exists
if not exist checkpoints\v5_final.pt (
    echo ERROR: Stage 1 checkpoint not found!
    echo Run run_stage1_training.bat first.
    pause
    exit /b 1
)

REM Backup Stage 1 checkpoint
copy checkpoints\v5_final.pt checkpoints\v5_stage1_final.pt

REM Run Stage 2 with resume
python train_v5.py --epochs 1 --data output/train_professions.jsonl --resume checkpoints/v5_final.pt

echo.
echo ============================================
echo STAGE 2 COMPLETE!
echo ============================================
echo.
echo Next: Run Stage 3 (add multilingual) with:
echo   run_stage3_training.bat
echo.
pause
