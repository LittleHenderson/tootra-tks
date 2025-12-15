@echo off
REM TKS Training Pipeline - Example Workflow (Windows)
REM This script demonstrates a complete training workflow from data generation to model training

setlocal enabledelayedexpansion

echo ======================================================================
echo TKS Training Pipeline - Example Workflow
echo ======================================================================
echo.

REM Configuration
set INPUT_DATA=data\pilot\stories.jsonl
set OUTPUT_DIR=output\example_training
set AUGMENTED_DATA=%OUTPUT_DIR%\augmented_corpus.jsonl

REM Step 1: Generate augmented data (if not already generated)
echo Step 1: Generating augmented data...
if not exist "%AUGMENTED_DATA%" (
    python scripts\generate_augmented_data.py ^
        --input "%INPUT_DATA%" ^
        --output "%AUGMENTED_DATA%" ^
        --axes W N ^
        --use-anti-attractor ^
        --validate ^
        --verbose
    echo   [OK] Augmented data generated: %AUGMENTED_DATA%
) else (
    echo   [OK] Using existing augmented data: %AUGMENTED_DATA%
)
echo.

REM Step 2: Run smoke test to validate data pipeline
echo Step 2: Running smoke test...
python scripts\train_with_augmented.py ^
    --data "%AUGMENTED_DATA%" ^
    --test
echo.

REM Step 3: Dry-run to validate training pipeline
echo Step 3: Running dry-run (single batch)...
python scripts\train_with_augmented.py ^
    --data "%AUGMENTED_DATA%" ^
    --dry-run ^
    --batch-size 8 ^
    --output-dir "%OUTPUT_DIR%\dry_run"
echo.

REM Step 4: Full training run (small scale)
echo Step 4: Running full training (5 epochs)...
python scripts\train_with_augmented.py ^
    --data "%AUGMENTED_DATA%" ^
    --epochs 5 ^
    --batch-size 32 ^
    --learning-rate 1e-4 ^
    --filter-validated ^
    --log-interval 10 ^
    --seed 42 ^
    --output-dir "%OUTPUT_DIR%\training_run_1"
echo.

REM Step 5: Advanced training with metadata
echo Step 5: Running advanced training with metadata...
python scripts\train_with_augmented.py ^
    --data "%AUGMENTED_DATA%" ^
    --epochs 3 ^
    --batch-size 16 ^
    --learning-rate 2e-4 ^
    --include-metadata ^
    --use-expr ^
    --log-interval 5 ^
    --seed 123 ^
    --output-dir "%OUTPUT_DIR%\training_run_2"
echo.

REM Step 6: Early stopping experiment
echo Step 6: Running early stopping experiment...
python scripts\train_with_augmented.py ^
    --data "%AUGMENTED_DATA%" ^
    --epochs 10 ^
    --batch-size 32 ^
    --max-steps 50 ^
    --log-interval 5 ^
    --output-dir "%OUTPUT_DIR%\early_stop"
echo.

echo ======================================================================
echo WORKFLOW COMPLETE
echo ======================================================================
echo.
echo Output files:
echo   - Augmented data:    %AUGMENTED_DATA%
echo   - Training run 1:    %OUTPUT_DIR%\training_run_1\metrics\
echo   - Training run 2:    %OUTPUT_DIR%\training_run_2\metrics\
echo   - Early stop run:    %OUTPUT_DIR%\early_stop\metrics\
echo.
echo Next steps:
echo   1. Review metrics JSON files for training statistics
echo   2. Plot loss curves using CSV files
echo   3. Analyze augmentation distribution and validation rates
echo   4. Replace DummyTKSModel with production model
echo ======================================================================

pause
