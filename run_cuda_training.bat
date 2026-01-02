@echo off
REM ============================================================
REM TKS CUDA Training Script for Windows
REM ============================================================
REM This script runs GPU-accelerated training for TKS models
REM
REM Requirements:
REM   - NVIDIA GPU with CUDA support
REM   - PyTorch with CUDA installed
REM   - Training data in output/teacher_augmented.jsonl
REM ============================================================

echo ============================================================
echo           TKS CUDA-Accelerated Training
echo ============================================================
echo.

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please install Python 3.10+ and add to PATH
    pause
    exit /b 1
)

REM Check CUDA availability
echo Checking CUDA setup...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}' if torch.cuda.is_available() else 'CUDA not available'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else '')"

if errorlevel 1 (
    echo.
    echo ERROR: PyTorch not installed or CUDA check failed
    echo.
    echo To install PyTorch with CUDA support:
    echo   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Starting Training...
echo ============================================================
echo.

REM Default training parameters
set DATA_FILE=output/teacher_augmented.jsonl
set OUTPUT_DIR=output/cuda_models
set EPOCHS=10
set BATCH_SIZE=16
set LEARNING_RATE=3e-4

REM Check if data file exists
if not exist "%DATA_FILE%" (
    echo WARNING: Default data file not found: %DATA_FILE%
    echo Looking for alternative data files...

    if exist "output/teacher_merged_v5.jsonl" (
        set DATA_FILE=output/teacher_merged_v5.jsonl
        echo Found: %DATA_FILE%
    ) else if exist "data/story_eq_pairs.jsonl" (
        set DATA_FILE=data/story_eq_pairs.jsonl
        echo Found: %DATA_FILE%
    ) else (
        echo.
        echo ERROR: No training data found!
        echo Please run the teacher generation pipeline first:
        echo   python scripts/run_teacher.py
        echo.
        pause
        exit /b 1
    )
)

echo Using data file: %DATA_FILE%
echo Output directory: %OUTPUT_DIR%
echo.

REM Run training
python scripts/train_cuda.py ^
    --data "%DATA_FILE%" ^
    --output-dir "%OUTPUT_DIR%" ^
    --epochs %EPOCHS% ^
    --batch-size %BATCH_SIZE% ^
    --learning-rate %LEARNING_RATE% ^
    --use-bf16

if errorlevel 1 (
    echo.
    echo Training failed! Check error messages above.
    echo.
    echo Common issues:
    echo   1. Out of GPU memory - try reducing --batch-size
    echo   2. CUDA error - check NVIDIA driver version
    echo   3. Data format error - check training data file
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Training Complete!
echo ============================================================
echo.
echo Model saved to: %OUTPUT_DIR%
echo.
echo To run inference:
echo   python scripts/serve_inference.py --model %OUTPUT_DIR%/best_model.pt
echo.
pause
