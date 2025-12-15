@echo off
REM scripts/run_teacher_agents.bat
REM Fan-out teacher generation: 5 agents per provider (Gemini, Anthropic, OpenAI)
REM
REM Canon Guardrails:
REM   - Worlds: A/B/C/D
REM   - Noetics: 1-10 (pairs 2<->3, 5<->6, 8<->9; self-duals 1,4,7,10)
REM   - Foundations: 1-7, Sub-foundations: 7x4=28
REM   - ALLOWED_OPS: +, -, +T, -T, ->, <-, *T, /T, o (9 total)
REM
REM Prerequisites:
REM   - data/equations.jsonl exists (canonical A/B/C/D equations)
REM   - Environment variables set:
REM       set GEMINI_API_KEY=your-gemini-key
REM       set ANTHROPIC_API_KEY=your-anthropic-key
REM       set OPENAI_API_KEY=your-openai-key
REM
REM Usage:
REM   scripts\run_teacher_agents.bat
REM   scripts\run_teacher_agents.bat 10

setlocal enabledelayedexpansion

REM Configuration
set CHUNKS=%1
if "%CHUNKS%"=="" set CHUNKS=5
set DATA_FILE=data\equations.jsonl
set MIN_CANON=0.8

echo ======================================================================
echo TKS MULTI-PROVIDER TEACHER GENERATION
echo ======================================================================
echo.

REM Directories
set CHUNK_DIR=output\chunks
set VALID_DIR=output\teacher_valid
set COMBINED=output\teacher_all.jsonl

REM Check prerequisites
if not exist "%DATA_FILE%" (
    echo ERROR: Input file not found: %DATA_FILE%
    exit /b 1
)

REM Check API keys (warnings only)
echo Checking API keys...
if "%GEMINI_API_KEY%"=="" echo   WARNING: GEMINI_API_KEY not set
if "%ANTHROPIC_API_KEY%"=="" echo   WARNING: ANTHROPIC_API_KEY not set
if "%OPENAI_API_KEY%"=="" echo   WARNING: OPENAI_API_KEY not set
echo.

echo Configuration:
echo   Input file: %DATA_FILE%
echo   Chunks: %CHUNKS%
echo   Min canon score: %MIN_CANON%
echo.

REM =============================================================================
REM Step 1: Split input into chunks
REM =============================================================================
echo Step 1: Splitting input into %CHUNKS% chunks...
if not exist "%CHUNK_DIR%" mkdir "%CHUNK_DIR%"

REM Count total lines
set /a TOTAL_LINES=0
for /f %%a in ('type "%DATA_FILE%" ^| find /c /v ""') do set TOTAL_LINES=%%a
echo   Total equations: %TOTAL_LINES%

REM Calculate chunk size
set /a CHUNK_SIZE=(%TOTAL_LINES% + %CHUNKS% - 1) / %CHUNKS%
echo   Chunk size: ~%CHUNK_SIZE% equations each

REM Split using Python helper
python -c "
import sys
lines = open('%DATA_FILE%', 'r').readlines()
chunk_size = %CHUNK_SIZE%
chunks = %CHUNKS%
for i in range(chunks):
    start = i * chunk_size
    end = min((i + 1) * chunk_size, len(lines))
    if start < len(lines):
        with open(f'%CHUNK_DIR%\\equations_{i}.jsonl', 'w') as f:
            f.writelines(lines[start:end])
        print(f'  Created %CHUNK_DIR%\\equations_{i}.jsonl ({end - start} equations)')
"
echo.

REM =============================================================================
REM Step 2: Run teacher generation per chunk per provider
REM =============================================================================
echo Step 2: Running teacher generation...
echo.

set PROVIDERS=gemini:gemini-1.5-pro anthropic:claude-3-sonnet-20240229 openai:gpt-4o

for %%P in (%PROVIDERS%) do (
    echo   Provider: %%P

    for /L %%I in (0, 1, 4) do (
        set CHUNK_PATH=%CHUNK_DIR%\equations_%%I.jsonl

        REM Extract provider tag (first part before colon)
        for /f "tokens=1 delims=:" %%T in ("%%P") do set PROV_TAG=%%T

        set OUT_PATH=output\teacher_!PROV_TAG!_%%I.jsonl

        if exist "!CHUNK_PATH!" (
            echo     Processing chunk %%I...
            python scripts\run_teacher.py generate "!CHUNK_PATH!" --output "!OUT_PATH!" --providers %%P --min-canon %MIN_CANON% --strict

            if exist "!OUT_PATH!" (
                for /f %%L in ('type "!OUT_PATH!" ^| find /c /v ""') do echo       Generated %%L examples
            )
        )
    )
    echo.
)

REM =============================================================================
REM Step 3: Validate each output
REM =============================================================================
echo Step 3: Validating teacher outputs...
if not exist "%VALID_DIR%" mkdir "%VALID_DIR%"

for %%F in (output\teacher_*.jsonl) do (
    echo %%~nxF | findstr /v "valid all augmented" >nul
    if not errorlevel 1 (
        echo   Validating %%~nxF...
        python scripts\canonical_validator.py --input "%%F" --output "%VALID_DIR%\%%~nF.valid.jsonl"
    )
)
echo.

REM =============================================================================
REM Step 4: Concatenate validated outputs
REM =============================================================================
echo Step 4: Combining validated outputs...

REM Delete old combined file if exists
if exist "%COMBINED%" del "%COMBINED%"

REM Concatenate all valid files
set /a COMBINED_COUNT=0
for %%F in (%VALID_DIR%\*.valid.jsonl) do (
    type "%%F" >> "%COMBINED%"
    for /f %%L in ('type "%%F" ^| find /c /v ""') do (
        echo   Added %%~nxF (%%L lines^)
        set /a COMBINED_COUNT+=%%L
    )
)

if exist "%COMBINED%" (
    echo.
    echo   Combined output: %COMBINED% (!COMBINED_COUNT! total entries^)
) else (
    echo   WARNING: No teacher outputs found!
)
echo.

REM =============================================================================
REM Step 5: Augment (optional)
REM =============================================================================
echo Step 5: Augmenting teacher data (inversion + anti-attractor^)...
set AUGMENTED=output\teacher_augmented.jsonl

if exist "%COMBINED%" (
    python scripts\generate_augmented_data.py --input "%COMBINED%" --output "%AUGMENTED%" --use-anti-attractor --validate

    if exist "%AUGMENTED%" (
        for /f %%L in ('type "%AUGMENTED%" ^| find /c /v ""') do echo   Augmented output: %AUGMENTED% (%%L entries^)
    )
)
echo.

REM =============================================================================
REM Summary
REM =============================================================================
echo ======================================================================
echo TEACHER GENERATION COMPLETE
echo ======================================================================
echo.
echo Output files:

if exist "%COMBINED%" (
    for /f %%L in ('type "%COMBINED%" ^| find /c /v ""') do echo   Combined teacher data: %COMBINED% (%%L entries^)
)

if exist "%AUGMENTED%" (
    for /f %%L in ('type "%AUGMENTED%" ^| find /c /v ""') do echo   Augmented data: %AUGMENTED% (%%L entries^)
)

echo.
echo Next steps:
echo   # Train on augmented data:
echo   python scripts/train_with_augmented.py ^
echo     --data output/teacher_augmented.jsonl ^
echo     --epochs 3 --batch-size 8 --learning-rate 5e-4
echo.

endlocal
