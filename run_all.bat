@echo off
REM TKS Master Run Script

IF "%1"=="" GOTO help

IF "%1"=="setup" (
    echo Setting up environment...
    pip install -r requirements.txt
    copy .env.example .env
    echo Environment setup complete.
    GOTO end
)

IF "%1"=="run_episode" (
    echo Running TKS Episode...
    set PYTHONPATH=%CD%\src
    python src\cli\run_episode.py %2 %3 %4 %5
    GOTO end
)

IF "%1"=="build_canon" (
    echo Building Canon...
    python scripts/canon/build_canon.py
    GOTO end
)

IF "%1"=="test" (
    echo Running Tests...
    pytest tests/
    GOTO end
)

:help
echo Usage: run_all.bat [command]
echo Commands:
echo   setup        - Install dependencies and setup env
echo   run_episode  - Run a single TKS episode
echo   build_canon  - Rebuild canon from source
echo   test         - Run unit tests
GOTO end

:end
