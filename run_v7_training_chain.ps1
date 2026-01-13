$ErrorActionPreference = "Stop"

$py = ".\.venv-cuda\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "FATAL: CUDA venv python not found at $py"
    exit 1
}

$env:PYTHONUTF8 = "1"

function Run-Step {
    param(
        [string]$Name,
        [string]$Script,
        [string]$LogFile,
        [string]$Epochs
    )
    Write-Host "=== $Name ==="
    $env:TKS_LOG_FILE = $LogFile
    $env:TKS_EPOCHS = $Epochs
    & $py $Script 2>&1 | Tee-Object -FilePath $LogFile
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

Run-Step -Name "v7 discovery" -Script "scripts\\train_v7_discovery.py" -LogFile "training_log_v7_discovery.txt" -Epochs "3"
Run-Step -Name "v7 math_code" -Script "scripts\\train_v7_math_code.py" -LogFile "training_log_v7_math_code.txt" -Epochs "5"
Run-Step -Name "v7 multidomain" -Script "scripts\\train_v7_multidomain.py" -LogFile "training_log_v7_multidomain.txt" -Epochs "1"
Run-Step -Name "v7 inversion" -Script "scripts\\train_v7_inversion.py" -LogFile "training_log_v7_inversion.txt" -Epochs "3"

Write-Host "All v7 stages complete."
