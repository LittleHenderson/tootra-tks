param(
    [string]$Configuration = "Release",
    [string]$OutDir = "",
    [switch]$Gpu
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptRoot "..")
$tksRoot = Join-Path $repoRoot "tks-rs"

if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $repoRoot "packaging\\windows"
} elseif (-not [System.IO.Path]::IsPathRooted($OutDir)) {
    $OutDir = Join-Path $repoRoot $OutDir
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDir = Resolve-Path $OutDir

$profileArgs = @()
if ($Configuration -ine "Debug") {
    $profileArgs = @("--release")
}

$featureArgs = @()
if ($Gpu) {
    $featureArgs = @("--features", "gpu")
}

Push-Location $tksRoot
try {
    cargo build -p tks @profileArgs @featureArgs
} finally {
    Pop-Location
}

$buildDir = if ($Configuration -ieq "Debug") {
    Join-Path $tksRoot "target\\debug"
} else {
    Join-Path $tksRoot "target\\release"
}

$exePath = Join-Path $buildDir "tks.exe"
if (-not (Test-Path $exePath)) {
    throw "expected executable not found: $exePath"
}

$dest = Join-Path $OutDir "tks.exe"
Copy-Item -Force $exePath $dest
Write-Host "Staged tks.exe to $OutDir"
