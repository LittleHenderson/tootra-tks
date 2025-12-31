param(
    [string]$Configuration = "Release",
    [string]$OutDir = "",
    [string]$Version = "",
    [switch]$Gpu
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptRoot "..")

if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $repoRoot "dist"
} elseif (-not [System.IO.Path]::IsPathRooted($OutDir)) {
    $OutDir = Join-Path $repoRoot $OutDir
}

if ([string]::IsNullOrWhiteSpace($Version)) {
    $cargoToml = Join-Path $repoRoot "tks-rs"
    $cargoToml = Join-Path $cargoToml "crates"
    $cargoToml = Join-Path $cargoToml "tks"
    $cargoToml = Join-Path $cargoToml "Cargo.toml"
    $versionLine = Get-Content -Path $cargoToml | Where-Object { $_ -match '^\s*version\s*=' } | Select-Object -First 1
    if (-not $versionLine) {
        throw "Unable to read version from $cargoToml"
    }
    if ($versionLine -match '^\s*version\s*=\s*"([^"]+)"') {
        $Version = $Matches[1]
    } else {
        throw "Unable to parse version from $cargoToml"
    }
}

$bundleDir = Join-Path $OutDir "tks-$Version-windows"
New-Item -ItemType Directory -Force -Path $bundleDir | Out-Null

& (Join-Path $scriptRoot "package_tks_bundle.ps1") -Configuration $Configuration -OutDir $bundleDir

$zipPath = Join-Path $OutDir "tks-$Version-windows.zip"
if (Test-Path $zipPath) {
    Remove-Item -Force $zipPath
}
Compress-Archive -Path (Join-Path $bundleDir "*") -DestinationPath $zipPath

Write-Host "Bundle staged to $bundleDir"
Write-Host "Zip written to $zipPath"

if ($Gpu) {
    $tksRoot = Join-Path $repoRoot "tks-rs"
    $profileArgs = @()
    if ($Configuration -ine "Debug") {
        $profileArgs = @("--release")
    }

    Push-Location $tksRoot
    try {
        cargo build -p tks @profileArgs --features gpu
    } finally {
        Pop-Location
    }

    $buildDir = if ($Configuration -ieq "Debug") {
        Join-Path $tksRoot "target\\debug"
    } else {
        Join-Path $tksRoot "target\\release"
    }

    $gpuDir = Join-Path $OutDir "tks-$Version-windows-gpu"
    New-Item -ItemType Directory -Force -Path $gpuDir | Out-Null

    $tksExe = Join-Path $buildDir "tks.exe"
    if (-not (Test-Path $tksExe)) {
        throw "expected GPU executable not found: $tksExe"
    }
    Copy-Item -Force $tksExe (Join-Path $gpuDir "tks.exe")

    $tkscExe = Join-Path $bundleDir "tksc.exe"
    if (-not (Test-Path $tkscExe)) {
        throw "expected tksc.exe not found in bundle: $tkscExe"
    }
    Copy-Item -Force $tkscExe (Join-Path $gpuDir "tksc.exe")

    $gpuZipPath = Join-Path $OutDir "tks-$Version-windows-gpu.zip"
    if (Test-Path $gpuZipPath) {
        Remove-Item -Force $gpuZipPath
    }
    Compress-Archive -Path (Join-Path $gpuDir "*") -DestinationPath $gpuZipPath

    Write-Host "GPU bundle staged to $gpuDir"
    Write-Host "GPU zip written to $gpuZipPath"
}
