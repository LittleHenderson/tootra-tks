param(
    [string]$Configuration = "Release",
    [string]$OutDir = "",
    [switch]$Gpu
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptRoot "..")
$tksRoot = Join-Path $repoRoot "tks-rs"

function Get-TksVersion {
    param([string]$ManifestPath)
    $line = Get-Content -Path $ManifestPath | Where-Object { $_ -match '^\s*version\s*=\s*"(.*)"' } | Select-Object -First 1
    if (-not $line) {
        throw "version not found in $ManifestPath"
    }
    if ($line -match '^\s*version\s*=\s*"(.*)"') {
        return $Matches[1]
    }
    throw "version not found in $ManifestPath"
}

$zipPath = $null
if ($Gpu) {
    $manifestPath = Join-Path $tksRoot "crates\\tks\\Cargo.toml"
    $version = Get-TksVersion -ManifestPath $manifestPath
    $OutDir = Join-Path $repoRoot ("dist\\tks-$version-windows-gpu")
    $zipPath = Join-Path $repoRoot ("dist\\tks-$version-windows-gpu.zip")
} elseif ([string]::IsNullOrWhiteSpace($OutDir)) {
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

Push-Location $tksRoot
try {
    if ($Gpu) {
        cargo build -p tks @profileArgs --features gpu
        cargo build -p tksc @profileArgs
    } else {
        cargo build -p tks @profileArgs
    }
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

if ($Gpu) {
    $tkscPath = Join-Path $buildDir "tksc.exe"
    if (-not (Test-Path $tkscPath)) {
        throw "expected executable not found: $tkscPath"
    }
    $tkscDest = Join-Path $OutDir "tksc.exe"
    Copy-Item -Force $tkscPath $tkscDest
    Compress-Archive -Path (Join-Path $OutDir "*") -DestinationPath $zipPath -Force
    Write-Host "Staged tksc.exe to $OutDir"
    Write-Host "Created $zipPath"
}
