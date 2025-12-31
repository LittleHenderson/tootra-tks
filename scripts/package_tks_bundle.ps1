param(
    [string]$Configuration = "Release",
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptRoot "..")

if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $repoRoot "packaging\\windows"
} elseif (-not [System.IO.Path]::IsPathRooted($OutDir)) {
    $OutDir = Join-Path $repoRoot $OutDir
}

& (Join-Path $scriptRoot "package_tks.ps1") -Configuration $Configuration -OutDir $OutDir
& (Join-Path $scriptRoot "package_tksc.ps1") -Configuration $Configuration -OutDir $OutDir

Write-Host "Bundle staged to $OutDir"
