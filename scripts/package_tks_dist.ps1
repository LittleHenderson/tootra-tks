param(
    [string]$Configuration = "Release",
    [string]$OutDir = "",
    [string]$Version = ""
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
    $match = Select-String -Path $cargoToml -Pattern '^[\\s]*version\\s*=\\s*\"([^\"]+)\"' | Select-Object -First 1
    if (-not $match) {
        throw "Unable to read version from $cargoToml"
    }
    $Version = $match.Matches[0].Groups[1].Value
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
