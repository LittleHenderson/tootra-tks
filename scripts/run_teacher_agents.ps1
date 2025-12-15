# scripts/run_teacher_agents.ps1
# Fan-out teacher generation: 5 agents per provider (Gemini, Anthropic, OpenAI)
#
# Canon Guardrails:
#   - Worlds: A/B/C/D
#   - Noetics: 1-10 (pairs 2<->3, 5<->6, 8<->9; self-duals 1,4,7,10)
#   - Foundations: 1-7, Sub-foundations: 7x4=28
#   - ALLOWED_OPS: +, -, +T, -T, ->, <-, *T, /T, o (9 total)
#
# Prerequisites:
#   - data/equations.jsonl exists (canonical A/B/C/D equations)
#   - Environment variables set:
#       $env:GEMINI_API_KEY = "your-gemini-key"
#       $env:ANTHROPIC_API_KEY = "your-anthropic-key"
#       $env:OPENAI_API_KEY = "your-openai-key"
#
# Usage:
#   .\scripts\run_teacher_agents.ps1
#   .\scripts\run_teacher_agents.ps1 -Chunks 10 -Providers "gemini:gemini-1.5-pro"

param(
    [int]$Chunks = 5,
    [string]$DataFile = "data/equations.jsonl",
    [string[]]$Providers = @(
        "gemini:gemini-1.5-pro",
        "anthropic:claude-3-sonnet-20240229",
        "openai:gpt-4o"
    ),
    [double]$MinCanon = 0.8,
    [switch]$SkipValidation,
    [switch]$SkipAugmentation
)

$ErrorActionPreference = "Stop"

Write-Host "=" * 70
Write-Host "TKS MULTI-PROVIDER TEACHER GENERATION"
Write-Host "=" * 70
Write-Host ""

# Directories
$chunkDir = "output/chunks"
$validDir = "output/teacher_valid"
$combined = "output/teacher_all.jsonl"

# Check prerequisites
if (-not (Test-Path $DataFile)) {
    Write-Error "Input file not found: $DataFile"
    exit 1
}

# Check API keys
$keyWarnings = @()
foreach ($prov in $Providers) {
    $provTag = $prov.Split(':')[0]
    switch ($provTag) {
        "gemini" {
            if (-not $env:GEMINI_API_KEY) {
                $keyWarnings += "GEMINI_API_KEY not set"
            }
        }
        "anthropic" {
            if (-not $env:ANTHROPIC_API_KEY) {
                $keyWarnings += "ANTHROPIC_API_KEY not set"
            }
        }
        "openai" {
            if (-not $env:OPENAI_API_KEY) {
                $keyWarnings += "OPENAI_API_KEY not set"
            }
        }
    }
}

if ($keyWarnings.Count -gt 0) {
    Write-Host "WARNING: Missing API keys:" -ForegroundColor Yellow
    foreach ($warn in $keyWarnings) {
        Write-Host "  - $warn" -ForegroundColor Yellow
    }
    Write-Host ""
    $continue = Read-Host "Continue anyway? (y/N)"
    if ($continue -ne "y") {
        exit 1
    }
}

Write-Host "Configuration:"
Write-Host "  Input file: $DataFile"
Write-Host "  Chunks: $Chunks"
Write-Host "  Providers: $($Providers -join ', ')"
Write-Host "  Min canon score: $MinCanon"
Write-Host ""

# =============================================================================
# Step 1: Split input into chunks
# =============================================================================
Write-Host "Step 1: Splitting input into $Chunks chunks..."
New-Item -ItemType Directory -Force -Path $chunkDir | Out-Null

$lines = Get-Content $DataFile
$totalLines = $lines.Count
$size = [math]::Ceiling($totalLines / $Chunks)

Write-Host "  Total equations: $totalLines"
Write-Host "  Chunk size: ~$size equations each"

for ($i = 0; $i -lt $Chunks; $i++) {
    $chunkPath = Join-Path $chunkDir ("equations_{0}.jsonl" -f $i)
    $startIdx = $i * $size
    $endIdx = [math]::Min(($i + 1) * $size - 1, $totalLines - 1)

    if ($startIdx -le $endIdx) {
        $lines[$startIdx..$endIdx] | Set-Content $chunkPath
        $count = $endIdx - $startIdx + 1
        Write-Host "  Created $chunkPath ($count equations)"
    }
}
Write-Host ""

# =============================================================================
# Step 2: Run teacher generation per chunk per provider
# =============================================================================
Write-Host "Step 2: Running teacher generation (${Chunks} agents x $($Providers.Count) providers)..."
Write-Host ""

$totalJobs = $Chunks * $Providers.Count
$currentJob = 0

foreach ($prov in $Providers) {
    $provTag = $prov.Split(':')[0]
    Write-Host "  Provider: $prov" -ForegroundColor Cyan

    for ($i = 0; $i -lt $Chunks; $i++) {
        $currentJob++
        $chunkPath = Join-Path $chunkDir ("equations_{0}.jsonl" -f $i)
        $outPath = "output/teacher_${provTag}_${i}.jsonl"

        if (-not (Test-Path $chunkPath)) {
            Write-Host "    [$currentJob/$totalJobs] Skipping chunk $i (empty)"
            continue
        }

        Write-Host "    [$currentJob/$totalJobs] $chunkPath -> $outPath"

        try {
            python scripts/run_teacher.py generate $chunkPath `
                --output $outPath `
                --providers $prov `
                --min-canon $MinCanon --strict

            if (Test-Path $outPath) {
                $outLines = (Get-Content $outPath | Measure-Object -Line).Lines
                Write-Host "      Generated $outLines examples" -ForegroundColor Green
            }
        }
        catch {
            Write-Host "      ERROR: $_" -ForegroundColor Red
        }
    }
    Write-Host ""
}

# =============================================================================
# Step 3: Validate each output
# =============================================================================
if (-not $SkipValidation) {
    Write-Host "Step 3: Validating teacher outputs..."
    New-Item -ItemType Directory -Force -Path $validDir | Out-Null

    $teacherFiles = Get-ChildItem "output" -Filter "teacher_*.jsonl" | Where-Object { $_.Name -notmatch "valid|all|augmented" }

    foreach ($file in $teacherFiles) {
        $inPath = $file.FullName
        $outPath = Join-Path $validDir ($file.BaseName + ".valid.jsonl")
        Write-Host "  Validating $($file.Name)..."

        try {
            python scripts/canonical_validator.py --input $inPath --output $outPath

            if (Test-Path $outPath) {
                $validLines = (Get-Content $outPath | Measure-Object -Line).Lines
                Write-Host "    Valid entries: $validLines" -ForegroundColor Green
            }
        }
        catch {
            Write-Host "    ERROR: $_" -ForegroundColor Red
        }
    }
    Write-Host ""
}

# =============================================================================
# Step 4: Concatenate validated outputs
# =============================================================================
Write-Host "Step 4: Combining validated outputs..."

$validFiles = Get-ChildItem $validDir -Filter "*.valid.jsonl" -ErrorAction SilentlyContinue

if ($validFiles.Count -eq 0) {
    # Fall back to unvalidated files
    Write-Host "  No validated files found, using raw teacher outputs..."
    $validFiles = Get-ChildItem "output" -Filter "teacher_*.jsonl" | Where-Object { $_.Name -notmatch "valid|all|augmented" }
}

if ($validFiles.Count -gt 0) {
    $allContent = @()
    foreach ($file in $validFiles) {
        $content = Get-Content $file.FullName
        $allContent += $content
        Write-Host "  Added $($file.Name) ($($content.Count) lines)"
    }

    $allContent | Set-Content $combined
    Write-Host ""
    Write-Host "  Combined output: $combined ($($allContent.Count) total entries)" -ForegroundColor Green
}
else {
    Write-Host "  WARNING: No teacher outputs found!" -ForegroundColor Yellow
}
Write-Host ""

# =============================================================================
# Step 5: Augment (optional)
# =============================================================================
if (-not $SkipAugmentation -and (Test-Path $combined)) {
    Write-Host "Step 5: Augmenting teacher data (inversion + anti-attractor)..."
    $augmented = "output/teacher_augmented.jsonl"

    try {
        python scripts/generate_augmented_data.py `
            --input $combined `
            --output $augmented `
            --use-anti-attractor `
            --validate

        if (Test-Path $augmented) {
            $augLines = (Get-Content $augmented | Measure-Object -Line).Lines
            Write-Host "  Augmented output: $augmented ($augLines entries)" -ForegroundColor Green
        }
    }
    catch {
        Write-Host "  ERROR: $_" -ForegroundColor Red
    }
    Write-Host ""
}

# =============================================================================
# Summary
# =============================================================================
Write-Host "=" * 70
Write-Host "TEACHER GENERATION COMPLETE"
Write-Host "=" * 70
Write-Host ""
Write-Host "Output files:"

if (Test-Path $combined) {
    $combinedLines = (Get-Content $combined | Measure-Object -Line).Lines
    Write-Host "  Combined teacher data: $combined ($combinedLines entries)"
}

$augmented = "output/teacher_augmented.jsonl"
if (Test-Path $augmented) {
    $augLines = (Get-Content $augmented | Measure-Object -Line).Lines
    Write-Host "  Augmented data: $augmented ($augLines entries)"
}

Write-Host ""
Write-Host "Next steps:"
Write-Host "  # Train on augmented data:"
Write-Host "  python scripts/train_with_augmented.py ``"
Write-Host "    --data output/teacher_augmented.jsonl ``"
Write-Host "    --epochs 3 --batch-size 8 --learning-rate 5e-4"
Write-Host ""
