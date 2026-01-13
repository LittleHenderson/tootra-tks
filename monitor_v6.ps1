# TKS v6 Reasoning Monitor - Real-time Dashboard
# Tracks: Loss, Recursion Depth, Halting Probability
# Updates every 2 seconds

$OutputEncoding = [Console]::OutputEncoding = [Text.Encoding]::UTF8
$logFile = "training_log.txt"

$script:lossHistory = @()
$script:depthHistory = @()
$script:haltHistory = @()
$maxHistory = 50

function Draw-Graph {
    param([string]$label, [array]$data, [double]$minVal, [double]$maxVal, [string]$color)
    if ($data.Count -eq 0) { return }
    
    $width = 50
    $blocks = @(' ', [char]0x2581, [char]0x2582, [char]0x2583, [char]0x2584, [char]0x2585, [char]0x2586, [char]0x2587, [char]0x2588)
    
    $range = $maxVal - $minVal
    if ($range -eq 0) { $range = 1 }
    
    $graphLine = ""
    $startIdx = [Math]::Max(0, $data.Count - $width)
    
    for ($i = $startIdx; $i -lt $data.Count; $i++) {
        $val = $data[$i]
        $normalized = [Math]::Min(8, [Math]::Max(0, [int]((($val - $minVal) / $range) * 8)))
        $graphLine += $blocks[$normalized]
    }
    
    $graphLine = $graphLine.PadLeft($width, " ")
    
    $current = if ($data.Count -gt 0) { "{0:F4}" -f $data[-1] } else { "N/A" }
    
    Write-Host "  $($label.PadRight(10))" -NoNewline -ForegroundColor $color
    Write-Host "$graphLine" -ForegroundColor $color
    Write-Host "             Now: $current" -ForegroundColor DarkGray
    Write-Host ""
}

Clear-Host
$lastStep = -1

while ($true) {
    if (Test-Path $logFile) {
        $lines = Get-Content $logFile -Tail 100
        
        foreach ($line in $lines) {
            # Epoch 0.00 | Step 0: Loss=12.456, Depth=2, Halt=0.15, LR=5.00e-05
            if ($line -match 'Epoch\s+([\d.]+)\s*\|\s*Step\s+(\d+):\s*Loss=([\d.]+),\s*Depth=(\d+),\s*Halt=([\d.]+),\s*LR=([\d.e+-]+)') {
                $step = [int]$matches[2]
                if ($step -gt $lastStep) {
                    $lastStep = $step
                    $script:lossHistory += [double]$matches[3]
                    $script:depthHistory += [double]$matches[4]
                    $script:haltHistory += [double]$matches[5]
                    
                    if ($script:lossHistory.Count -gt $maxHistory) { $script:lossHistory = $script:lossHistory[-$maxHistory..-1] }
                    if ($script:depthHistory.Count -gt $maxHistory) { $script:depthHistory = $script:depthHistory[-$maxHistory..-1] }
                    if ($script:haltHistory.Count -gt $maxHistory) { $script:haltHistory = $script:haltHistory[-$maxHistory..-1] }
                }
            }
        }
        
        $lastLine = $lines | Where-Object { $_ -match 'Epoch.*Step.*Loss' } | Select-Object -Last 1
        Clear-Host
        
        Write-Host "============================================" -ForegroundColor Magenta
        Write-Host "   TKS v6 REASONING ENGINE MONITOR" -ForegroundColor Yellow
        Write-Host "   $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
        Write-Host "============================================" -ForegroundColor Magenta
        Write-Host ""
        
        if ($lastLine -match 'Epoch\s+([\d.]+)\s*\|\s*Step\s+(\d+):\s*Loss=([\d.]+),\s*Depth=(\d+),\s*Halt=([\d.]+),\s*LR=([\d.e+-]+)') {
            $epoch = $matches[1]; $step = $matches[2]; $loss = $matches[3]
            $depth = $matches[4]; $halt = $matches[5]; $lr = $matches[6]
            
            # Progress (Assume ~125000 steps for 10 epochs of 50k data / batch 4)
            $totalSteps = 125000
            $progress = [math]::Round(([int]$step / $totalSteps) * 100, 2)
            $pBar = "[" + ("#" * [math]::Floor($progress/2.5)) + ("-" * (40 - [math]::Floor($progress/2.5))) + "]"
            
            Write-Host "  PROGRESS: $progress%" -ForegroundColor Green
            Write-Host "  $pBar" -ForegroundColor White
            Write-Host "  Step $step / $totalSteps" -ForegroundColor Gray
            Write-Host ""
            
            Write-Host "  +------------+------------------+" -ForegroundColor DarkGray
            Write-Host "  | Epoch      | $($epoch.PadRight(16)) |" -ForegroundColor White
            Write-Host "  | Step       | $($step.ToString().PadRight(16)) |" -ForegroundColor White
            Write-Host "  | Loss       | " -NoNewline; Write-Host "$($loss.PadRight(16))" -NoNewline -ForegroundColor Red; Write-Host " |"
            Write-Host "  | Depth      | " -NoNewline; Write-Host "$($depth.PadRight(16))" -NoNewline -ForegroundColor Cyan; Write-Host " |"
            Write-Host "  | Halt Prob  | " -NoNewline; Write-Host "$($halt.PadRight(16))" -NoNewline -ForegroundColor Yellow; Write-Host " |"
            Write-Host "  | LR         | $($lr.PadRight(16)) |" -ForegroundColor Magenta
            Write-Host "  +------------+------------------+" -ForegroundColor DarkGray
            Write-Host ""
            
            Write-Host "  ==================== GRAPHS ====================" -ForegroundColor Blue
            
            if ($script:lossHistory.Count -gt 0) {
                $lMin = ($script:lossHistory | Measure-Object -Minimum).Minimum
                $lMax = ($script:lossHistory | Measure-Object -Maximum).Maximum
                Draw-Graph "LOSS" $script:lossHistory $lMin $lMax "Red"
                Draw-Graph "RECURSION" $script:depthHistory 0 4 "Cyan"
                Draw-Graph "HALT" $script:haltHistory 0 1 "Yellow"
            }
            
            Write-Host "  ================================================" -ForegroundColor Blue
            Write-Host "  DEPTH: Recursion stack level (1-4 is reasoning)" -ForegroundColor DarkGray
            Write-Host "  HALT: Noetic 9 confidence (Effect reached)" -ForegroundColor DarkGray
        } else {
            Write-Host "Waiting for training to start..." -ForegroundColor Gray
            Get-Content $logFile -Tail 5
        }
    }
    Write-Host ""; Write-Host "  [Refresh: 2s | Ctrl+C to exit]" -ForegroundColor DarkGray
    Start-Sleep -Seconds 2
}
