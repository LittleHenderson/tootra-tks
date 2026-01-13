# TKS v5 Training Monitor - Real-time Dashboard with Graphs
# Updates every 2 seconds with key metrics and visual graphs

$OutputEncoding = [Console]::OutputEncoding = [Text.Encoding]::UTF8
$logFile = "C:\Users\wakil\AppData\Local\Temp\claude\C--Users-wakil-downloads-everthing-tootra-tks\tasks\bc19b63.output"

$script:lossHistory = @()
$script:tempHistory = @()
$script:entropyHistory = @()
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
    $minD = if ($data.Count -gt 0) { "{0:F4}" -f ($data | Measure-Object -Minimum).Minimum } else { "N/A" }
    $maxD = if ($data.Count -gt 0) { "{0:F4}" -f ($data | Measure-Object -Maximum).Maximum } else { "N/A" }
    
    Write-Host "  $($label.PadRight(10))" -NoNewline -ForegroundColor $color
    Write-Host "$graphLine" -ForegroundColor $color
    Write-Host "             Min: $minD | Now: $current | Max: $maxD" -ForegroundColor DarkGray
    Write-Host ""
}

Clear-Host
$lastStep = -1

while ($true) {
    if (Test-Path $logFile) {
        $lines = Get-Content $logFile -Tail 100
        
        foreach ($line in $lines) {
            if ($line -match 'Epoch\s+([\d.]+)\s*\|\s*Step\s+(\d+):\s*Loss=([\d.]+),\s*Aux=([\d.]+),\s*Temp=([\d.]+),\s*Entropy=([\d.]+),\s*LR=([\d.e+-]+)') {
                $step = [int]$matches[2]
                if ($step -gt $lastStep) {
                    $lastStep = $step
                    $script:lossHistory += [double]$matches[3]
                    $script:tempHistory += [double]$matches[5]
                    $script:entropyHistory += [double]$matches[6]
                    
                    if ($script:lossHistory.Count -gt $maxHistory) { $script:lossHistory = $script:lossHistory[-$maxHistory..-1] }
                    if ($script:tempHistory.Count -gt $maxHistory) { $script:tempHistory = $script:tempHistory[-$maxHistory..-1] }
                    if ($script:entropyHistory.Count -gt $maxHistory) { $script:entropyHistory = $script:entropyHistory[-$maxHistory..-1] }
                }
            }
        }
        
        $lastLine = $lines | Where-Object { $_ -match 'Epoch.*Step.*Loss' } | Select-Object -Last 1
        Clear-Host
        
        Write-Host "============================================" -ForegroundColor Cyan
        Write-Host "  TKS v5 STAGE 1 - TRAINING MONITOR" -ForegroundColor Cyan
        Write-Host "  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
        Write-Host "============================================" -ForegroundColor Cyan
        Write-Host ""
        
        if ($lastLine -match 'Epoch\s+([\d.]+)\s*\|\s*Step\s+(\d+):\s*Loss=([\d.]+),\s*Aux=([\d.]+),\s*Temp=([\d.]+),\s*Entropy=([\d.]+),\s*LR=([\d.e+-]+)') {
            $epoch = $matches[1]; $step = $matches[2]; $loss = $matches[3]
            $aux = $matches[4]; $temp = $matches[5]; $entropy = $matches[6]; $lr = $matches[7]
            
            $totalSteps = 336671
            $progress = [math]::Round(([int]$step / $totalSteps) * 100, 2)
            $pBar = "[" + ("=" * [math]::Floor($progress/2.5)) + ("." * (40 - [math]::Floor($progress/2.5))) + "]"
            $eta = [math]::Round(($totalSteps - [int]$step) / 0.8 / 3600, 1)
            
            Write-Host "  PROGRESS: $progress%" -ForegroundColor Green
            Write-Host "  $pBar" -ForegroundColor White
            Write-Host "  Step $step / $totalSteps | ETA: ~$eta hours" -ForegroundColor Gray
            Write-Host ""
            
            Write-Host "  +------------+------------------+" -ForegroundColor DarkGray
            Write-Host "  | Epoch      | $($epoch.PadRight(16)) |" -ForegroundColor White
            Write-Host "  | Step       | $($step.ToString().PadRight(16)) |" -ForegroundColor White
            $lc = if ([double]$loss -lt 0.5){"Green"}elseif([double]$loss -lt 1.0){"Cyan"}elseif([double]$loss -lt 2.0){"Yellow"}else{"Red"}
            Write-Host "  | Loss       | " -NoNewline; Write-Host "$($loss.PadRight(16))" -NoNewline -ForegroundColor $lc; Write-Host " |"
            Write-Host "  | Temp       | $($temp.PadRight(16)) |" -ForegroundColor Cyan
            Write-Host "  | Entropy    | $($entropy.PadRight(16)) |" -ForegroundColor Cyan
            Write-Host "  | LR         | $($lr.PadRight(16)) |" -ForegroundColor Magenta
            Write-Host "  +------------+------------------+" -ForegroundColor DarkGray
            Write-Host ""
            
            Write-Host "  ==================== GRAPHS ====================" -ForegroundColor Blue
            Write-Host "  LOSS: Cross-entropy loss (lower=better learning)" -ForegroundColor DarkGray
            Write-Host "  TEMP: Routing temperature (anneals from 2.0 to 0)" -ForegroundColor DarkGray
            Write-Host "  ENTROPY: Router diversity (stable ~2.2 is good)" -ForegroundColor DarkGray
            Write-Host ""

            if ($script:lossHistory.Count -gt 0) {
                $lMin = ($script:lossHistory | Measure-Object -Minimum).Minimum
                $lMax = ($script:lossHistory | Measure-Object -Maximum).Maximum
                Draw-Graph "LOSS" $script:lossHistory $lMin $lMax "DarkRed"
            }
            Draw-Graph "TEMP" $script:tempHistory 0.3 2.0 "Cyan"
            Draw-Graph "ENTROPY" $script:entropyHistory 2.1 2.4 "Green"
            
            Write-Host "  ================================================" -ForegroundColor Blue
        }
    }
    Write-Host ""; Write-Host "  [Refresh: 2s | Ctrl+C to exit]" -ForegroundColor DarkGray
    Start-Sleep -Seconds 2
}
