# Task: Create Colorful Real-Time Training Monitor for TKS v6

## Context

We are training TKS v6 (Theosophic Kabbalah System) - a reasoning-capable LLM with recursive prerequisite processing. The training outputs to `training_log.txt` with this format:

```
Epoch 0.02 | Step 255: Loss=2.8058, Depth=3, Halt=0.58, LR=5.00e-05
```

We need a **colorful, real-time terminal monitor** that displays training progress visually.

## Log File Format

**Location:** `training_log.txt` (same directory as script)

**Training Line Format:**
```
Epoch {float} | Step {int}: Loss={float}, Depth={int}, Halt={float}, LR={scientific}
```

Example:
```
Epoch 0.02 | Step 255: Loss=2.8058, Depth=3, Halt=0.58, LR=5.00e-05
```

**Validation Line Format:**
```
Train Loss: {float} | Val Loss: {float}
```

**Epoch Complete Format:**
```
=== Epoch {int}/{int} Complete ===
```

**Best Model Line:**
```
New best model saved (val_loss={float})
```

## Requirements

### 1. Windows Compatibility (CRITICAL)
- Must work on Windows PowerShell
- Enable ANSI escape codes using ctypes:
```python
import ctypes
kernel32 = ctypes.windll.kernel32
kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
```

### 2. Visual Elements Needed

#### A. Header Box (Magenta/Purple)
```
╔══════════════════════════════════════════════════════════════════╗
║         TKS v6 REASONING ENGINE - TRAINING MONITOR               ║
╚══════════════════════════════════════════════════════════════════╝
```

#### B. Progress Bar for Epoch
- 10 total epochs
- Use █ for filled, ░ for empty
- Green color
- Format: `Epoch [████████░░░░░░░░░░░░] 4.25/10`

#### C. Loss Display (Color-Coded)
- Loss > 5.0: RED (bad)
- Loss > 3.0: YELLOW (warming up)
- Loss > 2.0: CYAN (learning)
- Loss <= 2.0: GREEN (good reasoning)

#### D. Depth Visualization (IMPORTANT - This is recursion depth)
- Max depth is 4
- Show as 4 blocks: ▓▓▓░ means depth=3
- Color each level differently:
  - Level 0-1: Green
  - Level 2: Cyan
  - Level 3: Yellow
  - Level 4: Red

#### E. Halt Probability Indicator
- Value 0.0 to 1.0
- Show as 10 circles: ●●●●●●○○○○ means halt=0.6
- Green if > 0.7, Yellow if > 0.5, Cyan otherwise

#### F. Stats Footer
- Best loss achieved
- Target: "< 2.0 for good reasoning"
- Elapsed time

### 3. Behavior

1. **On Start:** Clear screen, show header, display last 20-30 lines from log
2. **Live Updates:** Poll file every 0.3-0.5 seconds for new content
3. **Screen Refresh:** Clear and redraw entire screen on each update (prevents garbled output)
4. **Handle Missing File:** Show "Waiting for training_log.txt..." message
5. **Graceful Exit:** Catch Ctrl+C, show final stats

### 4. ANSI Color Codes Reference
```python
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
WHITE = '\033[97m'
BOLD = '\033[1m'
DIM = '\033[2m'
RESET = '\033[0m'
BG_GREEN = '\033[42m'
BG_BLUE = '\033[44m'
```

### 5. Regex Pattern for Parsing
```python
import re
pattern = re.compile(r'Epoch ([\d.]+) \| Step (\d+): Loss=([\d.]+), Depth=(\d+), Halt=([\d.]+)')
match = pattern.search(line)
if match:
    epoch = float(match.group(1))
    step = int(match.group(2))
    loss = float(match.group(3))
    depth = int(match.group(4))
    halt = float(match.group(5))
```

### 6. Sample Output (ASCII representation)

```
╔══════════════════════════════════════════════════════════════════╗
║         TKS v6 REASONING ENGINE - TRAINING MONITOR               ║
╚══════════════════════════════════════════════════════════════════╝

  Ep [████░░░░░░░░░░░]  0.52/10  Step   650  Loss  2.4521  Dpth ▓▓▓░  Halt ●●●●●●○○○○
  Ep [████░░░░░░░░░░░]  0.52/10  Step   655  Loss  2.5102  Dpth ▓▓▓░  Halt ●●●●●○○○○○
  Ep [████░░░░░░░░░░░]  0.52/10  Step   660  Loss  2.3844  Dpth ▓▓▓░  Halt ●●●●●●○○○○

  ────────────────────────────────────────────────────────────────
  Best Loss: 2.3201    Target: < 2.0 for good reasoning
  Press Ctrl+C to exit
```

### 7. File Structure
Save as: `monitor_colorful.py` in the project root.

### 8. Testing
Run with:
```cmd
cd C:\Users\wakil\downloads\everthing-tootra-tks
python monitor_colorful.py
```

Training should already be running in another terminal writing to `training_log.txt`.

## Key Metrics Explanation (for display labels)

- **Epoch**: Training progress (0-10)
- **Step**: Individual batch updates
- **Loss**: Cross-entropy loss (lower = better)
- **Depth**: Recursion stack depth (0-4) - THIS IS THE KEY V6 FEATURE
- **Halt**: Noetic 9 halt probability - when to stop reasoning

## Common Issues to Avoid

1. **Don't use `print(end='\r')`** - breaks on Windows with ANSI
2. **Always clear screen before redraw** - use `os.system('cls')`
3. **Use `encoding='utf-8', errors='replace'`** when reading file
4. **Check file exists before reading**
5. **Handle empty file gracefully**

## Deliverable

A single Python file `monitor_colorful.py` that:
- Works on Windows PowerShell
- Shows colorful real-time training progress
- Updates every 0.5 seconds
- Displays all metrics with visual indicators
- Has a clean, professional appearance
