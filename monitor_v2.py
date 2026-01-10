"""TKS v6 Training Monitor - Colorful Edition"""
import time
import os
import re
import sys

# Force Windows terminal to use ANSI
if sys.platform == 'win32':
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

# Colors
class C:
    R = '\033[91m'
    G = '\033[92m'
    Y = '\033[93m'
    B = '\033[94m'
    M = '\033[95m'
    C = '\033[96m'
    W = '\033[97m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RST = '\033[0m'
    BG_G = '\033[42m'
    BG_R = '\033[41m'
    BG_B = '\033[44m'
    BG_M = '\033[45m'

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def loss_color(loss):
    if loss > 5: return C.R
    if loss > 3: return C.Y
    if loss > 2: return C.C
    return C.G

def make_bar(val, max_val, width=20, fill='█', empty='░'):
    if max_val == 0: return empty * width
    filled = int((val / max_val) * width)
    filled = min(filled, width)
    return C.G + fill * filled + C.DIM + empty * (width - filled) + C.RST

def depth_display(d):
    symbols = ['▓', '▓', '▓', '▓']
    colors = [C.G, C.C, C.Y, C.R]
    result = ''
    for i in range(4):
        if i < d:
            result += colors[i] + symbols[i] + C.RST
        else:
            result += C.DIM + '░' + C.RST
    return result

def halt_display(h):
    filled = int(h * 10)
    if h > 0.7: col = C.G
    elif h > 0.5: col = C.Y
    else: col = C.C
    return col + '●' * filled + C.DIM + '○' * (10 - filled) + C.RST

# Header
clear()
print(f"""
{C.M}{C.BOLD}╔══════════════════════════════════════════════════════════════════╗
║         TKS v6 REASONING ENGINE - TRAINING MONITOR               ║
║                    Recursive Prerequisite Model                   ║
╚══════════════════════════════════════════════════════════════════╝{C.RST}
""")

log_file = "training_log.txt"
pattern = re.compile(r'Epoch ([\d.]+) \| Step (\d+): Loss=([\d.]+), Depth=(\d+), Halt=([\d.]+)')

best_loss = 999.0
total_epochs = 10
last_size = 0

print(f"{C.Y}Waiting for training data...{C.RST}\n")

while True:
    try:
        if not os.path.exists(log_file):
            time.sleep(0.5)
            continue

        # Check if file has new content
        current_size = os.path.getsize(log_file)
        if current_size == last_size:
            time.sleep(0.3)
            continue

        # Read entire file and get new lines
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        last_size = current_size
        lines = content.strip().split('\n')

        # Process last 30 lines for display
        recent = lines[-30:] if len(lines) > 30 else lines

        clear()
        print(f"""
{C.M}{C.BOLD}╔══════════════════════════════════════════════════════════════════╗
║         TKS v6 REASONING ENGINE - TRAINING MONITOR               ║
║                    Recursive Prerequisite Model                   ║
╚══════════════════════════════════════════════════════════════════╝{C.RST}
""")

        for line in recent:
            m = pattern.search(line)
            if m:
                epoch = float(m.group(1))
                step = int(m.group(2))
                loss = float(m.group(3))
                depth = int(m.group(4))
                halt = float(m.group(5))

                if loss < best_loss:
                    best_loss = loss

                lc = loss_color(loss)
                bar = make_bar(epoch, total_epochs, 15)
                dv = depth_display(depth)
                hd = halt_display(halt)

                print(f"  {C.W}Ep{C.RST} {bar} {C.BOLD}{epoch:>5.2f}{C.RST}/{total_epochs}  "
                      f"{C.W}Step{C.RST} {C.C}{step:>6}{C.RST}  "
                      f"{C.W}Loss{C.RST} {lc}{C.BOLD}{loss:>7.4f}{C.RST}  "
                      f"{C.W}Dpth{C.RST} {dv}  "
                      f"{C.W}Halt{C.RST} {hd}")

            elif 'Val Loss' in line:
                print(f"\n  {C.BG_B}{C.W}{C.BOLD} VALIDATION {C.RST} {line.strip()}\n")

            elif 'Complete' in line:
                print(f"\n  {C.BG_G}{C.W}{C.BOLD} {line.strip()} {C.RST}\n")

            elif 'best model' in line.lower():
                print(f"  {C.G}{C.BOLD}★ NEW BEST MODEL SAVED ★{C.RST}")

        # Stats footer
        print(f"\n  {C.DIM}{'─' * 64}{C.RST}")
        print(f"  {C.W}Best Loss:{C.RST} {C.G}{C.BOLD}{best_loss:.4f}{C.RST}    "
              f"{C.W}Target:{C.RST} < 2.0 for good reasoning")
        print(f"  {C.DIM}Press Ctrl+C to exit{C.RST}")

        time.sleep(0.5)

    except KeyboardInterrupt:
        print(f"\n\n{C.Y}Monitor stopped. Training continues in background.{C.RST}")
        print(f"{C.W}Best loss achieved: {C.G}{C.BOLD}{best_loss:.4f}{C.RST}")
        break
    except Exception as e:
        print(f"{C.R}Error: {e}{C.RST}")
        time.sleep(1)
