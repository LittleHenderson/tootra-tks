"""
TKS v6 Training Monitor - Real-time colorful progress display
"""
import time
import os
import sys
import re
from collections import deque

# ANSI Colors
class C:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[35m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'
    BG_BLUE = '\033[44m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_RED = '\033[41m'

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def progress_bar(value, max_val, width=30, fill_char='█', empty_char='░'):
    """Create a colorful progress bar."""
    if max_val == 0:
        return empty_char * width
    pct = min(value / max_val, 1.0)
    filled = int(width * pct)

    # Color based on progress
    if pct < 0.33:
        color = C.RED
    elif pct < 0.66:
        color = C.YELLOW
    else:
        color = C.GREEN

    bar = color + fill_char * filled + C.DIM + empty_char * (width - filled) + C.RESET
    return bar

def loss_color(loss):
    """Color based on loss value."""
    if loss > 5:
        return C.RED
    elif loss > 3:
        return C.YELLOW
    elif loss > 2:
        return C.CYAN
    else:
        return C.GREEN

def depth_bar(depth, max_depth=4):
    """Visual depth indicator."""
    colors = [C.GREEN, C.CYAN, C.YELLOW, C.RED]
    bar = ""
    for i in range(max_depth):
        if i < depth:
            bar += colors[i] + "▓" + C.RESET
        else:
            bar += C.DIM + "░" + C.RESET
    return bar

def halt_indicator(halt_prob):
    """Visual halt probability indicator."""
    blocks = int(halt_prob * 10)
    if halt_prob > 0.7:
        color = C.GREEN
    elif halt_prob > 0.5:
        color = C.YELLOW
    else:
        color = C.CYAN
    return color + "●" * blocks + C.DIM + "○" * (10 - blocks) + C.RESET

def parse_line(line):
    """Parse a training log line."""
    # Match: Epoch 0.02 | Step 205: Loss=2.8232, Depth=0, Halt=0.56, LR=5.00e-05
    pattern = r'Epoch ([\d.]+) \| Step (\d+): Loss=([\d.]+), Depth=(\d+), Halt=([\d.]+), LR=([\d.e-]+)'
    match = re.search(pattern, line)
    if match:
        return {
            'epoch': float(match.group(1)),
            'step': int(match.group(2)),
            'loss': float(match.group(3)),
            'depth': int(match.group(4)),
            'halt': float(match.group(5)),
            'lr': float(match.group(6))
        }

    # Match validation line
    if 'Val Loss' in line:
        match = re.search(r'Train Loss: ([\d.]+) \| Val Loss: ([\d.]+)', line)
        if match:
            return {
                'type': 'validation',
                'train_loss': float(match.group(1)),
                'val_loss': float(match.group(2))
            }

    # Match epoch complete
    if 'Epoch' in line and 'Complete' in line:
        match = re.search(r'Epoch (\d+)/(\d+) Complete', line)
        if match:
            return {
                'type': 'epoch_complete',
                'current': int(match.group(1)),
                'total': int(match.group(2))
            }

    return None

def main():
    log_file = "training_log.txt"

    # Enable ANSI on Windows
    if os.name == 'nt':
        os.system('color')
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

    print(f"{C.CYAN}TKS v6 Training Monitor{C.RESET}")
    print(f"{C.DIM}Watching: {log_file}{C.RESET}")
    print(f"{C.DIM}Press Ctrl+C to exit{C.RESET}\n")

    last_pos = 0
    loss_history = deque(maxlen=50)
    depth_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    total_steps = 0
    best_loss = float('inf')
    start_time = time.time()
    last_step = 0
    steps_per_epoch = 12500  # 50000 / 4
    total_epochs = 10

    while True:
        try:
            if not os.path.exists(log_file):
                print(f"{C.YELLOW}Waiting for training to start...{C.RESET}", end='\r')
                time.sleep(1)
                continue

            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(last_pos)
                new_lines = f.readlines()
                last_pos = f.tell()

            for line in new_lines:
                data = parse_line(line)
                if not data:
                    continue

                if data.get('type') == 'validation':
                    # Validation results
                    print(f"\n{C.BG_BLUE}{C.WHITE}{C.BOLD} VALIDATION {C.RESET}")
                    print(f"  Train: {loss_color(data['train_loss'])}{data['train_loss']:.4f}{C.RESET}")
                    print(f"  Val:   {loss_color(data['val_loss'])}{data['val_loss']:.4f}{C.RESET}\n")
                    continue

                if data.get('type') == 'epoch_complete':
                    # Epoch complete
                    print(f"\n{C.BG_GREEN}{C.WHITE}{C.BOLD} EPOCH {data['current']}/{data['total']} COMPLETE {C.RESET}\n")
                    continue

                # Regular training step
                total_steps += 1
                loss_history.append(data['loss'])
                depth_counts[min(data['depth'], 4)] += 1

                if data['loss'] < best_loss:
                    best_loss = data['loss']

                # Calculate stats
                avg_loss = sum(loss_history) / len(loss_history)
                elapsed = time.time() - start_time
                steps_done = data['step']
                total_target = steps_per_epoch * total_epochs

                if steps_done > last_step:
                    speed = (steps_done - last_step) / 5 if elapsed > 5 else 0
                    last_step = steps_done

                # Clear and redraw
                clear_screen()

                # Header
                print(f"{C.BOLD}{C.MAGENTA}╔══════════════════════════════════════════════════════════════╗{C.RESET}")
                print(f"{C.BOLD}{C.MAGENTA}║{C.RESET}    {C.CYAN}{C.BOLD}TKS v6 REASONING ENGINE - TRAINING MONITOR{C.RESET}        {C.BOLD}{C.MAGENTA}║{C.RESET}")
                print(f"{C.BOLD}{C.MAGENTA}╚══════════════════════════════════════════════════════════════╝{C.RESET}\n")

                # Progress
                epoch_pct = data['epoch'] / total_epochs
                print(f"  {C.WHITE}EPOCH{C.RESET}    {progress_bar(data['epoch'], total_epochs, 40)} {C.BOLD}{data['epoch']:.2f}/{total_epochs}{C.RESET}")
                print(f"  {C.WHITE}STEP{C.RESET}     {C.CYAN}{data['step']:,}{C.RESET} / {total_target:,}\n")

                # Current metrics
                print(f"  {C.BOLD}┌─────────────────────────────────────────────────────────┐{C.RESET}")
                print(f"  {C.BOLD}│{C.RESET} {C.WHITE}LOSS{C.RESET}      {loss_color(data['loss'])}{C.BOLD}{data['loss']:.4f}{C.RESET}  (avg: {avg_loss:.4f}, best: {C.GREEN}{best_loss:.4f}{C.RESET})")
                print(f"  {C.BOLD}│{C.RESET} {C.WHITE}DEPTH{C.RESET}     {depth_bar(data['depth'])}  Level {C.BOLD}{data['depth']}{C.RESET}")
                print(f"  {C.BOLD}│{C.RESET} {C.WHITE}HALT{C.RESET}      {halt_indicator(data['halt'])}  {data['halt']:.2f}")
                print(f"  {C.BOLD}│{C.RESET} {C.WHITE}LR{C.RESET}        {C.DIM}{data['lr']:.2e}{C.RESET}")
                print(f"  {C.BOLD}└─────────────────────────────────────────────────────────┘{C.RESET}\n")

                # Depth distribution
                total_depth = sum(depth_counts.values())
                if total_depth > 0:
                    print(f"  {C.WHITE}RECURSION DEPTH DISTRIBUTION:{C.RESET}")
                    for d in range(5):
                        pct = depth_counts[d] / total_depth
                        bar_width = int(pct * 30)
                        colors = [C.GREEN, C.CYAN, C.YELLOW, C.RED, C.MAGENTA]
                        print(f"    D{d}: {colors[d]}{'█' * bar_width}{C.DIM}{'░' * (30-bar_width)}{C.RESET} {pct*100:5.1f}%")
                    print()

                # Status
                if data['loss'] > 5:
                    status = f"{C.YELLOW}Warming up...{C.RESET}"
                elif data['loss'] > 3:
                    status = f"{C.CYAN}Learning patterns...{C.RESET}"
                elif data['loss'] > 2:
                    status = f"{C.GREEN}Good progress!{C.RESET}"
                else:
                    status = f"{C.GREEN}{C.BOLD}Excellent! Reasoning acquired!{C.RESET}"

                print(f"  {C.WHITE}STATUS:{C.RESET} {status}")
                print(f"  {C.DIM}Elapsed: {elapsed/60:.1f} min{C.RESET}")

            time.sleep(0.5)

        except KeyboardInterrupt:
            print(f"\n{C.YELLOW}Monitor stopped.{C.RESET}")
            break
        except Exception as e:
            print(f"{C.RED}Error: {e}{C.RESET}")
            time.sleep(1)

if __name__ == "__main__":
    main()
