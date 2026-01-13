import os
import re
import time
import ctypes
import sys
from datetime import datetime

# Enable ANSI escape codes on Windows
kernel32 = ctypes.windll.kernel32
kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

# ANSI Color Codes
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

# Configuration
LOG_FILE = "training_log.txt"
TOTAL_EPOCHS = 10
UPDATE_INTERVAL = 0.5

# Regex pattern for parsing training lines (More flexible for v6)
TRAIN_PATTERN = re.compile(r'Epoch\s+([\d.]+)\s*\|\s*Step\s+(\d+):\s*Loss=([\d.]+),\s*Depth=(\d+),\s*Halt=([\d.]+)')
BEST_LOSS_PATTERN = re.compile(r'New best model saved \(val_loss=([\\d.]+)\)')

def get_loss_color(loss):
    if loss > 5.0: return RED
    if loss > 3.0: return YELLOW
    if loss > 2.0: return CYAN
    return GREEN

def get_depth_viz(depth):
    # Max depth is 4
    blocks = ""
    for i in range(1, 5):
        if i <= depth:
            if i <= 1: color = GREEN
            elif i == 2: color = CYAN
            elif i == 3: color = YELLOW
            else: color = RED
            blocks += f"{color}▓{RESET}"
        else:
            blocks += f"{DIM}░{RESET}"
    return blocks

def get_halt_viz(halt):
    # 10 circles representing 0.0 to 1.0
    num_full = int(round(halt * 10))
    if halt > 0.7: color = GREEN
    elif halt > 0.5: color = YELLOW
    else: color = CYAN
    
    viz = f"{color}"
    for i in range(10):
        if i < num_full:
            viz += "●"
        else:
            viz += f"{RESET}{DIM}○{color}"
    viz += RESET
    return viz

def get_progress_bar(epoch, total_epochs, width=20):
    filled = int(width * epoch / total_epochs)
    bar = f"{GREEN}" + "█" * filled + f"{RESET}{DIM}" + "░" * (width - filled) + f"{RESET}"
    return f"[{bar}] {epoch:.2f}/{total_epochs}"

def run_monitor():
    best_loss = float('inf')
    start_time = datetime.now()
    
    while True:
        try:
            if not os.path.exists(LOG_FILE):
                os.system('cls')
                print(f"{YELLOW}Waiting for {LOG_FILE}...{RESET}")
                time.sleep(1)
                continue

            # Robust file reading (Try multiple encodings)
            encodings = ['utf-8', 'utf-16', 'cp1252']
            lines = []
            for enc in encodings:
                try:
                    with open(LOG_FILE, 'r', encoding=enc) as f:
                        lines = f.readlines()
                    if lines and len(lines) > 0:
                        # Check if it looks like garbage (lots of null bytes)
                        if '\x00' in lines[0] and enc != 'utf-16':
                            continue
                        break # Success
                except UnicodeError:
                    continue
            
            if not lines:
                os.system('cls')
                print(f"{YELLOW}Log file is empty. Waiting...{RESET}")
                time.sleep(1)
                continue

            # Find training lines and best loss
            train_lines = []
            for line in lines:
                match = TRAIN_PATTERN.search(line)
                if match:
                    train_lines.append(match)
                    loss = float(match.group(3))
                    if loss < best_loss:
                        best_loss = loss
                
                best_match = BEST_LOSS_PATTERN.search(line)
                if best_match:
                    val_loss = float(best_match.group(1))
                    if val_loss < best_loss:
                        best_loss = val_loss

            os.system('cls')
            
            # Print Header
            print(f"{MAGENTA}╔══════════════════════════════════════════════════════════════════╗")
            print(f"║         {BOLD}TKS v6 REASONING ENGINE - TRAINING MONITOR{RESET}{MAGENTA}               ║")
            print(f"╚══════════════════════════════════════════════════════════════════╝{RESET}\n")

            if train_lines:
                # Display last 15 lines of training
                display_lines = train_lines[-15:]
                for match in display_lines:
                    epoch = float(match.group(1))
                    step = int(match.group(2))
                    loss = float(match.group(3))
                    depth = int(match.group(4))
                    halt = float(match.group(5))
                    
                    prog = get_progress_bar(epoch, TOTAL_EPOCHS)
                    l_col = get_loss_color(loss)
                    d_viz = get_depth_viz(depth)
                    h_viz = get_halt_viz(halt)
                    
                    print(f"  Ep {prog}  Step {step:6d}  Loss {l_col}{loss:7.4f}{RESET}  Dpth {d_viz}  Halt {h_viz}")

            # Print Footer
            elapsed = datetime.now() - start_time
            elapsed_str = str(elapsed).split('.')[0] # Remove microseconds
            
            print(f"\n  {DIM}────────────────────────────────────────────────────────────────{RESET}")
            print(f"  {BOLD}Best Loss:{RESET} {GREEN}{best_loss:7.4f}{RESET}    {BOLD}Target:{RESET} {CYAN}< 2.0 for good reasoning{RESET}")
            
            # Legend
            print(f"\n  {BOLD}LEGEND:{RESET}")
            print(f"  {BOLD}Depth:{RESET} {GREEN}0-1 (Linear){RESET} {CYAN}2 (Nested){RESET} {YELLOW}3 (Deep){RESET} {RED}4 (Max){RESET}")
            print(f"  {BOLD}Halt :{RESET} {CYAN}<0.5 (Thinking){RESET} {YELLOW}0.5-0.7 (Sure){RESET} {GREEN}>0.7 (Done){RESET}")
            
            print(f"  {BOLD}Elapsed:{RESET} {elapsed_str}        {DIM}Press Ctrl+C to exit{RESET}")

        except KeyboardInterrupt:
            print(f"\n{YELLOW}Monitor stopped by user.{RESET}")
            sys.exit(0)
        except Exception as e:
            print(f"{RED}Error: {e}{RESET}")
        
        time.sleep(UPDATE_INTERVAL)

if __name__ == "__main__":
    run_monitor()
