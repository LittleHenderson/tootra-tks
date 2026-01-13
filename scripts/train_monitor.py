"""Live Training Monitor - Watch training progress in real-time."""
import os
import sys
import time
import re
from collections import deque
from pathlib import Path

# ANSI colors
class C:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    DIM = '\033[2m'

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def draw_bar(value, max_val, width=40, fill_char='█', empty_char='░'):
    """Draw a progress bar."""
    if max_val == 0:
        filled = 0
    else:
        filled = int((value / max_val) * width)
    return fill_char * filled + empty_char * (width - filled)

def draw_loss_graph(losses, width=60, height=10):
    """Draw ASCII loss graph."""
    if len(losses) < 2:
        return "  Waiting for data..."

    # Get last N losses that fit in width
    recent = list(losses)[-width:]

    if not recent:
        return "  No data"

    min_loss = min(recent)
    max_loss = max(recent)

    if max_loss == min_loss:
        max_loss = min_loss + 0.1

    lines = []
    for row in range(height):
        threshold = max_loss - (row / (height - 1)) * (max_loss - min_loss)
        line = ""
        for val in recent:
            if val >= threshold:
                line += "█"
            else:
                line += " "

        # Y-axis label
        if row == 0:
            label = f"{max_loss:.4f}"
        elif row == height - 1:
            label = f"{min_loss:.4f}"
        else:
            label = ""

        lines.append(f"  {label:>8} │{line}│")

    # X-axis
    lines.append(f"           └{'─' * len(recent)}┘")
    lines.append(f"            Step {len(losses)-len(recent)+1} → {len(losses)}")

    return "\n".join(lines)

def parse_log_line(line):
    """Parse a training log line for metrics."""
    metrics = {}

    # Common patterns
    patterns = [
        (r'[Ee]poch[:\s]+(\d+)', 'epoch'),
        (r'[Ss]tep[:\s]+(\d+)', 'step'),
        (r'[Bb]atch[:\s]+(\d+)', 'batch'),
        (r'[Ll]oss[:\s]+([0-9.]+)', 'loss'),
        (r'CE[:\s]+([0-9.]+)', 'ce_loss'),
        (r'[Ll]r[:\s]+([0-9.e-]+)', 'lr'),
        (r'[Dd]epth[:\s]+(\d+)', 'depth'),
        (r'NW[:\s]+([0-9.]+)', 'nw'),
        (r'[Aa]cc[:\s]+([0-9.]+)', 'acc'),
    ]

    for pattern, name in patterns:
        match = re.search(pattern, line)
        if match:
            try:
                val = float(match.group(1))
                metrics[name] = val
            except:
                pass

    return metrics

def monitor_file(log_path):
    """Monitor a log file for training progress."""
    losses = deque(maxlen=200)
    current_metrics = {}
    start_time = time.time()
    last_size = 0

    print(f"{C.CYAN}Monitoring: {log_path}{C.RESET}")
    print(f"{C.DIM}Press Ctrl+C to exit{C.RESET}\n")

    while True:
        try:
            clear_screen()

            elapsed = time.time() - start_time

            # Header
            print(f"{C.BOLD}{C.CYAN}╔══════════════════════════════════════════════════════════════════╗{C.RESET}")
            print(f"{C.BOLD}{C.CYAN}║           TKS TRAINING MONITOR                                   ║{C.RESET}")
            print(f"{C.BOLD}{C.CYAN}╚══════════════════════════════════════════════════════════════════╝{C.RESET}")
            print()

            # Read new lines from log
            if os.path.exists(log_path):
                with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                    f.seek(last_size)
                    new_content = f.read()
                    last_size = f.tell()

                    for line in new_content.split('\n'):
                        if line.strip():
                            metrics = parse_log_line(line)
                            if metrics:
                                current_metrics.update(metrics)
                                if 'loss' in metrics:
                                    losses.append(metrics['loss'])
                                elif 'ce_loss' in metrics:
                                    losses.append(metrics['ce_loss'])

            # Status
            print(f"  {C.BOLD}Status:{C.RESET} {C.GREEN}TRAINING{C.RESET}  |  Elapsed: {int(elapsed//60)}m {int(elapsed%60)}s")
            print()

            # Current metrics
            print(f"  {C.BOLD}Current Metrics:{C.RESET}")
            print(f"  ─────────────────────────────────────")

            if 'epoch' in current_metrics:
                print(f"  Epoch:  {C.YELLOW}{int(current_metrics['epoch'])}{C.RESET}")
            if 'step' in current_metrics or 'batch' in current_metrics:
                step = current_metrics.get('step', current_metrics.get('batch', 0))
                print(f"  Step:   {C.YELLOW}{int(step)}{C.RESET}")
            if 'loss' in current_metrics or 'ce_loss' in current_metrics:
                loss = current_metrics.get('loss', current_metrics.get('ce_loss', 0))
                color = C.GREEN if loss < 1.0 else C.YELLOW if loss < 3.0 else C.RED
                print(f"  Loss:   {color}{loss:.6f}{C.RESET}")
            if 'depth' in current_metrics:
                print(f"  Depth:  {C.CYAN}{int(current_metrics['depth'])}{C.RESET}")
            if 'nw' in current_metrics:
                print(f"  NW:     {C.BLUE}{current_metrics['nw']:.4f}{C.RESET}")
            if 'lr' in current_metrics:
                print(f"  LR:     {current_metrics['lr']:.2e}")
            if 'acc' in current_metrics:
                print(f"  Acc:    {C.GREEN}{current_metrics['acc']:.2%}{C.RESET}")

            print()

            # Loss graph
            print(f"  {C.BOLD}Loss History:{C.RESET}")
            print(draw_loss_graph(losses))
            print()

            # Progress bar if we have epoch info
            if 'epoch' in current_metrics:
                epoch = int(current_metrics['epoch'])
                total_epochs = 10  # Default assumption
                print(f"  {C.BOLD}Progress:{C.RESET}")
                print(f"  [{draw_bar(epoch, total_epochs, 50)}] {epoch}/{total_epochs}")

            print()
            print(f"  {C.DIM}Log: {log_path}{C.RESET}")
            print(f"  {C.DIM}Ctrl+C to exit{C.RESET}")

            time.sleep(1)

        except KeyboardInterrupt:
            print(f"\n{C.YELLOW}Monitor stopped.{C.RESET}")
            break
        except Exception as e:
            print(f"{C.RED}Error: {e}{C.RESET}")
            time.sleep(2)

def monitor_stdout():
    """Monitor stdout from a piped training process."""
    losses = deque(maxlen=200)
    current_metrics = {}
    start_time = time.time()
    lines_buffer = deque(maxlen=20)

    print(f"{C.CYAN}Reading from stdin...{C.RESET}")
    print(f"{C.DIM}Pipe training output here, e.g.: python train.py | python train_monitor.py{C.RESET}\n")

    import select
    import threading

    def read_input():
        for line in sys.stdin:
            lines_buffer.append(line.strip())
            metrics = parse_log_line(line)
            if metrics:
                current_metrics.update(metrics)
                if 'loss' in metrics:
                    losses.append(metrics['loss'])

    reader = threading.Thread(target=read_input, daemon=True)
    reader.start()

    while reader.is_alive():
        try:
            clear_screen()
            elapsed = time.time() - start_time

            print(f"{C.BOLD}{C.CYAN}╔══════════════════════════════════════════════════════════════════╗{C.RESET}")
            print(f"{C.BOLD}{C.CYAN}║           TKS TRAINING MONITOR (LIVE)                            ║{C.RESET}")
            print(f"{C.BOLD}{C.CYAN}╚══════════════════════════════════════════════════════════════════╝{C.RESET}")
            print()
            print(f"  Elapsed: {int(elapsed//60)}m {int(elapsed%60)}s")
            print()

            # Current metrics
            if current_metrics:
                print(f"  {C.BOLD}Metrics:{C.RESET}")
                for k, v in current_metrics.items():
                    if isinstance(v, float):
                        print(f"    {k}: {v:.6f}")
                    else:
                        print(f"    {k}: {v}")

            print()
            print(f"  {C.BOLD}Loss Graph:{C.RESET}")
            print(draw_loss_graph(losses))
            print()

            # Recent output
            print(f"  {C.BOLD}Recent Output:{C.RESET}")
            for line in list(lines_buffer)[-5:]:
                print(f"  {C.DIM}{line[:70]}{C.RESET}")

            time.sleep(0.5)
        except KeyboardInterrupt:
            break

def find_latest_log():
    """Find the most recent training log."""
    log_patterns = [
        'training_log.txt',
        'logs/*.log',
        'logs/*.txt',
        '*.log',
    ]

    candidates = []
    for pattern in log_patterns:
        for p in Path('.').glob(pattern):
            if p.is_file():
                candidates.append((p.stat().st_mtime, p))

    if candidates:
        candidates.sort(reverse=True)
        return str(candidates[0][1])
    return None

if __name__ == '__main__':
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
        if os.path.exists(log_path):
            monitor_file(log_path)
        else:
            print(f"{C.RED}File not found: {log_path}{C.RESET}")
    elif not sys.stdin.isatty():
        # Piped input
        monitor_stdout()
    else:
        # Try to find a log file
        log_path = find_latest_log()
        if log_path:
            print(f"{C.GREEN}Found log: {log_path}{C.RESET}")
            monitor_file(log_path)
        else:
            print(f"{C.BOLD}TKS Training Monitor{C.RESET}")
            print()
            print("Usage:")
            print(f"  python train_monitor.py <log_file>      {C.DIM}# Watch a log file{C.RESET}")
            print(f"  python train.py | python train_monitor.py  {C.DIM}# Pipe training output{C.RESET}")
            print()
            print("Or place a training_log.txt in the current directory.")
