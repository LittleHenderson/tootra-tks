"""TKS v7 Training Monitor - Earned Depth Edition"""
import time
import os
import re
import sys

# Enable Windows ANSI
os.system('')

# Colors
R = '\033[91m'  # Red
G = '\033[92m'  # Green
Y = '\033[93m'  # Yellow
B = '\033[94m'  # Blue
M = '\033[95m'  # Magenta
C = '\033[96m'  # Cyan
W = '\033[97m'  # White
BOLD = '\033[1m'
DIM = '\033[2m'
RST = '\033[0m'

def color_loss(loss):
    if loss > 5: return R
    if loss > 3: return Y
    if loss > 2: return C
    return G

def depth_viz(d, p_max):
    """Visualize current depth vs max allowed (earned depth)."""
    colors = [G, C, Y, R, M]
    result = ''
    for i in range(5):
        if i < d:
            result += colors[min(i, 4)] + '█' + RST
        elif i < p_max:
            result += DIM + '▒' + RST  # Unlocked but not used
        else:
            result += DIM + '░' + RST  # Locked
    return result

def nw_viz(nw):
    """Visualize novelty weight."""
    if nw is None:
        return DIM + "NW:-" + RST
    if nw >= 0.7:
        color = G
    elif nw >= 0.45:
        color = Y
    else:
        color = C
    return f"{color}NW:{nw:.3f}{RST}"

def tokens_viz(tokens, threshold=5):
    """Visualize token progress toward unlock."""
    filled = min(tokens, threshold)
    return C + '●' * filled + DIM + '○' * (threshold - filled) + RST

print(f"""
{M}{BOLD}╔════════════════════════════════════════════════════════════╗
║       TKS v7 EARNED DEPTH ENGINE - TRAINING MONITOR        ║
║                   Creative Freedom Mode                     ║
╚════════════════════════════════════════════════════════════╝{RST}
""")

log_file = os.environ.get("TKS_LOG_FILE", "training_log_v7.txt")
total_epochs = float(os.environ.get("TKS_EPOCHS", "5"))
best_loss = 999
total_unlocks = 0

# v7 pattern: Epoch N | Batch B/Total | Loss: X.XXXX (CE: X.XXXX) | Depth: D | Tokens: T | NW: X.XXXX | Mem: C/CAP
base_pattern = re.compile(
    r'Epoch\s+([\d.]+)\s*\|\s*Batch\s+(\d+)(?:/\d+)?\s*\|\s*'
    r'Loss:\s*([-+]?[\d.]+)\s*\(CE:\s*([-+]?[\d.]+)\)\s*\|\s*Depth:\s*(\d+)'
)
tokens_pattern = re.compile(r'Tokens:\s*(\d+)')
mem_pattern = re.compile(r'Mem:\s*(\d+)/(\d+)')
nw_pattern = re.compile(r'NW:\s*([\d.]+)')

# Start by showing last 20 lines
if os.path.exists(log_file):
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        last_pos = f.tell()
    # Process last 20 lines to catch up
    for line in lines[-20:]:
        m = base_pattern.search(line)
        if m:
            epoch = float(m[1])
            step = int(m[2])
            loss = float(m[3])
            depth = int(m[5])
            tokens_match = tokens_pattern.search(line)
            mem_match = mem_pattern.search(line)
            nw_match = nw_pattern.search(line)

            tokens = int(tokens_match.group(1)) if tokens_match else None
            mem_cur = int(mem_match.group(1)) if mem_match else None
            mem_cap = int(mem_match.group(2)) if mem_match else None
            nw = float(nw_match.group(1)) if nw_match else None

            if loss < best_loss:
                best_loss = loss

            lc = color_loss(loss)
            dv = depth_viz(depth, depth)
            nv = nw_viz(nw)
            tv = tokens_viz(tokens or 0)
            mem = f"{mem_cur}/{mem_cap}" if mem_cur is not None else "-"

            pct = (epoch / total_epochs) * 100 if total_epochs else 0
            bar = G + '█' * int(pct/5) + DIM + '░' * (20-int(pct/5)) + RST

            print(f"{W}Epoch{RST} {bar} {BOLD}{epoch:.2f}/5{RST}  "
                  f"{W}Step{RST} {C}{step:>6}{RST}  "
                  f"{W}Loss{RST} {lc}{BOLD}{loss:.4f}{RST}  "
                  f"{W}Depth{RST} {dv}  "
                  f"{W}Tok{RST} {tv}  "
                  f"{W}Mem{RST} {M}{mem}{RST}  "
                  f"{nv}")
    print(f"\n{Y}--- Live updates below ---{RST}\n")
else:
    last_pos = 0

while True:
    try:
        if not os.path.exists(log_file):
            print(f"{Y}Waiting for training_log_v7.txt...{RST}", end='\r')
            time.sleep(1)
            continue

        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            f.seek(last_pos)
            lines = f.readlines()
            last_pos = f.tell()

        for line in lines:
            # Parse v7 training line
            m = base_pattern.search(line)
            if m:
                epoch = float(m[1])
                step = int(m[2])
                loss = float(m[3])
                depth = int(m[5])
                tokens_match = tokens_pattern.search(line)
                mem_match = mem_pattern.search(line)
                nw_match = nw_pattern.search(line)

                tokens = int(tokens_match.group(1)) if tokens_match else None
                mem_cur = int(mem_match.group(1)) if mem_match else None
                mem_cap = int(mem_match.group(2)) if mem_match else None
                nw = float(nw_match.group(1)) if nw_match else None

                if loss < best_loss:
                    best_loss = loss

                lc = color_loss(loss)
                dv = depth_viz(depth, depth)
                nv = nw_viz(nw)
                tv = tokens_viz(tokens or 0)
                mem = f"{mem_cur}/{mem_cap}" if mem_cur is not None else "-"

                pct = (epoch / total_epochs) * 100 if total_epochs else 0
                bar = G + '█' * int(pct/5) + DIM + '░' * (20-int(pct/5)) + RST

                print(f"{W}Epoch{RST} {bar} {BOLD}{epoch:.2f}/5{RST}  "
                      f"{W}Step{RST} {C}{step:>6}{RST}  "
                      f"{W}Loss{RST} {lc}{BOLD}{loss:.4f}{RST}  "
                      f"{W}Depth{RST} {dv}  "
                      f"{W}Tok{RST} {tv}  "
                      f"{W}Mem{RST} {M}{mem}{RST}  "
                      f"{nv}")

            # Validation
            if 'Val Loss' in line:
                m2 = re.search(r'Train Loss: ([\d.]+) \| Val Loss: ([\d.]+)', line)
                if m2:
                    print(f"\n{M}{BOLD}>>> VALIDATION: Train={m2[1]}, Val={m2[2]}{RST}\n")

            # Epoch complete
            if 'Complete' in line and 'Epoch' in line:
                print(f"\n{G}{BOLD}{'='*60}{RST}")
                print(f"{G}{BOLD}   {line.strip()}{RST}")
                print(f"{G}{BOLD}{'='*60}{RST}\n")

            # DPS state
            if 'DPS State' in line:
                print(f"{M}{line.strip()}{RST}")

            # Unlocks
            if 'unlocks' in line.lower() and 'total' in line.lower():
                print(f"{G}{BOLD}★ DEPTH UNLOCKED! {line.strip()}{RST}")

        time.sleep(0.3)

    except KeyboardInterrupt:
        print(f"\n{Y}Stopped. Best loss: {best_loss:.4f}{RST}")
        break
    except Exception as e:
        time.sleep(1)
