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

def novelty_viz(novelty_class):
    """Visualize novelty classification."""
    if novelty_class == "HEAVY":
        return G + BOLD + "HEAVY" + RST + " ★"
    elif novelty_class == "COUNT":
        return Y + "COUNT" + RST + " +"
    else:
        return DIM + "NOCOUNT" + RST

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

log_file = "training_log_v7.txt"
best_loss = 999
total_unlocks = 0

# v7 pattern: Epoch X.XX | Step XXX: Loss=X.XXXX, Depth=X, p_max=X, Tokens=X, Novelty=XXX, LR=X.XXe-XX
pattern = re.compile(r'Epoch ([\d.]+) \| Step (\d+): Loss=([\d.]+), Depth=(\d+), p_max=(\d+), Tokens=(\d+), Novelty=(\w+)')

# Start by showing last 20 lines
if os.path.exists(log_file):
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        last_pos = f.tell()
    # Process last 20 lines to catch up
    for line in lines[-20:]:
        m = pattern.search(line)
        if m:
            epoch = float(m[1])
            step = int(m[2])
            loss = float(m[3])
            depth = int(m[4])
            p_max = int(m[5])
            tokens = int(m[6])
            novelty = m[7]

            if loss < best_loss:
                best_loss = loss

            lc = color_loss(loss)
            dv = depth_viz(depth, p_max)
            nv = novelty_viz(novelty)
            tv = tokens_viz(tokens)

            pct = (epoch / 5) * 100  # 5 epochs for v7
            bar = G + '█' * int(pct/5) + DIM + '░' * (20-int(pct/5)) + RST

            print(f"{W}Epoch{RST} {bar} {BOLD}{epoch:.2f}/5{RST}  "
                  f"{W}Step{RST} {C}{step:>6}{RST}  "
                  f"{W}Loss{RST} {lc}{BOLD}{loss:.4f}{RST}  "
                  f"{W}Depth{RST} {dv}  "
                  f"{W}p_max{RST}={M}{p_max}{RST}  "
                  f"{W}Tok{RST} {tv}  "
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
            m = pattern.search(line)
            if m:
                epoch = float(m[1])
                step = int(m[2])
                loss = float(m[3])
                depth = int(m[4])
                p_max = int(m[5])
                tokens = int(m[6])
                novelty = m[7]

                if loss < best_loss:
                    best_loss = loss

                lc = color_loss(loss)
                dv = depth_viz(depth, p_max)
                nv = novelty_viz(novelty)
                tv = tokens_viz(tokens)

                pct = (epoch / 5) * 100
                bar = G + '█' * int(pct/5) + DIM + '░' * (20-int(pct/5)) + RST

                print(f"{W}Epoch{RST} {bar} {BOLD}{epoch:.2f}/5{RST}  "
                      f"{W}Step{RST} {C}{step:>6}{RST}  "
                      f"{W}Loss{RST} {lc}{BOLD}{loss:.4f}{RST}  "
                      f"{W}Depth{RST} {dv}  "
                      f"{W}p_max{RST}={M}{p_max}{RST}  "
                      f"{W}Tok{RST} {tv}  "
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
