"""Simple TKS v6 Training Monitor"""
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

def depth_viz(d):
    colors = [G, C, Y, R]
    return ''.join([colors[i] + '█' + RST if i < d else DIM + '░' + RST for i in range(4)])

print(f"""
{M}{BOLD}╔════════════════════════════════════════════════════════════╗
║       TKS v6 REASONING ENGINE - TRAINING MONITOR           ║
╚════════════════════════════════════════════════════════════╝{RST}
""")

log_file = "training_log.txt"
best_loss = 999
step_count = 0

# Start by showing last 20 lines
if os.path.exists(log_file):
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        last_pos = f.tell()
    # Process last 20 lines to catch up
    for line in lines[-20:]:
        m = re.search(r'Epoch ([\d.]+) \| Step (\d+): Loss=([\d.]+), Depth=(\d+), Halt=([\d.]+)', line)
        if m:
            epoch, step, loss, depth, halt = float(m[1]), int(m[2]), float(m[3]), int(m[4]), float(m[5])
            if loss < best_loss:
                best_loss = loss
            lc = color_loss(loss)
            dv = depth_viz(depth)
            hb = G + '●' * int(halt*10) + DIM + '○' * (10-int(halt*10)) + RST
            pct = (epoch / 10) * 100
            bar = G + '█' * int(pct/5) + DIM + '░' * (20-int(pct/5)) + RST
            print(f"{W}Epoch{RST} {bar} {BOLD}{epoch:.2f}/10{RST}  "
                  f"{W}Step{RST} {C}{step:>6}{RST}  "
                  f"{W}Loss{RST} {lc}{BOLD}{loss:.4f}{RST}  "
                  f"{W}Depth{RST} {dv}  "
                  f"{W}Halt{RST} {hb}")
    print(f"\n{Y}--- Live updates below ---{RST}\n")
else:
    last_pos = 0

while True:
    try:
        if not os.path.exists(log_file):
            print(f"{Y}Waiting for training_log.txt...{RST}", end='\r')
            time.sleep(1)
            continue

        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            f.seek(last_pos)
            lines = f.readlines()
            last_pos = f.tell()

        for line in lines:
            # Parse training line
            m = re.search(r'Epoch ([\d.]+) \| Step (\d+): Loss=([\d.]+), Depth=(\d+), Halt=([\d.]+)', line)
            if m:
                epoch, step, loss, depth, halt = float(m[1]), int(m[2]), float(m[3]), int(m[4]), float(m[5])
                step_count += 1
                if loss < best_loss:
                    best_loss = loss

                lc = color_loss(loss)
                dv = depth_viz(depth)
                hb = G + '●' * int(halt*10) + DIM + '○' * (10-int(halt*10)) + RST

                pct = (epoch / 10) * 100
                bar = G + '█' * int(pct/5) + DIM + '░' * (20-int(pct/5)) + RST

                print(f"{W}Epoch{RST} {bar} {BOLD}{epoch:.2f}/10{RST}  "
                      f"{W}Step{RST} {C}{step:>6}{RST}  "
                      f"{W}Loss{RST} {lc}{BOLD}{loss:.4f}{RST}  "
                      f"{W}Depth{RST} {dv}  "
                      f"{W}Halt{RST} {hb}")

            # Validation
            if 'Val Loss' in line:
                m = re.search(r'Train Loss: ([\d.]+) \| Val Loss: ([\d.]+)', line)
                if m:
                    print(f"\n{M}{BOLD}>>> VALIDATION: Train={m[1]}, Val={m[2]}{RST}\n")

            # Epoch complete
            if 'Complete' in line and 'Epoch' in line:
                print(f"\n{G}{BOLD}{'='*60}{RST}")
                print(f"{G}{BOLD}   {line.strip()}{RST}")
                print(f"{G}{BOLD}{'='*60}{RST}\n")

            # Best model
            if 'best model' in line.lower():
                print(f"{G}{BOLD}★ {line.strip()}{RST}")

        time.sleep(0.3)

    except KeyboardInterrupt:
        print(f"\n{Y}Stopped. Best loss: {best_loss:.4f}{RST}")
        break
    except Exception as e:
        time.sleep(1)
