"""
TKS Checkpoint v1 Trainer (Long Run) - Resumable

Trains the robust model checkpoint using:
- Mass Pack v2 (Diversified Phrasing)
- Gold Pack v2 (Regression)

Settings:
- 30 Epochs
- Early Stopping (Patience 5)
- BF16 / GPU Enforced
- Auto-Resume support
"""

import os
import sys
import subprocess
import shutil
import glob

# Ensure we're at root
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, "datasets", "jsonl")
OUT_DIR = os.path.join(ROOT_DIR, "checkpoints", "v1")

MASS_PACK_PATH = os.path.join(DATA_DIR, "tks_mass_v2.jsonl")
GOLD_PACK_PATH = os.path.join(DATA_DIR, "tks_gold_v2.jsonl")
MIXED_PACK_PATH = os.path.join(DATA_DIR, "tks_v1_training_mix.jsonl")

def mix_datasets(mass_path, gold_path, out_path):
    print("Mixing Datasets...")
    with open(out_path, "wb") as outfile:
        # Mix Mass Pack
        if os.path.exists(mass_path):
            with open(mass_path, "rb") as infile:
                shutil.copyfileobj(infile, outfile)
        
        # Mix Gold Pack (Duplicate it 5x)
        if os.path.exists(gold_path):
            with open(gold_path, "rb") as infile:
                gold_data = infile.read()
                for _ in range(5):
                    outfile.write(gold_data)
    
    print(f"Created Mixed Dataset: {out_path}")

def find_latest_checkpoint(out_dir):
    """Finds the checkpoint with the highest epoch number."""
    checkpoints = glob.glob(os.path.join(out_dir, "checkpoint_epoch_*.pt"))
    if not checkpoints:
        # Check for best_model as fallback
        best = os.path.join(out_dir, "best_model.pt")
        return best if os.path.exists(best) else None
    
    # Sort by epoch number
    checkpoints.sort(key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))
    return checkpoints[-1]

import re

def run_training(data_path, out_dir):
    print("\n--- Starting v1 Training (Long Run) ---")
    
    latest_ckpt = find_latest_checkpoint(out_dir)
    
    cmd = [
        sys.executable, "scripts/train_cuda.py",
        "--data", data_path,
        "--output-dir", out_dir,
        "--epochs", "30",
        "--batch-size", "8",
        "--learning-rate", "3e-4",
        "--early-stopping", "5",
        "--force-gpu",
        "--use-bf16",
        "--gradient-checkpointing",
        "--num-workers", "0",
        "--disable-auxiliary-losses"
    ]
    
    if latest_ckpt:
        print(f"Resuming from checkpoint: {latest_ckpt}")
        cmd.extend(["--checkpoint", latest_ckpt])
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Training Complete.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training Failed: {e}")
        sys.exit(1)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # Only mix if dataset doesn't exist to save time on resume
    if not os.path.exists(MIXED_PACK_PATH):
        mix_datasets(MASS_PACK_PATH, GOLD_PACK_PATH, MIXED_PACK_PATH)
    else:
        print(f"Using existing mixed dataset: {MIXED_PACK_PATH}")
        
    run_training(MIXED_PACK_PATH, OUT_DIR)

if __name__ == "__main__":
    main()
