"""
TKS Checkpoint v0 Trainer

Trains the first official model checkpoint using:
- Mass Pack v1 (Capability)
- Gold Pack v2 (Regression/Alignment)

Enforces GPU usage and Canon consistency.
"""

import os
import sys
import subprocess
import shutil

# Ensure we're at root
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, "datasets", "jsonl")
OUT_DIR = os.path.join(ROOT_DIR, "checkpoints", "v0")

MASS_PACK = os.path.join(DATA_DIR, "tks_mass_v1.jsonl")
GOLD_PACK = os.path.join(DATA_DIR, "tks_gold_v2.jsonl")
MIXED_PACK = os.path.join(DATA_DIR, "tks_v0_training_mix.jsonl")

def mix_datasets():
    print("Mixing Datasets...")
    with open(MIXED_PACK, "wb") as outfile:
        for fname in [GOLD_PACK, MASS_PACK]:
            if os.path.exists(fname):
                with open(fname, "rb") as infile:
                    shutil.copyfileobj(infile, outfile)
            else:
                print(f"Warning: {fname} not found.")
    
    print(f"Created Mixed Dataset: {MIXED_PACK}")

def run_training():
    print("\n--- Starting v0 Training on GPU ---")
    
    cmd = [
        sys.executable, "scripts/train_cuda.py",
        "--data", MIXED_PACK,
        "--output-dir", OUT_DIR,
        "--epochs", "3",
        "--batch-size", "8",
        "--learning-rate", "3e-4",
        "--force-gpu",
        "--use-bf16",
        "--gradient-checkpointing",
        "--num-workers", "0",
        "--disable-auxiliary-losses"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Training Complete.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training Failed: {e}")
        sys.exit(1)

def main():
    if not os.path.exists(MASS_PACK) or not os.path.exists(GOLD_PACK):
        print(f"CRITICAL: Missing datasets.\nMass: {os.path.exists(MASS_PACK)}\nGold: {os.path.exists(GOLD_PACK)}")
        sys.exit(1)
        
    os.makedirs(OUT_DIR, exist_ok=True)
    
    mix_datasets()
    run_training()
    
    if os.path.exists(os.path.join(OUT_DIR, "final_model.pt")):
        print(f"\nCheckpoint Artifact Ready: {os.path.join(OUT_DIR, 'final_model.pt')}")
    else:
        print("\nWarning: Final checkpoint not found.")

if __name__ == "__main__":
    main()