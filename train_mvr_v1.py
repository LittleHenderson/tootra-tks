"""
MVR v1 Model Training Script
Agent B - MVR RPM Realignment
Trains model on regenerated MVR-aligned data with canonical mapping:
- Desire = {ν1, ν4, ν7}
- Wisdom = {ν5, ν6}
- Power = {ν8, ν9}
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the training function from quick_train
from scripts.quick_train import train_model

if __name__ == '__main__':
    print("=" * 70)
    print("AGENT B - MVR v1 MODEL TRAINING")
    print("=" * 70)
    print("\nCanonical MVR Mapping:")
    print("  Desire = {ν1, ν4, ν7}")
    print("  Wisdom = {ν5, ν6}")
    print("  Power = {ν8, ν9}")
    print("\nData: output/teacher_mvr_converted.jsonl (681 samples)")
    print("Output: output/teacher_model_mvr_v1/")
    print("\nHyperparameters (matching long_v4):")
    print("  Epochs: 10")
    print("  Batch size: 4")
    print("  Learning rate: 0.001")
    print("  Weight decay: 0.01")
    print("  Scheduler: CosineAnnealingWarmRestarts")
    print("=" * 70)
    print()

    # Execute training
    data_path = "output/teacher_mvr_converted.jsonl"
    output_dir = "output/teacher_model_mvr_v1"
    epochs = 10
    batch_size = 4
    learning_rate = 0.001

    metrics = train_model(
        data_path=data_path,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=learning_rate
    )

    print("\n" + "=" * 70)
    print("MVR v1 TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nFinal Results:")
    print(f"  Training loss: {metrics['epoch_losses'][-1]['loss']:.4f}")
    print(f"  Eval loss: {metrics['eval_losses'][-1]['loss']:.4f}")
    print(f"\nCheckpoint saved to: {output_dir}/final_model.pt")
    print(f"Metrics saved to: {output_dir}/training_metrics.json")
    print("=" * 70)
