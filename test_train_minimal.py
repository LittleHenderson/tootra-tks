"""Minimal training test"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.train_story_equation_lite import train_model

print("Starting minimal training test...")
print("This will train for 3 epochs with batch size 8")
print("-" * 70)

metrics = train_model(
    data_path='output/teacher_story_eq_train.jsonl',
    output_dir='output/teacher_model_story',
    epochs=3,
    batch_size=8,
    lr=1e-3
)

print("\n" + "=" * 70)
print("TRAINING TEST COMPLETE!")
print("=" * 70)
print(f"\nMetrics saved. Final eval loss: {metrics['eval_losses'][-1]['loss']:.4f}")
