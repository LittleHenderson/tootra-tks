"""
Track 3: Capacity Tweak Training Script

Model Changes:
- Increased hidden_dim from 128 to 192 (50% increase)
- Keeps num_layers=2 and nhead=4 unchanged
- Same tokenizer (vocab_size=1000, max_length=256)

Training Recipe (v3):
- reverse_oversample: 2.0
- reverse_loss_weight: 1.5
- weight_decay: 0.01
- Cosine LR with warm restarts
- Early stop patience: 6
- Batch size: 8

Usage:
    python scripts/train_capacity.py --data output/merged_capacity_train.jsonl --output-dir output/teacher_model_capacity
"""
import json
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from collections import Counter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Simple tokenizer (unchanged from quick_train.py)
class SimpleTokenizer:
    def __init__(self, vocab_size=1000, max_length=256):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.token_to_id = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}

        # Add TKS elements
        next_id = 4
        for world in ['A', 'B', 'C', 'D']:
            for noetic in range(1, 11):
                self.token_to_id[f"{world}{noetic}"] = next_id
                next_id += 1

        # Add operators
        for op in ['+', '-', '+T', '-T', '->', '<-', '*T', '/T', 'o']:
            self.token_to_id[op] = next_id
            next_id += 1

        # Add characters (including ^ and _ for extended notation)
        for c in ' .,!?;:\'"()-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789^_':
            if c not in self.token_to_id:
                self.token_to_id[c] = next_id
                next_id += 1

        self.actual_vocab_size = next_id

    def tokenize(self, text):
        tokens = [2]  # BOS
        for c in text[:self.max_length-2]:
            tokens.append(self.token_to_id.get(c, 1))
        tokens.append(3)  # EOS
        while len(tokens) < self.max_length:
            tokens.append(0)  # PAD
        return tokens[:self.max_length]

# Dataset with reverse operation oversampling
class TKSCapacityDataset(Dataset):
    def __init__(self, data_path, tokenizer, reverse_oversample=2.0):
        self.tokenizer = tokenizer
        self.entries = []
        self.reverse_oversample = reverse_oversample

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    self.entries.append(entry)

        # Apply reverse oversampling
        self._apply_reverse_oversampling()

    def _apply_reverse_oversampling(self):
        """Oversample entries with reverse operations (<-, -T, /T)."""
        reverse_ops = {'<-', '-T', '/T'}
        oversampled = []

        for entry in self.entries:
            oversampled.append(entry)

            # Check if entry contains reverse operations
            expr_ops = entry.get('expr_ops', [])
            has_reverse = any(op in reverse_ops for op in expr_ops)

            if has_reverse:
                # Add duplicates based on oversample factor
                num_duplicates = int(self.reverse_oversample - 1)
                for _ in range(num_duplicates):
                    oversampled.append(entry)

        self.entries = oversampled

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        story = entry.get('story', '')
        input_ids = self.tokenizer.tokenize(story)
        targets = input_ids[1:] + [0]

        # Track if this is a reverse operation sample
        expr_ops = entry.get('expr_ops', [])
        reverse_ops = {'<-', '-T', '/T'}
        is_reverse = any(op in reverse_ops for op in expr_ops)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'targets': torch.tensor(targets, dtype=torch.long),
            'aug_type': entry.get('aug_type', 'original'),
            'is_reverse': is_reverse,
        }

# Enhanced Transformer with increased capacity
class CapacityTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim=192, num_layers=2, nhead=4):
        """
        Enhanced transformer with increased capacity.

        Args:
            vocab_size: Size of vocabulary
            hidden_dim: Hidden dimension (increased from 128 to 192)
            num_layers: Number of transformer layers (kept at 2)
            nhead: Number of attention heads (kept at 4)
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True),
            num_layers=num_layers
        )
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.output(x)

def train_model(data_path, output_dir, epochs=10, batch_size=8, lr=1e-3,
                reverse_oversample=2.0, reverse_loss_weight=1.5, weight_decay=0.01,
                early_stop_patience=6):
    print("=" * 70)
    print("TRACK 3: CAPACITY TWEAK TRAINING")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Setup
    tokenizer = SimpleTokenizer(max_length=256)
    dataset = TKSCapacityDataset(data_path, tokenizer, reverse_oversample=reverse_oversample)
    print(f"Loaded {len(dataset)} training samples (after oversampling)")

    # Count augmentation types
    aug_counts = Counter()
    reverse_count = 0
    for entry in dataset.entries:
        aug_type = entry.get('aug_type', 'unknown')
        aug_counts[aug_type] += 1

        expr_ops = entry.get('expr_ops', [])
        if any(op in {'<-', '-T', '/T'} for op in expr_ops):
            reverse_count += 1

    print(f"\nAugmentation distribution:")
    for aug_type, count in sorted(aug_counts.items()):
        print(f"  {aug_type}: {count}")
    print(f"\nReverse operation samples: {reverse_count} ({reverse_count/len(dataset)*100:.1f}%)")

    # Create dataloaders
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Initialize model with increased capacity
    model = CapacityTransformer(tokenizer.actual_vocab_size, hidden_dim=192, num_layers=2, nhead=4)
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")
    print(f"Model architecture:")
    print(f"  hidden_dim: 192 (vs 128 baseline, +50%)")
    print(f"  num_layers: 2")
    print(f"  num_heads: 4")

    # Setup training
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction='none')

    # Learning rate scheduler (cosine annealing with warm restarts)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=lr * 0.01)

    print(f"\nTraining configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Reverse oversample: {reverse_oversample}x")
    print(f"  Reverse loss weight: {reverse_loss_weight}x")
    print(f"  Early stop patience: {early_stop_patience}")
    print(f"  LR Scheduler: CosineAnnealingWarmRestarts")
    print(f"  Output dir: {output_dir}")

    # Training loop with early stopping
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    metrics = {
        'epoch_losses': [],
        'augmentation': dict(aug_counts),
        'config': {
            'hidden_dim': 192,
            'num_layers': 2,
            'nhead': 4,
            'reverse_oversample': reverse_oversample,
            'reverse_loss_weight': reverse_loss_weight,
            'weight_decay': weight_decay,
            'batch_size': batch_size,
            'learning_rate': lr,
        }
    }

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 70)

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)
            is_reverse = batch['is_reverse']

            optimizer.zero_grad()
            logits = model(input_ids)

            # Compute per-sample loss
            losses = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
            losses = losses.view(input_ids.size(0), -1)

            # Apply reverse loss weighting
            loss_weights = torch.ones(input_ids.size(0), device=device)
            for i, is_rev in enumerate(is_reverse):
                if is_rev:
                    loss_weights[i] = reverse_loss_weight

            # Compute weighted average loss
            mask = (targets != 0)
            sample_losses = (losses * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            loss = (sample_losses * loss_weights).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            # Print every 10 steps
            if batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  Step {batch_idx}/{len(train_loader)}: loss={loss.item():.4f}, lr={current_lr:.6f}")

        avg_loss = epoch_loss / max(num_batches, 1)
        metrics['epoch_losses'].append({'epoch': epoch + 1, 'loss': avg_loss})
        print(f"\n  Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            print(f"  New best loss! Saving checkpoint...")

            # Save best model
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            best_model_path = output_path / "best_model.pt"
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{early_stop_patience}")

            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        # Step scheduler
        scheduler.step()
        print(f"  Next epoch LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Save final model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    final_model_path = output_path / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"\nSaved final model to: {final_model_path}")

    # Save metrics
    metrics['best_loss'] = best_loss
    metrics['total_epochs'] = epoch + 1
    metrics_path = output_path / "training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nTotal epochs: {epoch + 1}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final loss: {metrics['epoch_losses'][-1]['loss']:.4f}")

    return metrics

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to training data')
    parser.add_argument('--output-dir', default='output/teacher_model_capacity', help='Output directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--reverse-oversample', type=float, default=2.0, help='Reverse operation oversample factor')
    parser.add_argument('--reverse-loss-weight', type=float, default=1.5, help='Reverse operation loss weight')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--early-stop-patience', type=int, default=6, help='Early stopping patience')

    args = parser.parse_args()
    train_model(
        args.data,
        args.output_dir,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.reverse_oversample,
        args.reverse_loss_weight,
        args.weight_decay,
        args.early_stop_patience
    )
