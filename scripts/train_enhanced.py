#!/usr/bin/env python3
"""
Enhanced training script with:
- Reverse oversampling
- Per-aug-type loss weighting
- Early stopping on holdout loss
- Per-type accuracy tracking
"""

import argparse
import json
import sys
import copy
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

sys.path.insert(0, str(Path(__file__).parent.parent))


class SimpleTokenizer:
    def __init__(self, vocab_size=1000, max_length=256):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.token_to_id = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}

        next_id = 4
        for world in ['A', 'B', 'C', 'D']:
            for noetic in range(1, 11):
                self.token_to_id[f"{world}{noetic}"] = next_id
                next_id += 1

        for op in ['+', '-', '+T', '-T', '->', '<-', '*T', '/T', 'o']:
            self.token_to_id[op] = next_id
            next_id += 1

        for c in ' .,!?;:\'"()-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789^_':
            if c not in self.token_to_id:
                self.token_to_id[c] = next_id
                next_id += 1

        self.actual_vocab_size = next_id

    def tokenize(self, text):
        tokens = [2]
        for c in text[:self.max_length-2]:
            tokens.append(self.token_to_id.get(c, 1))
        tokens.append(3)
        while len(tokens) < self.max_length:
            tokens.append(0)
        return tokens[:self.max_length]


class TKSDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.tokenizer = tokenizer
        self.entries = []

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    self.entries.append(entry)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        story = entry.get('story', '')
        input_ids = self.tokenizer.tokenize(story)
        targets = input_ids[1:] + [0]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'targets': torch.tensor(targets, dtype=torch.long),
            'aug_type': entry.get('aug_type', 'original'),
        }


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True),
            num_layers=num_layers
        )
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.output(x)


def create_weighted_sampler(dataset, reverse_weight=2.0):
    """Create weighted sampler that oversamples reverse examples."""
    weights = []
    for entry in dataset.entries:
        aug_type = entry.get('aug_type', 'original')
        if aug_type == 'reverse':
            weights.append(reverse_weight)
        else:
            weights.append(1.0)

    return WeightedRandomSampler(weights, len(weights), replacement=True)


def get_loss_weight(aug_type, reverse_loss_weight=1.5):
    """Get loss weight for augmentation type."""
    if aug_type == 'reverse':
        return reverse_loss_weight
    return 1.0


def evaluate_model(model, dataloader, loss_fn, device):
    """Evaluate model and return per-type metrics."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0

    per_type_correct = defaultdict(int)
    per_type_tokens = defaultdict(int)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)
            aug_types = batch['aug_type']

            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))

            total_loss += loss.item()
            num_batches += 1

            # Per-token accuracy
            preds = logits.argmax(dim=-1)
            mask = targets != 0

            for i, aug_type in enumerate(aug_types):
                sample_mask = mask[i]
                sample_correct = ((preds[i] == targets[i]) & sample_mask).sum().item()
                sample_tokens = sample_mask.sum().item()

                per_type_correct[aug_type] += sample_correct
                per_type_tokens[aug_type] += sample_tokens
                total_correct += sample_correct
                total_tokens += sample_tokens

    avg_loss = total_loss / max(num_batches, 1)
    accuracy = total_correct / max(total_tokens, 1)

    per_type_accuracy = {}
    for aug_type in per_type_correct:
        per_type_accuracy[aug_type] = per_type_correct[aug_type] / max(per_type_tokens[aug_type], 1)

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'per_type_accuracy': per_type_accuracy,
        'per_type_tokens': dict(per_type_tokens)
    }


def train_model(
    train_path,
    holdout_path,
    output_dir,
    epochs=25,
    batch_size=8,
    lr=1e-3,
    weight_decay=0.01,
    reverse_oversample=2.0,
    reverse_loss_weight=1.5,
    patience=5,
    min_delta=0.001
):
    print("=" * 70)
    print("ENHANCED TKS TRAINING")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Setup
    tokenizer = SimpleTokenizer(max_length=256)
    train_dataset = TKSDataset(train_path, tokenizer)
    holdout_dataset = TKSDataset(holdout_path, tokenizer)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Holdout samples: {len(holdout_dataset)}")

    # Count augmentation types
    aug_counts = defaultdict(int)
    for entry in train_dataset.entries:
        aug_counts[entry.get('aug_type', 'unknown')] += 1

    print(f"\nTrain augmentation distribution:")
    for aug_type, count in sorted(aug_counts.items()):
        print(f"  {aug_type}: {count}")

    # Create weighted sampler for oversampling reverse
    sampler = create_weighted_sampler(train_dataset, reverse_oversample)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
    holdout_loader = DataLoader(holdout_dataset, batch_size=batch_size, shuffle=False)

    # Also create unweighted eval loader for train set
    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = SimpleTransformer(tokenizer.actual_vocab_size, hidden_dim=128, num_layers=2)
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")

    # Setup training
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=lr * 0.01)

    print(f"\nTraining configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Reverse oversample: {reverse_oversample}x")
    print(f"  Reverse loss weight: {reverse_loss_weight}x")
    print(f"  Early stopping patience: {patience}")
    print(f"  Output dir: {output_dir}")

    # Training metrics
    metrics = {
        'epoch_losses': [],
        'holdout_losses': [],
        'train_accuracy': [],
        'holdout_accuracy': [],
        'per_type_accuracy': [],
        'config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'weight_decay': weight_decay,
            'reverse_oversample': reverse_oversample,
            'reverse_loss_weight': reverse_loss_weight,
        }
    }

    # Early stopping
    best_holdout_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 70)

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)
            aug_types = batch['aug_type']

            optimizer.zero_grad()
            logits = model(input_ids)

            # Compute per-sample loss with weighting
            per_token_loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
            per_token_loss = per_token_loss.view(input_ids.size(0), -1)

            # Apply per-sample weights based on aug_type
            weighted_loss = 0.0
            for i, aug_type in enumerate(aug_types):
                weight = get_loss_weight(aug_type, reverse_loss_weight)
                sample_loss = per_token_loss[i].mean()
                weighted_loss += weight * sample_loss

            loss = weighted_loss / len(aug_types)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if batch_idx % 20 == 0 or batch_idx == len(train_loader) - 1:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  Step {batch_idx}/{len(train_loader)}: loss={loss.item():.4f}, lr={current_lr:.6f}")

        # Epoch summary
        avg_train_loss = epoch_loss / max(num_batches, 1)
        metrics['epoch_losses'].append({'epoch': epoch + 1, 'loss': avg_train_loss})

        # Evaluate on holdout
        holdout_metrics = evaluate_model(model, holdout_loader, nn.CrossEntropyLoss(ignore_index=0), device)
        metrics['holdout_losses'].append({'epoch': epoch + 1, 'loss': holdout_metrics['loss']})
        metrics['holdout_accuracy'].append({'epoch': epoch + 1, 'accuracy': holdout_metrics['accuracy']})

        # Evaluate on train (for comparison)
        train_metrics = evaluate_model(model, train_eval_loader, nn.CrossEntropyLoss(ignore_index=0), device)
        metrics['train_accuracy'].append({'epoch': epoch + 1, 'accuracy': train_metrics['accuracy']})

        # Per-type accuracy
        metrics['per_type_accuracy'].append({
            'epoch': epoch + 1,
            'train': train_metrics['per_type_accuracy'],
            'holdout': holdout_metrics['per_type_accuracy']
        })

        print(f"\n  Train loss: {avg_train_loss:.4f}, accuracy: {train_metrics['accuracy']:.4f}")
        print(f"  Holdout loss: {holdout_metrics['loss']:.4f}, accuracy: {holdout_metrics['accuracy']:.4f}")
        print(f"  Per-type accuracy (holdout):")
        for aug_type, acc in sorted(holdout_metrics['per_type_accuracy'].items()):
            print(f"    {aug_type}: {acc:.4f}")

        # Step scheduler
        scheduler.step()
        print(f"  Next epoch LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping check
        if holdout_metrics['loss'] < best_holdout_loss - min_delta:
            best_holdout_loss = holdout_metrics['loss']
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            print(f"  * New best holdout loss: {best_holdout_loss:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s)")

            if epochs_without_improvement >= patience:
                print(f"\n  Early stopping triggered after {epoch + 1} epochs")
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nRestored best model (holdout loss: {best_holdout_loss:.4f})")

    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / "final_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nSaved model to: {model_path}")

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    final_train = evaluate_model(model, train_eval_loader, nn.CrossEntropyLoss(ignore_index=0), device)
    final_holdout = evaluate_model(model, holdout_loader, nn.CrossEntropyLoss(ignore_index=0), device)

    print(f"\nFinal Train: loss={final_train['loss']:.4f}, accuracy={final_train['accuracy']:.4f}")
    print(f"Final Holdout: loss={final_holdout['loss']:.4f}, accuracy={final_holdout['accuracy']:.4f}")

    print(f"\nPer-type accuracy:")
    print(f"  {'Type':<15} {'Train':>10} {'Holdout':>10} {'Gap':>10}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10}")
    for aug_type in sorted(final_train['per_type_accuracy'].keys()):
        train_acc = final_train['per_type_accuracy'].get(aug_type, 0)
        holdout_acc = final_holdout['per_type_accuracy'].get(aug_type, 0)
        gap = train_acc - holdout_acc
        print(f"  {aug_type:<15} {train_acc:>10.4f} {holdout_acc:>10.4f} {gap:>+10.4f}")

    # Save metrics
    metrics['final'] = {
        'train': final_train,
        'holdout': final_holdout,
        'best_holdout_loss': best_holdout_loss
    }

    metrics_path = output_path / "training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to: {metrics_path}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced TKS training with reverse oversampling and early stopping',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Locked-in defaults (v3 model settings):
  - Reverse oversample: 2.0x
  - Reverse loss weight: 1.5x
  - Cosine LR with warm restarts
  - Early stop patience: 6 epochs
  - Weight decay: 0.01
        '''
    )
    parser.add_argument('--train', default='output/teacher_rich_v2_train.jsonl',
                       help='Training data JSONL (default: v3 train split)')
    parser.add_argument('--holdout', default='output/teacher_rich_v2_holdout.jsonl',
                       help='Holdout data JSONL (default: v3 holdout split)')
    parser.add_argument('--output-dir', default='output/teacher_model_v3',
                       help='Output directory (default: v3 model dir)')
    parser.add_argument('--epochs', type=int, default=30, help='Max epochs (default: 30)')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size (default: 8)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay (default: 0.01)')
    parser.add_argument('--reverse-oversample', type=float, default=2.0,
                       help='Reverse oversample factor (default: 2.0)')
    parser.add_argument('--reverse-loss-weight', type=float, default=1.5,
                       help='Reverse loss weight (default: 1.5)')
    parser.add_argument('--patience', type=int, default=6, help='Early stopping patience (default: 6)')
    args = parser.parse_args()

    train_model(
        train_path=args.train,
        holdout_path=args.holdout,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        reverse_oversample=args.reverse_oversample,
        reverse_loss_weight=args.reverse_loss_weight,
        patience=args.patience
    )


if __name__ == '__main__':
    main()
