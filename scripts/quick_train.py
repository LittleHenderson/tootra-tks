"""
Quick training script for Phase 5 - simplified to avoid LSTM output issues
"""
import json
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Simple tokenizer
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

        # Add characters
        for c in ' .,!?;:\'"()-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789':
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

# Simple dataset
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

# Simple transformer model
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

def train_model(data_path, output_dir, epochs=2, batch_size=4, lr=1e-3):
    print("=" * 70)
    print("PHASE 5 - TKS TRAINING WITH AUGMENTED DATA")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Setup
    tokenizer = SimpleTokenizer(max_length=256)
    dataset = TKSDataset(data_path, tokenizer)
    print(f"Loaded {len(dataset)} training samples")

    # Count augmentation types
    aug_counts = {}
    for entry in dataset.entries:
        aug_type = entry.get('aug_type', 'unknown')
        aug_counts[aug_type] = aug_counts.get(aug_type, 0) + 1

    print(f"\nAugmentation distribution:")
    for aug_type, count in sorted(aug_counts.items()):
        print(f"  {aug_type}: {count}")

    # Split data
    eval_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - eval_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    print(f"\nTrain size: {train_size}")
    print(f"Eval size: {eval_size}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = SimpleTransformer(tokenizer.actual_vocab_size, hidden_dim=128, num_layers=2)
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")

    # Setup training
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    print(f"\nTraining configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Output dir: {output_dir}")

    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    metrics = {
        'epoch_losses': [],
        'eval_losses': [],
        'augmentation': aug_counts,
    }

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 70)

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if batch_idx % 1 == 0:
                print(f"  Step {batch_idx}/{len(train_loader)}: loss={loss.item():.4f}")

        avg_loss = epoch_loss / max(num_batches, 1)
        metrics['epoch_losses'].append({'epoch': epoch + 1, 'loss': avg_loss})
        print(f"\n  Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        # Evaluate
        model.eval()
        eval_loss = 0.0
        eval_batches = 0

        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(device)
                targets = batch['targets'].to(device)

                logits = model(input_ids)
                loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
                eval_loss += loss.item()
                eval_batches += 1

        avg_eval_loss = eval_loss / max(eval_batches, 1)
        metrics['eval_losses'].append({'epoch': epoch + 1, 'loss': avg_eval_loss})
        print(f"  Eval loss: {avg_eval_loss:.4f}")

    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / "final_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nSaved model to: {model_path}")

    # Save metrics
    metrics_path = output_path / "training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nFinal train loss: {metrics['epoch_losses'][-1]['loss']:.4f}")
    print(f"Final eval loss: {metrics['eval_losses'][-1]['loss']:.4f}")

    return metrics

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output-dir', default='output/phase5_models')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=1e-3)

    args = parser.parse_args()
    train_model(args.data, args.output_dir, args.epochs, args.batch_size, args.learning_rate)
