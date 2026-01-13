"""
Train Coherence Model on Multi-Style TKS Data

This script trains the NoeticFractalEncoder and coherence scoring system
to recognize coherent TKS patterns across multiple styles/registers.

The model learns:
1. What coherent TKS verbiage looks like (across 8 styles)
2. How to distinguish coherent from gibberish
3. Style-invariant coherence features

Author: TKS Research Pipeline
Date: 2026-01-13
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import argparse
from tqdm import tqdm
import random

from tks_features.noetic_fractal_coherence import NoeticFractalEncoder
from tks_features.lacunary_pattern_flow import CanonicalLacunarity, TextureTuple


class CoherenceDataset(Dataset):
    """Dataset for coherence training."""

    def __init__(self, data_path: str, tokenizer=None, max_length: int = 256):
        self.entries = []
        self.max_length = max_length

        # Simple character-level tokenization if no tokenizer provided
        self.char_to_id = {chr(i): i for i in range(128)}
        self.char_to_id['<PAD>'] = 128
        self.char_to_id['<UNK>'] = 129

        print(f"Loading data from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    self.entries.append(entry)
                except json.JSONDecodeError:
                    continue

        print(f"Loaded {len(self.entries)} entries")

        # Count coherent vs incoherent
        coherent = sum(1 for e in self.entries if e.get('is_coherent', True))
        print(f"  Coherent: {coherent}, Incoherent: {len(self.entries) - coherent}")

    def __len__(self):
        return len(self.entries)

    def tokenize(self, text: str) -> torch.Tensor:
        """Simple character-level tokenization."""
        ids = []
        for char in text[:self.max_length]:
            ids.append(self.char_to_id.get(char, self.char_to_id['<UNK>']))

        # Pad
        while len(ids) < self.max_length:
            ids.append(self.char_to_id['<PAD>'])

        return torch.tensor(ids[:self.max_length], dtype=torch.long)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        # Get text
        text = entry.get('story', entry.get('interpretation', ''))

        # Tokenize
        tokens = self.tokenize(text)

        # Get label (1 = coherent, 0 = incoherent)
        label = 1.0 if entry.get('is_coherent', True) else 0.0

        # Get style (for analysis)
        style = entry.get('style', 'unknown')

        return {
            'tokens': tokens,
            'label': torch.tensor(label, dtype=torch.float32),
            'style': style,
            'text': text[:100],  # Truncated for debugging
        }


class CoherenceClassifier(nn.Module):
    """
    Coherence classifier that uses NoeticFractalEncoder features
    combined with learned coherence patterns and n-gram features.
    """

    def __init__(self, vocab_size: int = 130, embed_dim: int = 64,
                 hidden_dim: int = 128, noetic_dim: int = 40):
        super().__init__()

        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=128)

        # N-gram convolutions for local pattern detection (catches word shuffling)
        self.ngram_convs = nn.ModuleList([
            nn.Conv1d(embed_dim, embed_dim // 2, kernel_size=k, padding=k // 2)
            for k in [2, 3, 4, 5]  # bigram, trigram, 4-gram, 5-gram
        ])
        ngram_features = (embed_dim // 2) * 4  # 4 n-gram sizes

        # Text encoder (simple transformer-like)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=4,
                dim_feedforward=hidden_dim,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )

        # NF-inspired coherence features
        self.nf_encoder = NoeticFractalEncoder(
            embed_dim=embed_dim,
            noetic_dim=noetic_dim
        )

        # Coherence head (now includes n-gram features)
        # embed_dim + noetic_dim + 3 (lacunarity) + ngram_features
        total_features = embed_dim + noetic_dim + 3 + ngram_features
        self.coherence_head = nn.Sequential(
            nn.Linear(total_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Embed tokens
        x = self.embedding(tokens)  # [B, L, E]

        # Compute n-gram features (catches word shuffling patterns)
        x_conv = x.transpose(1, 2)  # [B, E, L] for conv1d
        ngram_outputs = []
        for conv in self.ngram_convs:
            conv_out = F.relu(conv(x_conv))  # [B, E//2, L]
            # Global max pooling over sequence
            ngram_pooled = conv_out.max(dim=2)[0]  # [B, E//2]
            ngram_outputs.append(ngram_pooled)
        ngram_features = torch.cat(ngram_outputs, dim=-1)  # [B, E//2 * 4]

        # Encode sequence
        encoded = self.encoder(x)  # [B, L, E]

        # Pool to get sequence representation
        # Use mean pooling (ignoring padding)
        mask = (tokens != 128).float().unsqueeze(-1)  # [B, L, 1]
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [B, E]

        # Get NF features (use full pooled embedding)
        nf_output, nf_validity, nf_coords = self.nf_encoder(pooled, return_nf=True)

        # Compute lacunarity features from encoded sequence
        lacunarity = self._compute_lacunarity_features(encoded, mask)

        # Combine all features: pooled + NF + lacunarity + n-grams
        features = torch.cat([pooled, nf_output, lacunarity, ngram_features], dim=-1)

        # Predict coherence
        coherence_score = self.coherence_head(features)

        return {
            'coherence': coherence_score.squeeze(-1),
            'nf_validity': nf_validity,
            'nf_coords': nf_coords,
            'lacunarity_features': lacunarity,
            'ngram_features': ngram_features,
        }

    def _compute_lacunarity_features(self, encoded: torch.Tensor,
                                     mask: torch.Tensor) -> torch.Tensor:
        """Compute lacunarity-based features from encoded sequence."""
        batch_size = encoded.shape[0]
        features = []

        for b in range(batch_size):
            seq = encoded[b]  # [L, E]
            seq_mask = mask[b].squeeze(-1)  # [L]

            # Get valid positions
            valid_len = int(seq_mask.sum().item())
            if valid_len < 2:
                features.append(torch.tensor([1.0, 0.5, 0.5], device=encoded.device))
                continue

            valid_seq = seq[:valid_len]

            # Compute lacunarity
            lac = CanonicalLacunarity.compute_lacunarity(valid_seq)

            # Compute variance of norms (measure of consistency)
            norms = torch.norm(valid_seq, dim=-1)
            norm_var = norms.var().item() if valid_len > 1 else 0.0

            # Compute sequential coherence (similarity of adjacent tokens)
            if valid_len > 1:
                diffs = (valid_seq[1:] - valid_seq[:-1]).norm(dim=-1)
                seq_coherence = 1.0 / (1.0 + diffs.mean().item())
            else:
                seq_coherence = 1.0

            features.append(torch.tensor([lac, norm_var, seq_coherence], device=encoded.device))

        return torch.stack(features)


def train_epoch(model: nn.Module, dataloader: DataLoader,
                optimizer: torch.optim.Optimizer, device: str) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Training"):
        tokens = batch['tokens'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(tokens)
        predictions = outputs['coherence']

        # Binary cross entropy loss
        loss = F.binary_cross_entropy(predictions, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Accuracy
        predicted_labels = (predictions > 0.5).float()
        correct += (predicted_labels == labels).sum().item()
        total += labels.shape[0]

    return {
        'loss': total_loss / len(dataloader),
        'accuracy': correct / total if total > 0 else 0,
    }


def evaluate(model: nn.Module, dataloader: DataLoader, device: str) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    style_correct = {}
    style_total = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            tokens = batch['tokens'].to(device)
            labels = batch['label'].to(device)
            styles = batch['style']

            outputs = model(tokens)
            predictions = outputs['coherence']

            loss = F.binary_cross_entropy(predictions, labels)
            total_loss += loss.item()

            predicted_labels = (predictions > 0.5).float()
            correct += (predicted_labels == labels).sum().item()
            total += labels.shape[0]

            # Per-style accuracy
            for i, style in enumerate(styles):
                if style not in style_correct:
                    style_correct[style] = 0
                    style_total[style] = 0
                style_total[style] += 1
                if predicted_labels[i] == labels[i]:
                    style_correct[style] += 1

    # Compute per-style accuracy
    style_accuracy = {s: style_correct[s] / style_total[s]
                      for s in style_correct if style_total[s] > 0}

    return {
        'loss': total_loss / len(dataloader),
        'accuracy': correct / total if total > 0 else 0,
        'style_accuracy': style_accuracy,
    }


def main():
    parser = argparse.ArgumentParser(description="Train coherence classifier")
    parser.add_argument("--data", "-d", default="data/coherence_training_stories.jsonl",
                        help="Training data file")
    parser.add_argument("--epochs", "-e", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--output", "-o", default="checkpoints/coherence_classifier.pt",
                        help="Output model path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")

    args = parser.parse_args()

    print(f"Using device: {args.device}")

    # Load data
    dataset = CoherenceDataset(args.data)

    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Create model
    model = CoherenceClassifier().to(args.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop
    best_val_acc = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_metrics = train_epoch(model, train_loader, optimizer, args.device)
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")

        val_metrics = evaluate(model, val_loader, args.device)
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")

        # Print per-style accuracy
        print("Style accuracy:")
        for style, acc in sorted(val_metrics['style_accuracy'].items()):
            marker = "x" if "gibberish" in style else "o"
            print(f"  [{marker}] {style}: {acc:.3f}")

        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_accuracy': best_val_acc,
                'epoch': epoch,
            }, args.output)
            print(f"Saved best model (acc: {best_val_acc:.4f})")

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()
