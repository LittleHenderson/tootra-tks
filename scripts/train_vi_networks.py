#!/usr/bin/env python3
"""
Train V (Validity) and I (Impact) Networks for DPS

This trains the novelty component networks so that:
- V outputs HIGH for valid reasoning, LOW for invalid
- I outputs HIGH for impactful thoughts, LOW for trivial

Once trained, NW = V x I x (1-S) will reflect GENUINE novelty,
not just random values, enabling TRUE earned depth.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

if not torch.cuda.is_available():
    print("FATAL ERROR: GPU required for training.")
    sys.exit(1)

DEVICE = 'cuda'
print(f"USING GPU: {torch.cuda.get_device_name(0)}")

# =============================================================================
# TEXT ENCODER (Simple character-level for now)
# =============================================================================

class SimpleTextEncoder(nn.Module):
    """Encode text to fixed-size embedding for V/I training."""

    def __init__(self, vocab_size: int = 256, embed_dim: int = 64, hidden_dim: int = 128, output_dim: int = 40):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len] character indices
        Returns:
            [batch, output_dim] text embedding
        """
        emb = self.embed(x)  # [batch, seq, embed_dim]
        _, (h, _) = self.lstm(emb)  # h: [2, batch, hidden]
        h = torch.cat([h[0], h[1]], dim=-1)  # [batch, hidden*2]
        return self.proj(h)  # [batch, output_dim]


# =============================================================================
# DATASETS
# =============================================================================

class ValidityDataset(Dataset):
    """Dataset for training V (Validity) network."""

    def __init__(self, data_path: str, max_seq_len: int = 256):
        self.max_seq_len = max_seq_len
        self.data = []

        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

        print(f"Loaded {len(self.data)} validity examples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']

        # Character-level encoding
        ids = [ord(c) % 256 for c in text[:self.max_seq_len]]
        ids = ids + [0] * (self.max_seq_len - len(ids))  # Pad

        # Labels: [confidence, consistency, evidence_ok]
        labels = torch.tensor([
            item['confidence'],
            item['consistency'],
            item['evidence_ok'],
        ], dtype=torch.float32)

        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'labels': labels,
            'valid': item['valid'],
        }


class ImpactDataset(Dataset):
    """Dataset for training I (Impact) network."""

    def __init__(self, data_path: str, max_seq_len: int = 256):
        self.max_seq_len = max_seq_len
        self.data = []

        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

        print(f"Loaded {len(self.data)} impact examples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']

        # Character-level encoding
        ids = [ord(c) % 256 for c in text[:self.max_seq_len]]
        ids = ids + [0] * (self.max_seq_len - len(ids))  # Pad

        # Labels: [delta_reward, delta_uncertainty, delta_coverage, delta_cost]
        labels = torch.tensor([
            item['delta_reward'],
            item['delta_uncertainty'],
            item['delta_coverage'],
            item['delta_cost'],
        ], dtype=torch.float32)

        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'labels': labels,
            'is_breakthrough': item['is_breakthrough'],
        }


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_validity_network(
    data_path: str,
    output_path: str,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    validity_dim: int = 32,  # Match DPSGatingLayer config
    noetic_dim: int = 40,
):
    """Train the V (Validity) network."""
    print("\n" + "="*50)
    print("TRAINING V (VALIDITY) NETWORK")
    print("="*50)

    # Load data
    dataset = ValidityDataset(data_path)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    # Build model: TextEncoder -> ValidityNet
    # Architecture MUST match DPSGatingLayer.validity_net exactly
    encoder = SimpleTextEncoder(output_dim=noetic_dim).to(DEVICE)

    validity_net = nn.Sequential(
        nn.Linear(noetic_dim, validity_dim),
        nn.LayerNorm(validity_dim),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(validity_dim, validity_dim),
        nn.GELU(),
        nn.Linear(validity_dim, 3),  # confidence, consistency, evidence_ok
        nn.Sigmoid(),
    ).to(DEVICE)

    # Combined parameters
    params = list(encoder.parameters()) + list(validity_net.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Train
        encoder.train()
        validity_net.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            valid = batch['valid']

            optimizer.zero_grad()

            # Forward
            embedding = encoder(input_ids)
            pred = validity_net(embedding)

            loss = loss_fn(pred, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Accuracy: avg pred > 0.5 should match valid=True
            pred_valid = (pred.mean(dim=-1) > 0.5).cpu()
            correct += (pred_valid == torch.tensor(valid)).sum().item()
            total += len(valid)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        # Validate
        encoder.eval()
        validity_net.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                valid = batch['valid']

                embedding = encoder(input_ids)
                pred = validity_net(embedding)
                val_loss += loss_fn(pred, labels).item()

                pred_valid = (pred.mean(dim=-1) > 0.5).cpu()
                val_correct += (pred_valid == torch.tensor(valid)).sum().item()
                val_total += len(valid)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2%}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best model
            torch.save({
                'encoder': encoder.state_dict(),
                'validity_net': validity_net.state_dict(),
            }, output_path)

    print(f"\nSaved V network to {output_path}")
    return encoder, validity_net


def train_impact_network(
    data_path: str,
    output_path: str,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    impact_dim: int = 32,  # Match DPSGatingLayer config
    noetic_dim: int = 40,
):
    """Train the I (Impact) network."""
    print("\n" + "="*50)
    print("TRAINING I (IMPACT) NETWORK")
    print("="*50)

    # Load data
    dataset = ImpactDataset(data_path)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    # Build model: TextEncoder -> ImpactNet
    # Architecture MUST match DPSGatingLayer.impact_net exactly
    encoder = SimpleTextEncoder(output_dim=noetic_dim).to(DEVICE)

    # Impact network expects noetic_dim + 4 context signals
    impact_net = nn.Sequential(
        nn.Linear(noetic_dim + 4, impact_dim),
        nn.LayerNorm(impact_dim),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(impact_dim, impact_dim),
        nn.GELU(),
        nn.Linear(impact_dim, 4),  # delta_reward, delta_uncertainty, delta_coverage, delta_cost
        nn.Sigmoid(),
    ).to(DEVICE)

    # Combined parameters
    params = list(encoder.parameters()) + list(impact_net.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Train
        encoder.train()
        impact_net.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            is_breakthrough = batch['is_breakthrough']

            optimizer.zero_grad()

            # Forward
            embedding = encoder(input_ids)
            # Add dummy context signals (zeros for now)
            context = torch.zeros(embedding.shape[0], 4, device=DEVICE)
            impact_input = torch.cat([embedding, context], dim=-1)
            pred = impact_net(impact_input)

            loss = loss_fn(pred, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Accuracy: high avg pred should match is_breakthrough=True
            pred_breakthrough = (pred.mean(dim=-1) > 0.4).cpu()
            correct += (pred_breakthrough == torch.tensor(is_breakthrough)).sum().item()
            total += len(is_breakthrough)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        # Validate
        encoder.eval()
        impact_net.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                is_breakthrough = batch['is_breakthrough']

                embedding = encoder(input_ids)
                context = torch.zeros(embedding.shape[0], 4, device=DEVICE)
                impact_input = torch.cat([embedding, context], dim=-1)
                pred = impact_net(impact_input)
                val_loss += loss_fn(pred, labels).item()

                pred_breakthrough = (pred.mean(dim=-1) > 0.4).cpu()
                val_correct += (pred_breakthrough == torch.tensor(is_breakthrough)).sum().item()
                val_total += len(is_breakthrough)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2%}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best model
            torch.save({
                'encoder': encoder.state_dict(),
                'impact_net': impact_net.state_dict(),
            }, output_path)

    print(f"\nSaved I network to {output_path}")
    return encoder, impact_net


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--validity-data', default='data/dps_training/validity_training.jsonl')
    parser.add_argument('--impact-data', default='data/dps_training/impact_training.jsonl')
    parser.add_argument('--output-dir', default='checkpoints/dps')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Train V network
    train_validity_network(
        data_path=args.validity_data,
        output_path=f"{args.output_dir}/validity_net.pt",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # Train I network
    train_impact_network(
        data_path=args.impact_data,
        output_path=f"{args.output_dir}/impact_net.pt",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    print("\n" + "="*50)
    print("V/I NETWORK TRAINING COMPLETE")
    print("="*50)
    print(f"Validity network: {args.output_dir}/validity_net.pt")
    print(f"Impact network: {args.output_dir}/impact_net.pt")
    print("\nNext steps:")
    print("1. Integrate trained weights into DPSGatingLayer")
    print("2. Run depth verification benchmark")
    print("3. Observe NW values during TKS training")


if __name__ == "__main__":
    main()
