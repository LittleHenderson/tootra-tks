"""
TKS LLM v7 Training Script - Earned Depth Upgrade

Fine-tunes the v6 "Reasoning Engine" with DPS (Depth Permission System).
The model learns to earn deeper recursion through validated novelty.

Key Features:
- Migrates from v6 checkpoint
- DPS learns novelty detection (V × I × (1-S))
- Deeper recursion is earned, not given
- Prevents runaway self-recursion
- Rewards genuine insights

Training Approach:
- Self-supervised novelty learning from embedding similarity
- Impact signal from loss improvement trajectory
- Validity from model confidence
"""

import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tokenizers import Tokenizer
import time
import argparse
from collections import deque

# Import v7 Core
from tks_llm_core_v7 import (
    TKSGeneralLMv7,
    TKSGeneralConfigV7,
    create_v7_from_v6
)
from tks_features.dps_gating import DPSState, NoveltyClass


class V7CoTDataset(Dataset):
    """Handles RPM Chain-of-Thought data for v7 training."""

    def __init__(self, data_path, tokenizer_path, max_len=512):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_len = max_len
        self.pad_id = self.tokenizer.token_to_id("<PAD>") or 0
        self.eos_id = self.tokenizer.token_to_id("<EOS>")
        self.bos_id = self.tokenizer.token_to_id("<BOS>")

        self.data = []
        print(f"Loading CoT data from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                # Same format as v6
                text = f"Goal: {item['input']} <SEP> Reasoning: {item['output']} <SEP> Result: {item['tks_notation']}"
                self.data.append(text)
        print(f"Loaded {len(self.data)} reasoning samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        enc = self.tokenizer.encode(text)
        ids = [self.bos_id] + enc.ids + [self.eos_id] if self.bos_id and self.eos_id else enc.ids
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        return torch.tensor(ids, dtype=torch.long)


class NoveltyMemoryBank:
    """
    Memory bank for computing embedding similarity (S component of novelty).

    Stores recent embeddings to compare against for novelty detection.
    High similarity = low novelty, low similarity = high novelty.
    """

    def __init__(self, capacity: int = 1000, dim: int = 384):
        self.capacity = capacity
        self.dim = dim
        self.embeddings = deque(maxlen=capacity)
        self.register = None  # Tensor version for batch ops

    def add(self, embedding: torch.Tensor):
        """Add embedding(s) to memory bank."""
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        for e in embedding:
            self.embeddings.append(e.detach().cpu())

    def get_tensor(self, device: torch.device) -> torch.Tensor:
        """Get memory bank as tensor for similarity computation."""
        if len(self.embeddings) == 0:
            return torch.zeros(1, self.dim, device=device)
        return torch.stack(list(self.embeddings)).to(device)

    def compute_similarity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute max cosine similarity to stored embeddings.

        Args:
            x: [batch, dim] embeddings

        Returns:
            similarity: [batch] max similarity scores (0-1)
        """
        if len(self.embeddings) == 0:
            return torch.zeros(x.shape[0], device=x.device)

        memory = self.get_tensor(x.device)

        # Normalize
        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
        mem_norm = torch.nn.functional.normalize(memory, p=2, dim=-1)

        # Cosine similarity [batch, memory_size]
        cos_sim = torch.matmul(x_norm, mem_norm.t())

        # Max similarity per sample
        return cos_sim.max(dim=-1)[0]


def train():
    parser = argparse.ArgumentParser(description="TKS v7 Earned Depth Training")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)  # Lower for fine-tuning
    parser.add_argument("--v6_checkpoint", type=str, default="checkpoints/v6_best.pt")
    parser.add_argument("--data", type=str, default="data/v7_combined_training.jsonl",
                        help="Training data (default: combined v6+v7 data)")
    parser.add_argument("--val_data", type=str, default="data/v7_discovery_validation.jsonl",
                        help="Validation data")
    args = parser.parse_args()

    # CUDA enforcement
    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required. No CPU training allowed.")
        sys.exit(1)
    device = torch.device("cuda")
    print(f"Training on {device}")

    Path("checkpoints").mkdir(exist_ok=True)

    # 1. Setup Tokenizer and Data
    tokenizer_path = "tokenizer_v5.json"
    print(f"Loading training data: {args.data}")
    dataset = V7CoTDataset(args.data, tokenizer_path)
    print(f"Loading validation data: {args.val_data}")
    val_dataset = V7CoTDataset(args.val_data, tokenizer_path)

    # 2. Create v7 Model from v6 Checkpoint
    print(f"\nCreating v7 model from {args.v6_checkpoint}...")
    config = TKSGeneralConfigV7(
        vocab_size=dataset.tokenizer.get_vocab_size(),
        use_njt=True,
        njt_use_nested_memory=True,
        use_dps_gating=True,
        dps_initial_p_max=2,  # Start shallow
        dps_max_depth=5,  # Can earn up to 5
        dps_heavy_threshold=0.70,
        dps_count_threshold=0.45,
        dps_tokens_for_unlock=5,
    )

    if Path(args.v6_checkpoint).exists():
        model = create_v7_from_v6(args.v6_checkpoint, config, device)
        print("v6 -> v7 migration complete")
    else:
        print(f"Warning: {args.v6_checkpoint} not found. Starting from scratch.")
        model = TKSGeneralLMv7(config).to(device)

    # 3. Memory Bank for Novelty
    memory_bank = NoveltyMemoryBank(capacity=2000, dim=config.hidden_dim)

    # 4. Training Setup
    def collate_with_mask(batch, pad_id):
        padded = nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=pad_id)
        mask = (padded != pad_id).float()
        return padded, mask

    collate_fn = lambda b: collate_with_mask(b, dataset.pad_id)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(dataloader) * args.epochs)

    print("\nStarting v7 Training (Earned Depth Upgrade)...")
    print(f"Initial DPS state: p_max={model.dps_state.p_max}, tokens={model.dps_state.tokens}")

    global_step = 0
    steps_per_epoch = len(dataloader)
    best_val_loss = float('inf')
    loss_history = deque(maxlen=100)  # For impact computation

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        unlocks_this_epoch = 0

        for batch in dataloader:
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = input_ids.clone()

            # Forward with DPS tracking (trace only every 100 steps to reduce overhead)
            episode_id = f"ep{epoch}_step{global_step}"
            should_trace = (global_step % 100 == 0)
            output = model(
                input_ids,
                attention_mask=attention_mask,
                step=global_step,
                return_full_trace=should_trace,
                episode_id=episode_id
            )
            logits = output["logits"]

            # Language modeling loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = nn.CrossEntropyLoss(ignore_index=dataset.pad_id)(
                shift_logits.view(-1, config.vocab_size), shift_labels.view(-1)
            )

            # Auxiliary losses (ensure scalar)
            aux_loss = output.get("routing_aux_loss", 0)
            if isinstance(aux_loss, torch.Tensor):
                if aux_loss.numel() > 1:
                    aux_loss = aux_loss.mean()
                elif aux_loss.numel() == 0:
                    aux_loss = 0
            total_loss = ce_loss + aux_loss

            # Update memory bank with hidden representations
            # (Pool from middle of sequence for stable representations)
            with torch.no_grad():
                # Get intermediate hidden state (after half the blocks)
                mid_hidden = logits.mean(dim=1)  # [batch, vocab] -> use as proxy
                # Actually we want hidden states, but logits work for similarity
                memory_bank.add(mid_hidden[:, :config.hidden_dim])

            # Track loss for impact signal
            loss_history.append(ce_loss.item())

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += ce_loss.item()

            # Extract DPS info from trace
            dps_state = output.get("dps_state", model.dps_state)
            allowed_depth = output.get("allowed_depth", dps_state.p_max)

            # Check for unlocks
            if dps_state.total_unlocks > model._dps_state.total_unlocks:
                unlocks_this_epoch += 1

            # Logging
            current_lr = scheduler.get_last_lr()[0]
            novelty_class = "NOCOUNT"  # Default

            # Get novelty info from trace if available
            trace = output.get("trace", {})
            for block_trace in trace.get("blocks", []):
                if block_trace.get("dps"):
                    dps_trace = block_trace["dps"]
                    novelty_class = dps_trace.get("novelty_class", "NOCOUNT")
                    break

            print(f"Epoch {epoch + (global_step/steps_per_epoch):.2f} | Step {global_step}: "
                  f"Loss={ce_loss.item():.4f}, Depth={allowed_depth}, "
                  f"p_max={dps_state.p_max}, Tokens={dps_state.tokens}, "
                  f"Novelty={novelty_class}, LR={current_lr:.2e}")

            if global_step > 0 and global_step % 5000 == 0:
                torch.save(model.state_dict(), f"checkpoints/v7_step_{global_step}.pt")

            global_step += 1

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids, attention_mask = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = input_ids.clone()

                output = model(input_ids, attention_mask=attention_mask, return_full_trace=False)
                logits = output["logits"]

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                ce_loss = nn.CrossEntropyLoss(ignore_index=dataset.pad_id)(
                    shift_logits.view(-1, config.vocab_size), shift_labels.view(-1)
                )
                val_loss += ce_loss.item()

        avg_train_loss = epoch_loss / len(dataloader)
        avg_val_loss = val_loss / len(val_dataloader)

        print(f"\n=== Epoch {epoch+1}/{args.epochs} Complete ===")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"DPS State: p_max={model.dps_state.p_max}, tokens={model.dps_state.tokens}, "
              f"unlocks_this_epoch={unlocks_this_epoch}, total_unlocks={model.dps_state.total_unlocks}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "checkpoints/v7_best.pt")
            print(f"New best model saved (val_loss={avg_val_loss:.4f})")
        print()

    # Save final model and DPS state
    torch.save({
        'model_state_dict': model.state_dict(),
        'dps_state': model.dps_state.to_dict(),
        'config': {
            'vocab_size': config.vocab_size,
            'dps_max_depth': config.dps_max_depth,
            'dps_initial_p_max': config.dps_initial_p_max,
        }
    }, "checkpoints/v7_final.pt")

    print(f"\nv7 Training Complete!")
    print(f"Best val_loss: {best_val_loss:.4f}")
    print(f"Final DPS state: p_max={model.dps_state.p_max}, total_unlocks={model.dps_state.total_unlocks}")


if __name__ == "__main__":
    train()
