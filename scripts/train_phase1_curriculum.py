#!/usr/bin/env python3
"""
Phase 1 Curriculum Training for TKS Model
==========================================

Implements staged curriculum learning for semantic grounding:
- Stage 1: Grounding (ID lookups, definitions) - easiest
- Stage 2: Contrastive (disambiguation)
- Stage 3: TKS Notation (mappings, operations)
- Stage 4: Explanations (interpretations, reasoning)
- Stage 5: Full mixed (all data types)

Each stage builds on the previous, progressively teaching:
1. Element <-> Symbol mapping
2. Disambiguation between similar elements
3. Expression construction
4. Semantic understanding
5. Generalization

Usage:
    python scripts/train_phase1_curriculum.py --data datasets/phase1 --output checkpoints/phase1_curriculum
"""

import sys
import json
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tks_llm_core_v4 import TKSNoeticLM, TKSNoeticLMConfig


# =============================================================================
# CURRICULUM CONFIGURATION
# =============================================================================

@dataclass
class CurriculumStage:
    """Configuration for a curriculum stage."""
    name: str
    types: List[str]  # Data types to include
    epochs: int
    lr: float  # Learning rate for this stage
    warmup_steps: int = 100
    description: str = ""


CURRICULUM_STAGES = [
    CurriculumStage(
        name="Stage 1: Grounding",
        types=["grounding_id_forward", "grounding_id_reverse", "grounding_definition"],
        epochs=5,
        lr=5e-4,
        warmup_steps=50,
        description="Learn element <-> symbol mapping (40 elements)"
    ),
    CurriculumStage(
        name="Stage 2: World & Noetic",
        types=["grounding_world", "grounding_noetic", "contrastive_cross_world"],
        epochs=5,
        lr=3e-4,
        warmup_steps=50,
        description="Learn world structure and noetic levels"
    ),
    CurriculumStage(
        name="Stage 3: Disambiguation",
        types=["contrastive_same_world", "contrastive_choice", "contrastive_opposite", "contrastive_polarity"],
        epochs=5,
        lr=2e-4,
        warmup_steps=50,
        description="Learn to distinguish similar elements"
    ),
    CurriculumStage(
        name="Stage 4: TKS Notation",
        types=["mapping", "goal", "operation"],
        epochs=10,
        lr=2e-4,
        warmup_steps=100,
        description="Learn TKS expression construction"
    ),
    CurriculumStage(
        name="Stage 5: Semantics",
        types=["interpretation", "reasoning"],
        epochs=10,
        lr=1e-4,
        warmup_steps=100,
        description="Learn semantic understanding and reasoning"
    ),
    CurriculumStage(
        name="Stage 6: Dialogue",
        types=["dialogue_qa", "dialogue_followup", "dialogue_followup_multi"],
        epochs=5,
        lr=1e-4,
        warmup_steps=50,
        description="Learn conversational TKS"
    ),
    CurriculumStage(
        name="Stage 7: Full Mix",
        types=None,  # All types
        epochs=10,
        lr=5e-5,
        warmup_steps=200,
        description="Fine-tune on full mixed data"
    ),
]


# =============================================================================
# DATA HANDLING
# =============================================================================

class CurriculumDataset(Dataset):
    """Dataset that filters by data type for curriculum learning."""

    def __init__(
        self,
        data_file: str,
        vocab: Dict[str, int],
        max_len: int = 256,
        types: Optional[List[str]] = None,
        separator: str = " => "
    ):
        self.vocab = vocab
        self.max_len = max_len
        self.separator = separator
        self.samples = []

        # Load data
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line)
                sample_type = sample.get("type", "unknown")

                # Filter by types if specified
                if types is None or sample_type in types:
                    self.samples.append(sample)

        # Shuffle
        random.shuffle(self.samples)

        # Build reverse vocab
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        print(f"  Loaded {len(self.samples)} samples for types: {types or 'ALL'}")

    def __len__(self):
        return len(self.samples)

    def tokenize(self, text: str) -> List[int]:
        """Tokenize text with TKS-aware tokenization."""
        tokens = []
        i = 0

        while i < len(text):
            matched = False

            # Try to match multi-char tokens first (TKS operators, sephirots)
            for length in [3, 2]:  # Try 3-char, then 2-char
                if i + length <= len(text):
                    substr = text[i:i+length]
                    if substr in self.vocab:
                        tokens.append(self.vocab[substr])
                        i += length
                        matched = True
                        break

            if not matched:
                # Single character
                char = text[i]
                if char in self.vocab:
                    tokens.append(self.vocab[char])
                else:
                    tokens.append(self.vocab.get("<UNK>", 1))
                i += 1

        return tokens

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Build full sequence: input + separator + output
        text = sample["input"] + self.separator + sample["output"]

        # Tokenize
        tokens = self.tokenize(text)

        # Truncate if needed
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]

        # Create tensor
        input_ids = torch.tensor(tokens, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "type": sample.get("type", "unknown")
        }


def collate_curriculum(batch):
    """Collate function with variable length padding."""
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids_list = []
    attention_mask_list = []

    for item in batch:
        seq = item["input_ids"]
        seq_len = seq.size(0)

        # Pad to max_len
        if seq_len < max_len:
            padding = torch.zeros(max_len - seq_len, dtype=torch.long)
            seq = torch.cat([seq, padding])

        input_ids_list.append(seq)

        # Attention mask (True = masked/PAD positions)
        mask = torch.zeros(max_len, dtype=torch.bool)
        mask[seq_len:] = True
        attention_mask_list.append(mask)

    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
    }


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_samples(model, vocab, samples: List[Dict], device, separator=" => ") -> Dict:
    """Evaluate model on samples, checking exact match and syntax."""
    model.eval()

    id_to_token = {v: k for k, v in vocab.items()}
    results = defaultdict(lambda: {"total": 0, "exact": 0, "syntax_valid": 0})

    def tokenize(text):
        tokens = []
        i = 0
        while i < len(text):
            matched = False
            for length in [3, 2]:
                if i + length <= len(text):
                    substr = text[i:i+length]
                    if substr in vocab:
                        tokens.append(vocab[substr])
                        i += length
                        matched = True
                        break
            if not matched:
                char = text[i]
                tokens.append(vocab.get(char, vocab.get("<UNK>", 1)))
                i += 1
        return tokens

    def decode(ids):
        return ''.join(id_to_token.get(i, '?') for i in ids if i not in [0, 1, 2, 3])

    with torch.no_grad():
        for sample in samples[:200]:  # Limit for speed
            sample_type = sample.get("type", "unknown")
            prompt = sample["input"] + separator
            expected = sample["output"]

            # Tokenize prompt
            prompt_ids = tokenize(prompt)
            input_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)

            # Generate
            max_gen = 50
            generated_ids = prompt_ids.copy()

            for _ in range(max_gen):
                inp = torch.tensor([generated_ids[-256:]], dtype=torch.long).to(device)
                outputs = model(inp)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                next_token = logits[0, -1, :].argmax().item()

                if next_token == vocab.get("<EOS>", 3) or next_token == 0:
                    break
                generated_ids.append(next_token)

            # Decode output (after prompt)
            output_ids = generated_ids[len(prompt_ids):]
            predicted = decode(output_ids).strip()

            # Check results
            results[sample_type]["total"] += 1
            if predicted == expected:
                results[sample_type]["exact"] += 1

            # Simple syntax check for TKS
            if any(c in predicted for c in "ABCD"):
                results[sample_type]["syntax_valid"] += 1

    return dict(results)


# =============================================================================
# TRAINING
# =============================================================================

def train_stage(
    model: TKSNoeticLM,
    stage: CurriculumStage,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    output_dir: Path,
    stage_idx: int,
) -> Dict:
    """Train model for one curriculum stage."""

    print(f"\n{'='*60}")
    print(f"{stage.name}")
    print(f"{'='*60}")
    print(f"Description: {stage.description}")
    print(f"Types: {stage.types or 'ALL'}")
    print(f"Epochs: {stage.epochs}, LR: {stage.lr}")
    print(f"Training samples: {len(train_loader.dataset)}")

    # Optimizer for this stage
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=stage.lr,
        weight_decay=0.01,
        betas=(0.9, 0.95)
    )

    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * stage.epochs
    warmup_steps = min(stage.warmup_steps, total_steps // 4)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.1, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore PAD

    # Training loop
    model.train()
    best_val_loss = float('inf')
    stage_history = []

    for epoch in range(stage.epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)

            # Language modeling loss (predict next token)
            # Model returns dict with 'logits' key
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs

            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            # Flatten and compute loss
            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / max(num_batches, 1)

        # Validation
        val_loss = 0.0
        if val_loader is not None and len(val_loader) > 0:
            model.eval()
            with torch.no_grad():
                val_batches = 0
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)

                    outputs = model(input_ids, attention_mask=attention_mask)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs

                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = input_ids[:, 1:].contiguous()

                    loss = criterion(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                    val_loss += loss.item()
                    val_batches += 1

                val_loss = val_loss / max(1, val_batches)
            model.train()

        current_lr = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch+1}/{stage.epochs} | Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.2e}")

        stage_history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "lr": current_lr,
        })

        # Save best model for this stage
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stage_checkpoint = output_dir / f"stage_{stage_idx}_best.pt"
            torch.save({
                "stage": stage_idx,
                "stage_name": stage.name,
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
            }, stage_checkpoint)

    return {
        "stage": stage_idx,
        "name": stage.name,
        "final_train_loss": avg_train_loss,
        "best_val_loss": best_val_loss,
        "history": stage_history,
    }


def build_vocabulary() -> Dict[str, int]:
    """Build vocabulary matching the data generator."""
    vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}

    # Add TKS elements (40 sephirots)
    for world in "ABCD":
        for i in range(10):
            vocab[f"{world}{i}"] = len(vocab)

    # Add operators
    for op in ["+", "-", "+T", "-T", "*T", "/T", "->", "<-"]:
        vocab[op] = len(vocab)

    # Add structural tokens
    for s in [":", "(", ")", "^"]:
        vocab[s] = len(vocab)

    # Add separator
    vocab[" => "] = len(vocab)

    # Add noetic levels
    for i in range(10):
        vocab[f"^{i}"] = len(vocab)

    # Add ASCII printable characters
    for c in range(32, 127):
        char = chr(c)
        if char not in vocab:
            vocab[char] = len(vocab)

    # Add common chars
    for char in ['\n', '\t']:
        if char not in vocab:
            vocab[char] = len(vocab)

    return vocab


def run_curriculum_training(args):
    """Run full curriculum training."""

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data)
    train_file = data_dir / "phase1_training_train.jsonl"
    val_file = data_dir / "phase1_training_val.jsonl"

    print("="*60)
    print("PHASE 1 CURRICULUM TRAINING FOR TKS")
    print("="*60)
    print(f"Device: {device}")
    print(f"Data: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Stages: {len(CURRICULUM_STAGES)}")

    # Build vocabulary
    vocab = build_vocabulary()
    print(f"Vocabulary size: {len(vocab)}")

    # Initialize model
    config = TKSNoeticLMConfig(
        vocab_size=len(vocab),
        max_seq_len=args.max_len,
        num_layers=args.n_layers,
        num_scales=3,
        dropout=0.1,
    )
    model = TKSNoeticLM(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Architecture: num_layers={args.n_layers}, max_seq_len={args.max_len}")

    # Optionally load from checkpoint
    start_stage = 0
    if args.resume:
        print(f"Resuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_stage = checkpoint.get("stage", 0) + 1

    # Training loop across stages
    all_results = []

    for stage_idx, stage in enumerate(CURRICULUM_STAGES):
        if stage_idx < start_stage:
            print(f"\nSkipping {stage.name} (already completed)")
            continue

        # Create datasets for this stage
        print(f"\nLoading data for {stage.name}...")
        train_dataset = CurriculumDataset(
            str(train_file),
            vocab,
            max_len=args.max_len,
            types=stage.types,
        )

        val_dataset = CurriculumDataset(
            str(val_file),
            vocab,
            max_len=args.max_len,
            types=stage.types,
        )

        if len(train_dataset) == 0:
            print(f"\nSkipping {stage.name} - no training samples")
            continue

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_curriculum,
            num_workers=0,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_curriculum,
            num_workers=0,
        ) if len(val_dataset) > 0 else None

        # Train this stage
        result = train_stage(
            model=model,
            stage=stage,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            output_dir=output_dir,
            stage_idx=stage_idx,
        )

        all_results.append(result)

        # Save checkpoint after each stage
        checkpoint_path = output_dir / f"checkpoint_stage_{stage_idx}.pt"
        torch.save({
            "stage": stage_idx,
            "stage_name": stage.name,
            "model_state_dict": model.state_dict(),
            "config": config.__dict__,
            "vocab": vocab,
        }, checkpoint_path)
        print(f"  Saved: {checkpoint_path}")

    # Save final model
    final_path = output_dir / "final_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
        "vocab": vocab,
        "stages_completed": len(all_results),
        "results": all_results,
    }, final_path)

    # Save vocabulary
    vocab_path = output_dir / "vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2)

    # Save training log
    log_path = output_dir / "curriculum_log.json"
    with open(log_path, "w") as f:
        json.dump({
            "completed_at": datetime.now().isoformat(),
            "stages": all_results,
            "total_params": total_params,
            "config": {
                "n_layers": args.n_layers,
                "max_seq_len": args.max_len,
                "vocab_size": len(vocab),
            }
        }, f, indent=2)

    print("\n" + "="*60)
    print("CURRICULUM TRAINING COMPLETE")
    print("="*60)

    for result in all_results:
        print(f"  {result['name']}: Train={result['final_train_loss']:.4f}, Val={result['best_val_loss']:.4f}")

    print(f"\nFinal model: {final_path}")
    print(f"Training log: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1 curriculum training for TKS model")
    parser.add_argument("--data", type=str, default="datasets/phase1", help="Data directory")
    parser.add_argument("--output", type=str, default="checkpoints/phase1_curriculum", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--max-len", type=int, default=256, help="Max sequence length")
    parser.add_argument("--n-layers", type=int, default=4, help="Number of noetic layers")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    run_curriculum_training(args)


if __name__ == "__main__":
    main()
