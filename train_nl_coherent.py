#!/usr/bin/env python3
"""
TKS LLM Natural Language Coherent Training Script

Purpose: Train the model to output coherent paragraph answers to questions.

Data Sources:
- coherence_training_nl.jsonl (27MB) - Coherent paragraph interpretations
- nl_readable_training.jsonl (23MB) - Prompt/response paragraph pairs
- v7_nl_bridge_training.jsonl (1.3MB) - Q&A format NL bridges
- coherence_training_stories.jsonl (4MB) - Story-based coherence

*** GPU REQUIRED - CPU TRAINING IS DISABLED ***
This script will EXIT if no CUDA GPU is available.
"""

import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tokenizers import Tokenizer
import time
import random
import argparse

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)

# =============================================================================
# GPU ENFORCEMENT
# =============================================================================
def _enforce_gpu_required():
    """Enforce GPU requirement. Called immediately on script load."""
    if not torch.cuda.is_available():
        print("\n" + "=" * 70)
        print("FATAL ERROR: CUDA GPU NOT AVAILABLE - TRAINING BLOCKED")
        print("=" * 70)
        print("")
        print("  This training script REQUIRES a CUDA-capable GPU.")
        print("  CPU training is DISABLED.")
        print("")
        print("=" * 70)
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU Verified: {gpu_name} ({gpu_mem:.1f} GB)")
    return torch.device("cuda")

# ENFORCE GPU NOW
DEVICE = _enforce_gpu_required()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# =============================================================================
# CONFIGURATION
# =============================================================================
BATCH_SIZE = 8
EPOCHS = 3
LR = 5e-5  # Fine-tuning LR
MAX_SEQ_LEN = 512  # Longer for paragraphs
GRAD_ACCUM_STEPS = 4
SAVE_EVERY = 2000
LOG_EVERY = 100

# NL Training Data Files
NL_DATA_FILES = [
    ("data/nl_readable_training.jsonl", "nl_readable"),  # Main NL paragraphs
    ("data/coherence_training_nl.jsonl", "coherence_nl"),  # Coherent interpretations
    ("data/v7_nl_bridge_training.jsonl", "nl_bridge"),  # Q&A bridges
    ("data/coherence_training_stories.jsonl", "stories"),  # Stories
]


# =============================================================================
# DATASET
# =============================================================================
class NLCoherentDataset(Dataset):
    """
    Dataset for NL coherent paragraph training.

    Supports multiple data formats:
    - {"prompt": "...", "response": "..."} -> "Q: ... <SEP> A: ..."
    - {"input": "...", "output": "..."} -> "Q: ... <SEP> A: ..."
    - {"equation": "...", "interpretation": "..."} -> "Explain: ... <SEP> ..."
    - {"text": "..."} -> raw text
    """

    def __init__(self, data_paths, tokenizer_path, max_len=MAX_SEQ_LEN, max_samples=None):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_len = max_len
        self.pad_id = self.tokenizer.token_to_id("<PAD>") or 0
        self.eos_id = self.tokenizer.token_to_id("<EOS>")
        self.bos_id = self.tokenizer.token_to_id("<BOS>")
        self.sep_id = self.tokenizer.token_to_id("<SEP>")

        self.data = []

        for data_path, data_type in data_paths:
            if not Path(data_path).exists():
                print(f"  [SKIP] {data_path} not found")
                continue

            count = 0
            print(f"  Loading {data_path}...")

            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                        text = self._format_item(item, data_type)
                        if text and len(text) > 20:  # Filter very short
                            self.data.append(text)
                            count += 1

                            if max_samples and count >= max_samples:
                                break
                    except:
                        pass

            print(f"    Loaded {count} samples from {data_type}")

        # Shuffle data
        random.shuffle(self.data)
        print(f"Total NL training samples: {len(self.data)}")

    def _format_item(self, item, data_type):
        """Format item into training text."""

        if data_type == "nl_readable":
            # {"prompt": "Explain: A7 + B6", "response": "...paragraph..."}
            if "prompt" in item and "response" in item:
                prompt = item["prompt"]
                response = item["response"]
                return f"Question: {prompt} <SEP> Answer: {response}"

        elif data_type == "coherence_nl":
            # {"equation": "A7 + B6", "interpretation": "...long paragraph..."}
            if "equation" in item and "interpretation" in item:
                eq = item["equation"]
                interp = item["interpretation"]
                return f"Explain the meaning of {eq}: <SEP> {interp}"

        elif data_type == "nl_bridge":
            # {"input": "What does Noetic 1 mean?", "output": "...answer..."}
            if "input" in item and "output" in item:
                q = item["input"]
                a = item["output"]
                return f"Question: {q} <SEP> Answer: {a}"

        elif data_type == "stories":
            # {"interpretation": "...story paragraph..."}
            if "interpretation" in item:
                return f"Story: <SEP> {item['interpretation']}"
            elif "original_text" in item:
                return f"Story: <SEP> {item['original_text']}"

        # Fallback for generic formats
        if "text" in item:
            return item["text"]
        if "prompt" in item and "target" in item:
            return f"Question: {item['prompt']} <SEP> Answer: {item['target']}"
        if "input" in item and "output" in item:
            return f"Question: {item['input']} <SEP> Answer: {item['output']}"

        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        enc = self.tokenizer.encode(text)

        ids = enc.ids
        if self.bos_id is not None:
            ids = [self.bos_id] + ids
        if self.eos_id is not None:
            ids = ids + [self.eos_id]

        if len(ids) > self.max_len:
            ids = ids[:self.max_len]

        return torch.tensor(ids, dtype=torch.long)


def collate_fn(batch):
    """Collate batch with padding."""
    max_len = max(len(x) for x in batch)
    padded = []
    for x in batch:
        pad_len = max_len - len(x)
        if pad_len > 0:
            x = torch.cat([x, torch.zeros(pad_len, dtype=torch.long)])
        padded.append(x)
    return torch.stack(padded)


# =============================================================================
# TRAINING LOOP
# =============================================================================
def train():
    parser = argparse.ArgumentParser(description="TKS NL Coherent Training")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--model", type=str, default="v7", choices=["v6", "v7", "v7c"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint to load")
    args = parser.parse_args()

    print("=" * 70)
    print("TKS NL COHERENT PARAGRAPH TRAINING")
    print("=" * 70)

    # Load tokenizer
    tokenizer_path = "tokenizer_v5.json"
    if not Path(tokenizer_path).exists():
        print(f"ERROR: {tokenizer_path} not found")
        return

    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer: {vocab_size} tokens")

    # Load dataset
    print("\nLoading NL training data...")
    dataset = NLCoherentDataset(NL_DATA_FILES, tokenizer_path, max_len=MAX_SEQ_LEN)

    if len(dataset) == 0:
        print("ERROR: No training data loaded")
        return

    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    # Load model
    print(f"\nLoading model: {args.model}")

    if args.model == "v7c":
        # v7-Coherent (new integrated model)
        from tks_llm_core_v7_coherent import TKSCoherentConfigV7, TKSGeneralLMv7Coherent
        config = TKSCoherentConfigV7(
            vocab_size=vocab_size,
            hidden_dim=256,  # Match existing checkpoints
            num_layers=12,
            use_coherence_tracking=True,
            coherence_gate_enabled=True,
            use_dps_gating=True,
        )
        model = TKSGeneralLMv7Coherent(config)
        checkpoint_default = "checkpoints/v7_foundation/v7_foundation_best.pt"

    elif args.model == "v7":
        # Original v7
        from tks_llm_core_v7 import TKSGeneralConfigV7, TKSGeneralLMv7
        config = TKSGeneralConfigV7(
            vocab_size=vocab_size,
            hidden_dim=256,
            num_layers=12,
            use_dps_gating=True,
            use_regulators=False,
        )
        model = TKSGeneralLMv7(config)
        checkpoint_default = "checkpoints/v7_foundation/v7_foundation_best.pt"

    else:  # v6
        from tks_llm_core_v6 import TKSGeneralConfig, TKSGeneralLMv6
        config = TKSGeneralConfig(
            vocab_size=vocab_size,
            hidden_dim=256,
            num_layers=12,
        )
        model = TKSGeneralLMv6(config)
        checkpoint_default = "checkpoints/v6_best.pt"

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Load checkpoint
    checkpoint_path = args.checkpoint or checkpoint_default
    if Path(checkpoint_path).exists():
        print(f"Loading weights from {checkpoint_path}...")
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            model_state = model.state_dict()

            # Filter compatible weights
            filtered = {}
            for k, v in state_dict.items():
                if k in model_state and model_state[k].shape == v.shape:
                    filtered[k] = v

            model.load_state_dict(filtered, strict=False)
            print(f"Loaded {len(filtered)}/{len(state_dict)} weights")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
    else:
        print(f"No checkpoint found at {checkpoint_path}, training from scratch")

    model = model.to(DEVICE)
    model.train()

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps = len(loader) * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)

    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr / 10)

    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding

    # Training
    print(f"\nStarting NL coherent training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Total steps: {total_steps}")
    print(f"  Gradient accumulation: {GRAD_ACCUM_STEPS}")

    global_step = 0
    best_loss = float("inf")
    start_time = time.time()
    accum_loss = 0.0

    # Create checkpoint dir
    checkpoint_dir = Path(f"checkpoints/nl_coherent_{args.model}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for batch_idx, batch in enumerate(loader):
            batch = batch.to(DEVICE)

            # Forward pass
            try:
                outputs = model(batch[:, :-1], step=global_step)
                logits = outputs["logits"]
            except Exception as e:
                print(f"Forward error: {e}")
                continue

            # Compute loss
            targets = batch[:, 1:]
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            # Add auxiliary losses if available
            if "routing_aux_loss" in outputs and outputs["routing_aux_loss"] is not None:
                aux_loss = outputs["routing_aux_loss"]
                if isinstance(aux_loss, torch.Tensor):
                    loss = loss + 0.01 * aux_loss.mean()

            # Backward with gradient accumulation
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()

            accum_loss += loss.item()

            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                epoch_loss += accum_loss * GRAD_ACCUM_STEPS
                epoch_steps += 1

                # Log
                if global_step % LOG_EVERY == 0:
                    avg_loss = epoch_loss / epoch_steps
                    elapsed = time.time() - start_time
                    lr = scheduler.get_last_lr()[0]
                    print(f"  Step {global_step} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | Time: {elapsed:.0f}s")

                # Save checkpoint
                if global_step % SAVE_EVERY == 0:
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        save_path = checkpoint_dir / f"nl_coherent_best.pt"
                        torch.save(model.state_dict(), save_path)
                        print(f"  Saved best model to {save_path}")

                    save_path = checkpoint_dir / f"nl_coherent_step{global_step}.pt"
                    torch.save(model.state_dict(), save_path)

                accum_loss = 0.0

                if args.max_steps and global_step >= args.max_steps:
                    break

        # End of epoch
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        print(f"\nEpoch {epoch + 1}/{args.epochs} complete | Avg Loss: {avg_epoch_loss:.4f}")

        # Save epoch checkpoint
        save_path = checkpoint_dir / f"nl_coherent_epoch{epoch + 1}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"Saved epoch checkpoint to {save_path}")

        if args.max_steps and global_step >= args.max_steps:
            break

    # Final save
    save_path = checkpoint_dir / "nl_coherent_final.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining complete! Final model saved to {save_path}")

    # Quick test
    print("\n" + "=" * 70)
    print("QUICK INFERENCE TEST")
    print("=" * 70)

    model.eval()
    test_prompts = [
        "Question: What is spiritual wisdom? <SEP> Answer:",
        "Explain the meaning of A7 + B6: <SEP>",
    ]

    for prompt in test_prompts:
        enc = tokenizer.encode(prompt)
        input_ids = [tokenizer.token_to_id("<BOS>")] + enc.ids
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)

        generated = input_ids.copy()
        with torch.no_grad():
            for _ in range(100):
                out = model(torch.tensor([generated], dtype=torch.long).to(DEVICE))
                next_token = out["logits"][0, -1, :].argmax().item()
                generated.append(next_token)
                if next_token == tokenizer.token_to_id("<EOS>"):
                    break

        decoded = tokenizer.decode(generated)
        decoded_safe = decoded.encode('ascii', 'replace').decode('ascii')
        print(f"\nPrompt: {prompt[:50]}...")
        print(f"Output: {decoded_safe[:200]}...")


if __name__ == "__main__":
    train()
