#!/usr/bin/env python3
"""
TKS v7 Repair Training - Stage 7
Aggressive fine-tuning to break looping attractors.
"""
import argparse
import sys
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tks_llm_core_v7 import TKSGeneralLMv7, TKSGeneralConfigV7

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

DEVICE = "cuda"
print(f"USING GPU: {torch.cuda.get_device_name(0)}")
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


class RepairDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: Tokenizer, max_seq_len: int = 512):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.token_to_id("<PAD>") or 0
        self.bos_id = tokenizer.token_to_id("<BOS>")
        self.eos_id = tokenizer.token_to_id("<EOS>")

        self.data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        print(f"Loaded {len(self.data)} repair samples")

    def _encode(self, text: str) -> list:
        enc = self.tokenizer.encode(text)
        ids = enc.ids
        if self.bos_id is not None:
            ids = [self.bos_id] + ids
        if self.eos_id is not None:
            ids = ids + [self.eos_id]
        if len(ids) > self.max_seq_len:
            ids = ids[: self.max_seq_len]
        return ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        full_text = item["output"]
        encoded_ids = self._encode(full_text)
        input_ids = encoded_ids[:-1]
        labels = encoded_ids[1:]
        return {"input_ids": input_ids, "labels": labels}


def collate_fn(batch, pad_id=0):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = []
    labels = []
    attention_mask = []
    for item in batch:
        ids = item["input_ids"]
        labs = item["labels"]
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad_len)
        labels.append(labs + [-100] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def train(args):
    tokenizer = Tokenizer.from_file("tokenizer_v5.json")
    vocab_size = tokenizer.get_vocab_size()

    config = TKSGeneralConfigV7(
        vocab_size=vocab_size,
        hidden_dim=256,
        num_layers=6,
        num_scales=4,
        use_dps_gating=False,
        earned_depth_mode=False,
    )

    model = TKSGeneralLMv7(config).to(DEVICE)
    prev_path = "checkpoints/v7_integrated/v7_integrated_model.pt"
    if os.path.exists(prev_path):
        print(f"Loading weights from {prev_path}")
        checkpoint = torch.load(prev_path, map_location=DEVICE, weights_only=True)
        filtered_state = {
            k: v for k, v in checkpoint.items()
            if "stack_memory" not in k and "stack_ptr" not in k
        }
        model.load_state_dict(filtered_state, strict=False)

    dataset = RepairDataset(args.data, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, dataset.pad_id),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    print(f"Starting Stage 7 Repair: {len(dataset)} samples")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                outputs = model(input_ids, attention_mask=mask, return_full_trace=False)
                loss = loss_fn(outputs["logits"].view(-1, vocab_size), labels.view(-1))

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 50 == 0:
                print(
                    f"Epoch {epoch+1} | Batch {batch_idx}/{len(loader)} "
                    f"| Loss: {loss.item():.4f}"
                )

        print(f"Epoch {epoch+1} Avg Loss: {total_loss/len(loader):.4f}")

    os.makedirs("checkpoints/v7_repaired", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/v7_repaired/v7_repaired_model.pt")
    print("Repaired Model saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/v7_repair_training.jsonl")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    train(parser.parse_args())
