#!/usr/bin/env python3
"""
TKS v7 Syntax-Only Finetune
Targets nested operator syntax without "Trace:" scaffolding.
"""
import argparse
import sys
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tks_llm_core_v7 import TKSGeneralLMv7, TKSGeneralConfigV7

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

DEVICE = "cuda"
if not torch.cuda.is_available():
    print("FATAL ERROR: CUDA not available.")
    sys.exit(1)

print(f"USING GPU: {torch.cuda.get_device_name(0)}")
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


class SyntaxDataset(Dataset):
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
        print(f"Loaded {len(self.data)} syntax samples")

    def _encode(self, text: str):
        enc = self.tokenizer.encode(text)
        ids = enc.ids
        if self.bos_id is not None:
            ids = [self.bos_id] + ids
        if self.eos_id is not None:
            ids = ids + [self.eos_id]
        if len(ids) > self.max_seq_len:
            ids = ids[: self.max_seq_len]
        labels = ids[1:] + [-100]
        return ids, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        full_text = f"{item['input']}\n{item['output']}"
        input_ids, labels = self._encode(full_text)
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
    if os.path.exists(args.checkpoint):
        print(f"Loading weights from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=DEVICE, weights_only=True)
        filtered_state = {
            k: v for k, v in checkpoint.items()
            if "stack_memory" not in k and "stack_ptr" not in k
        }
        model.load_state_dict(filtered_state, strict=False)

    dataset = SyntaxDataset(args.data, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, dataset.pad_id),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    print(f"Starting Syntax Finetune: {len(dataset)} samples")

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

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch+1} | Batch {batch_idx}/{len(loader)} "
                    f"| Loss: {loss.item():.4f}"
                )

        print(f"Epoch {epoch+1} Avg Loss: {total_loss/len(loader):.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "v7_syntax_model.pt")
    torch.save(model.state_dict(), out_path)
    print(f"Syntax model saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/v7_syntax_training.jsonl")
    parser.add_argument("--checkpoint", default="checkpoints/v7_repaired/v7_repaired_model.pt")
    parser.add_argument("--output-dir", default="checkpoints/v7_syntax")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=5e-4)
    train(parser.parse_args())
