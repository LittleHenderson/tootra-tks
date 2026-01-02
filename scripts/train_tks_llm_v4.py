"""
Train TKS-LLM v4 (no-Transformer, noetic decoder-only LM).

Supports:
- Plain text data (one sample per line)
- Teacher JSONL data (input/target pairs)

Outputs:
- model.pt (state_dict)
- config.json
- tokenizer.json
"""
import argparse
import json
import math
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tks_llm_core_v4 import TKSNoeticLMConfig, TKSNoeticLM


class ByteTokenizer:
    """Byte-level tokenizer with special tokens."""

    def __init__(self):
        self.pad_token = "<PAD>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.sep_token = "<SEP>"
        self.unk_token = "<UNK>"

        self.special_tokens = [self.pad_token, self.bos_token, self.eos_token, self.sep_token, self.unk_token]
        self.special_to_id = {tok: i for i, tok in enumerate(self.special_tokens)}
        self.byte_offset = len(self.special_tokens)
        self.vocab_size = self.byte_offset + 256

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        tokens = []
        if add_bos:
            tokens.append(self.special_to_id[self.bos_token])
        for b in text.encode("utf-8", errors="replace"):
            tokens.append(self.byte_offset + b)
        if add_eos:
            tokens.append(self.special_to_id[self.eos_token])
        return tokens

    def decode(self, ids: List[int]) -> str:
        out = bytearray()
        for tid in ids:
            if tid < self.byte_offset:
                continue
            out.append(tid - self.byte_offset)
        return out.decode("utf-8", errors="replace")

    def to_dict(self) -> Dict[str, object]:
        return {
            "type": "byte",
            "special_tokens": self.special_tokens,
            "vocab_size": self.vocab_size,
        }


class LMTextDataset(Dataset):
    def __init__(self, sequences: List[List[int]], max_length: int, pad_id: int):
        self.sequences = sequences
        self.max_length = max_length
        self.pad_id = pad_id

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        seq = seq[: self.max_length]
        if len(seq) < self.max_length:
            seq = seq + [self.pad_id] * (self.max_length - len(seq))
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        target_ids = torch.tensor(seq[1:], dtype=torch.long)
        return {"input_ids": input_ids, "target_ids": target_ids}


def load_text_lines(path: Path) -> List[str]:
    lines = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip("\n")
            if line.strip():
                lines.append(line)
    return lines


def load_teacher_jsonl(path: Path) -> List[str]:
    lines = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            inp = entry.get("input", "")
            tgt = entry.get("target", "")
            if not inp and not tgt:
                continue
            combined = f"<PROMPT>\n{inp}\n\n<ANSWER>\n{tgt}"
            lines.append(combined)
    return lines


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    tokenizer = ByteTokenizer()

    samples: List[str] = []
    if args.text_data:
        samples.extend(load_text_lines(Path(args.text_data)))
    if args.teacher_jsonl:
        samples.extend(load_teacher_jsonl(Path(args.teacher_jsonl)))

    if not samples:
        raise SystemExit("No training samples loaded.")

    sequences = [tokenizer.encode(s, add_bos=True, add_eos=True) for s in samples]

    dataset = LMTextDataset(sequences, max_length=args.max_length, pad_id=tokenizer.special_to_id[tokenizer.pad_token])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    config = TKSNoeticLMConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.max_length,
        num_layers=args.num_layers,
        num_scales=args.num_scales,
        dropout=args.dropout,
        use_attractor=not args.disable_attractor,
        use_rpm=not args.disable_rpm,
        tie_embeddings=args.tie_embeddings,
    )

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    model = TKSNoeticLM(config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    pad_id = tokenizer.special_to_id[tokenizer.pad_token]
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    total_steps = args.epochs * math.ceil(len(dataset) / args.batch_size)
    step = 0

    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            optimizer.zero_grad()
            out = model(input_ids)
            logits = out["logits"]
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            step += 1
            epoch_loss += loss.item()
            if args.log_every and step % args.log_every == 0:
                print(f"step {step}/{total_steps} loss={loss.item():.4f}")

        avg_loss = epoch_loss / max(1, len(loader))
        print(f"epoch {epoch} avg_loss={avg_loss:.4f}")

        if args.save_every and epoch % args.save_every == 0:
            save_checkpoint(output_dir, model, config, tokenizer, epoch)

    save_checkpoint(output_dir, model, config, tokenizer, args.epochs)


def save_checkpoint(output_dir: Path, model: TKSNoeticLM, config: TKSNoeticLMConfig, tokenizer: ByteTokenizer, epoch: int) -> None:
    ckpt_dir = output_dir / f"epoch_{epoch:03d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "model.pt")
    with (ckpt_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)
    with (ckpt_dir / "tokenizer.json").open("w", encoding="utf-8") as f:
        json.dump(tokenizer.to_dict(), f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TKS-LLM v4 (no-Transformer noetic LM)")
    parser.add_argument("--text-data", type=str, default=None, help="Plain text file (one sample per line)")
    parser.add_argument("--teacher-jsonl", type=str, default=None, help="Teacher JSONL with input/target fields")
    parser.add_argument("--output-dir", type=str, default="output/tks_llm_v4")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-scales", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--disable-attractor", action="store_true")
    parser.add_argument("--disable-rpm", action="store_true")
    parser.add_argument("--tie-embeddings", action="store_true")
    args = parser.parse_args()

    if not args.text_data and not args.teacher_jsonl:
        raise SystemExit("Provide --text-data or --teacher-jsonl")

    train(args)


if __name__ == "__main__":
    main()
