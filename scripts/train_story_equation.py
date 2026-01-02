"""
Training script for bidirectional story↔equation translation (Track 2 Agent G)
Handles both story_to_equation and equation_to_story tasks
"""
import json
import random
import re
import sys
import time
import math
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Simple tokenizer for seq2seq
class Seq2SeqTokenizer:
    def __init__(self, vocab_size: int = 1000, max_length: int = 512, *, mode: str = "char"):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.mode = (mode or "char").lower()
        if self.mode not in {"char", "mixed"}:
            raise ValueError("tokenizer mode must be 'char' or 'mixed'")
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

        # Add characters (including ^/_ plus operator glyphs like /, *, <, > used in canon)
        for c in ' \\n.,!?;:\'"()-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789^_*/<>=':
            if c not in self.token_to_id:
                self.token_to_id[c] = next_id
                next_id += 1

        self.actual_vocab_size = next_id
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self._rebuild_multi_token_index()

    @classmethod
    def from_json(cls, path: Path) -> "Seq2SeqTokenizer":
        data = json.loads(path.read_text(encoding="utf-8"))
        tok = cls(
            vocab_size=int(data.get("vocab_size", 1000)),
            max_length=int(data.get("max_length", 512)),
            mode=str(data.get("mode", "char") or "char"),
        )
        tok.token_to_id = {k: int(v) for k, v in (data.get("token_to_id") or {}).items()}
        tok.id_to_token = {int(v): k for k, v in tok.token_to_id.items()}
        tok.max_length = int(data.get("max_length", tok.max_length))
        tok.actual_vocab_size = int(data.get("actual_vocab_size", max(tok.id_to_token.keys()) + 1))
        tok.mode = str(data.get("mode", tok.mode) or tok.mode).lower()
        tok._rebuild_multi_token_index()
        return tok

    def _rebuild_multi_token_index(self) -> None:
        # Longest-match scan index for multi-character tokens (e.g., A10, ->, +T).
        special = {"<PAD>", "<UNK>", "<BOS>", "<EOS>"}
        multi = [t for t in self.token_to_id.keys() if len(t) > 1 and t not in special]
        multi.sort(key=len, reverse=True)
        by_first: Dict[str, List[str]] = {}
        for t in multi:
            by_first.setdefault(t[0], []).append(t)
        self._multi_by_first = by_first

    def _normalize_text(self, text: str) -> str:
        if text is None:
            return ""
        # Dataset prompts include newlines; the historical tokenizer vocab does not.
        # Normalize them to spaces for stable tokenization across platforms (WSL/Windows).
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\t", " ")
        text = text.replace("\n", " ")
        return text

    def _tokenize_mixed_core(self, text: str) -> List[int]:
        ids: List[int] = []
        i = 0
        n = len(text)
        while i < n:
            c = text[i]
            matched = False
            for tok in self._multi_by_first.get(c, []):
                if text.startswith(tok, i):
                    ids.append(self.token_to_id.get(tok, 1))
                    i += len(tok)
                    matched = True
                    break
            if matched:
                continue
            ids.append(self.token_to_id.get(c, 1))
            i += 1
        return ids

    def tokenize(self, text, add_bos=True, add_eos=True):
        text = self._normalize_text(text)
        if self.mode == "mixed":
            core = self._tokenize_mixed_core(text)
        else:
            core = [self.token_to_id.get(c, 1) for c in text]

        reserve = 0
        if add_bos:
            reserve += 1
        if add_eos:
            reserve += 1
        max_core = max(self.max_length - reserve, 0)
        core = core[:max_core]

        tokens: List[int] = []
        if add_bos:
            tokens.append(2)
        tokens.extend(core)
        if add_eos:
            tokens.append(3)
        if len(tokens) < self.max_length:
            tokens.extend([0] * (self.max_length - len(tokens)))
        return tokens[: self.max_length]

    def decode(self, token_ids):
        """Decode token IDs back to text"""
        chars = []
        for tid in token_ids:
            if tid == 0:  # PAD
                break
            if tid == 2:  # BOS
                continue
            if tid == 3:  # EOS
                break
            chars.append(self.id_to_token.get(tid, '<UNK>'))
        return ''.join(chars)

class _JsonlIndex:
    """Memory-light JSONL random access via byte offsets."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.offsets: List[int] = []
        self._fp = None
        self._pid: Optional[int] = None

        with open(self.path, 'rb') as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)

        self.format = self._detect_format()

    def __len__(self) -> int:
        return len(self.offsets)

    def _ensure_open(self):
        pid = os.getpid()
        if self._fp is None or self._pid != pid:
            if self._fp is not None:
                try:
                    self._fp.close()
                except Exception:
                    pass
            self._fp = open(self.path, 'rb')
            self._pid = pid

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_fp'] = None
        state['_pid'] = None
        return state

    def get(self, idx: int) -> Dict:
        self._ensure_open()
        self._fp.seek(self.offsets[idx])
        line = self._fp.readline().decode('utf-8')
        return json.loads(line)

    def _detect_format(self) -> str:
        with open(self.path, 'rb') as f:
            for offset in self.offsets[: min(len(self.offsets), 50)]:
                f.seek(offset)
                line = f.readline()
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line.decode('utf-8'))
                except Exception:
                    continue
                if isinstance(entry, dict):
                    if 'input' in entry and 'target' in entry:
                        return 'teacher'
                    if 'story' in entry and 'equation' in entry:
                        return 'base'
        return 'unknown'


class StoryEquationDataset(Dataset):
    """
    Dataset supporting:
      - teacher format: {input, target, direction, metadata.pair_id, ...}
      - base-pair format: {story, equation, expr_elements, expr_ops, pair_id, ...}
    """

    def __init__(self, data_path: str, tokenizer: Seq2SeqTokenizer, *, compact_equations: bool = False):
        self.tokenizer = tokenizer
        self.compact_equations = bool(compact_equations)
        self.reader = _JsonlIndex(data_path)
        self.format = self.reader.format

        if self.format == 'base':
            self.num_pairs = len(self.reader)
            self.num_records = self.num_pairs * 2
        elif self.format == 'teacher':
            self.num_pairs = None
            self.num_records = len(self.reader)
        else:
            raise ValueError(f"Unrecognized dataset format for {data_path}")

    def task_distribution(self) -> Dict[str, int]:
        if self.format == 'base':
            return {'story_to_equation': self.num_pairs, 'equation_to_story': self.num_pairs}
        counts: Dict[str, int] = {}
        for i in range(len(self.reader)):
            entry = self.reader.get(i)
            t = entry.get('task_type', 'unknown')
            counts[t] = counts.get(t, 0) + 1
        return counts

    def __len__(self) -> int:
        return self.num_records

    def _build_story_to_eq_prompt(self, story: str) -> str:
        return (
            "Given this TKS narrative:\n\n"
            f"{story}\n\n"
            "Translate this into a TKS equation using canonical notation (worlds A,B,C,D; noetics 1-10)."
        )

    def _build_eq_to_story_prompt(self, equation: str, elements: List[str]) -> str:
        elements_text = ', '.join(elements) if elements else equation
        return (
            f"Given the TKS equation: {equation}\n\n"
            f"Elements: {elements_text}\n\n"
            "Generate a natural language narrative describing this TKS working."
        )

    def __getitem__(self, idx: int) -> Dict:
        if self.format == 'base':
            pair_idx = idx // 2
            direction_flag = idx % 2
            entry = self.reader.get(pair_idx)

            story = entry.get('story', '')
            equation = entry.get('equation', '')
            if self.compact_equations and equation:
                equation = re.sub(r'\s+', '', equation.strip())
            elements = entry.get('expr_elements', []) or []
            operators = entry.get('expr_ops', []) or []
            pair_id = entry.get('pair_id', f'pair_{pair_idx:06d}')

            if direction_flag == 0:
                task_type = 'story_to_equation'
                direction = 'story_to_eq'
                input_text = self._build_story_to_eq_prompt(story)
                target_text = equation
            else:
                task_type = 'equation_to_story'
                direction = 'eq_to_story'
                input_text = self._build_eq_to_story_prompt(equation, elements)
                target_text = story

            input_ids = self.tokenizer.tokenize(input_text, add_bos=True, add_eos=True)
            target_ids = self.tokenizer.tokenize(target_text, add_bos=True, add_eos=True)
            decoder_target = target_ids[1:] + [0]

            return {
                'encoder_input': torch.tensor(input_ids, dtype=torch.long),
                'decoder_input': torch.tensor(target_ids, dtype=torch.long),
                'decoder_target': torch.tensor(decoder_target, dtype=torch.long),
                'task_type': task_type,
                'direction': direction,
                'pair_id': pair_id,
            }

        # teacher format
        entry = self.reader.get(idx)
        input_text = entry.get('input', '')
        target_text = entry.get('target', '')
        task_type = entry.get('task_type', 'story_to_equation')
        direction = entry.get('direction', entry.get('task_type', 'unknown'))
        pair_id = entry.get('metadata', {}).get('pair_id', 'unknown')

        input_ids = self.tokenizer.tokenize(input_text, add_bos=True, add_eos=True)
        target_ids = self.tokenizer.tokenize(target_text, add_bos=True, add_eos=True)
        decoder_target = target_ids[1:] + [0]

        return {
            'encoder_input': torch.tensor(input_ids, dtype=torch.long),
            'decoder_input': torch.tensor(target_ids, dtype=torch.long),
            'decoder_target': torch.tensor(decoder_target, dtype=torch.long),
            'task_type': task_type,
            'direction': direction,
            'pair_id': pair_id,
        }


class PairSubsetDataset(Dataset):
    """Pair-level subset for base datasets (keeps both directions)."""

    def __init__(self, base: StoryEquationDataset, pair_indices: List[int]):
        if base.format != 'base':
            raise ValueError("PairSubsetDataset requires base-pair dataset format")
        self.base = base
        self.pair_indices = pair_indices
        self.pair_count = len(pair_indices)

    def __len__(self) -> int:
        return self.pair_count * 2

    def __getitem__(self, idx: int) -> Dict:
        pair_local = idx // 2
        direction_flag = idx % 2
        pair_idx = self.pair_indices[pair_local]
        return self.base[pair_idx * 2 + direction_flag]

# Seq2Seq Transformer model
class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, num_layers=4, nhead=8, max_length=512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_length = max_length

        # Embeddings
        self.encoder_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.decoder_embedding = nn.Embedding(vocab_size, hidden_dim)

        # Positional encoding (simple learned)
        self.encoder_pos = nn.Embedding(max_length, hidden_dim)
        self.decoder_pos = nn.Embedding(max_length, hidden_dim)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dropout=0.1
        )

        # Output layer
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, encoder_input, decoder_input):
        batch_size = encoder_input.size(0)
        enc_seq_len = encoder_input.size(1)
        dec_seq_len = decoder_input.size(1)

        # Create position indices
        enc_pos = torch.arange(enc_seq_len, device=encoder_input.device).unsqueeze(0).expand(batch_size, -1)
        dec_pos = torch.arange(dec_seq_len, device=decoder_input.device).unsqueeze(0).expand(batch_size, -1)

        # Embed encoder input
        enc_emb = self.encoder_embedding(encoder_input) + self.encoder_pos(enc_pos)

        # Embed decoder input
        dec_emb = self.decoder_embedding(decoder_input) + self.decoder_pos(dec_pos)

        # Create masks
        tgt_mask = torch.triu(
            torch.ones(dec_seq_len, dec_seq_len, device=decoder_input.device, dtype=torch.bool),
            diagonal=1,
        )

        # Padding masks
        src_key_padding_mask = (encoder_input == 0)
        tgt_key_padding_mask = (decoder_input == 0)

        # Forward through transformer
        output = self.transformer(
            src=enc_emb,
            tgt=dec_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        # Project to vocabulary
        logits = self.output(output)
        return logits

def _loss_sample_avg(
    loss_fn: nn.Module,
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute masked CE loss as sample-average (each sample equally weighted)."""
    batch_size, seq_len = targets.shape
    loss_tokens = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1)).view(batch_size, seq_len)
    mask = (targets != 0).float()
    loss_per_sample = (loss_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return loss_per_sample.mean()

def _make_base_pair_split(num_pairs: int, *, eval_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    pair_indices = list(range(num_pairs))
    rng = random.Random(seed)
    rng.shuffle(pair_indices)

    eval_pair_count = max(1, int(num_pairs * eval_ratio))
    eval_pairs = pair_indices[:eval_pair_count]
    train_pairs = pair_indices[eval_pair_count:]
    if not train_pairs or not eval_pairs:
        raise ValueError("Unable to split pairs into train/eval")
    return train_pairs, eval_pairs


def _make_pair_aware_split_teacher(dataset: StoryEquationDataset, *, eval_ratio: float, seed: int):
    """Split teacher-format datasets by pair_id so both directions stay together."""
    pair_to_indices: Dict[str, List[int]] = {}
    for idx in range(len(dataset)):
        entry = dataset.reader.get(idx)
        pair_id = entry.get('metadata', {}).get('pair_id', None)
        if not pair_id:
            return None
        pair_to_indices.setdefault(pair_id, []).append(idx)

    pair_ids = list(pair_to_indices.keys())
    if len(pair_ids) < 2:
        return None

    rng = random.Random(seed)
    rng.shuffle(pair_ids)

    eval_pair_count = max(1, int(len(pair_ids) * eval_ratio))
    eval_pairs = set(pair_ids[:eval_pair_count])

    eval_indices: List[int] = []
    train_indices: List[int] = []
    for pid in pair_ids:
        indices = pair_to_indices[pid]
        if pid in eval_pairs:
            eval_indices.extend(indices)
        else:
            train_indices.extend(indices)

    if not train_indices or not eval_indices:
        return None

    return train_indices, eval_indices


def train_model(
    data_path,
    output_dir,
    epochs=10,
    batch_size=8,
    lr=1e-3,
    *,
    eval_data_path=None,
    seed: int = 42,
    max_length: int = 512,
    tokenizer_mode: str = "mixed",
    tokenizer_path: Optional[str] = None,
    init_checkpoint: Optional[str] = None,
    init_mixed_embeddings: bool = False,
    compact_equations: Optional[bool] = None,
    hidden_dim: int = 256,
    num_layers: int = 4,
    nhead: int = 8,
    device: str = "auto",
    eval_ratio: float = 0.1,
    max_train_pairs_per_epoch: int = 0,
    max_eval_pairs: int = 0,
    log_every: int = 10,
    num_workers: int = 0,
    pin_memory: bool = True,
    amp: str = "none",
    grad_accum_steps: int = 1,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
):
    print("=" * 70)
    print("TRACK 2 AGENT G: STORY<->EQUATION BIDIRECTIONAL TRAINING")
    print("=" * 70)

    random.seed(seed)
    torch.manual_seed(seed)

    if device == "auto":
        torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        torch_device = torch.device(device)
        if torch_device.type == 'cuda' and not torch.cuda.is_available():
            raise SystemExit("CUDA requested but torch.cuda.is_available() is False. Run `python scripts/check_cuda.py`.")

    print(f"\nDevice: {torch_device}")
    if torch_device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Setup
    tokenizer_mode = (tokenizer_mode or "mixed").lower()
    if tokenizer_mode not in {"char", "mixed"}:
        raise SystemExit("--tokenizer-mode must be one of: char, mixed")

    resolved_tok_path: Optional[Path] = Path(tokenizer_path) if tokenizer_path else None
    resolved_ckpt_path: Optional[Path] = None
    if init_checkpoint:
        resolved_ckpt_path = Path(init_checkpoint)
        if resolved_ckpt_path.is_dir():
            for cand in ("best_model.pt", "final_model.pt"):
                p = resolved_ckpt_path / cand
                if p.exists():
                    resolved_ckpt_path = p
                    break
        if not resolved_ckpt_path.exists():
            raise SystemExit(f"--init-checkpoint not found: {resolved_ckpt_path}")

        if resolved_tok_path is None:
            sibling = resolved_ckpt_path.parent / "tokenizer.json"
            if sibling.exists():
                resolved_tok_path = sibling

    if resolved_tok_path and resolved_tok_path.exists():
        tokenizer = Seq2SeqTokenizer.from_json(resolved_tok_path)
    else:
        tokenizer = Seq2SeqTokenizer(max_length=max_length, mode=tokenizer_mode)

    # Apply CLI overrides (mode + length)
    tokenizer.max_length = int(max_length)
    tokenizer.mode = tokenizer_mode
    tokenizer._rebuild_multi_token_index()

    compact_equations_final = (tokenizer.mode == "mixed") if compact_equations is None else bool(compact_equations)
    dataset = StoryEquationDataset(data_path, tokenizer, compact_equations=compact_equations_final)
    if dataset.format == 'base':
        print(f"Loaded {dataset.num_pairs:,} base pairs ({len(dataset):,} bidirectional records)")
    else:
        print(f"Loaded {len(dataset):,} training records")

    task_counts = dataset.task_distribution()

    print(f"\nTask distribution:")
    for task_type, count in sorted(task_counts.items()):
        print(f"  {task_type}: {count}")

    # Split data (90/10 train/eval), pair-aware when possible
    if eval_data_path:
        eval_dataset = StoryEquationDataset(eval_data_path, tokenizer, compact_equations=compact_equations_final)
        train_dataset = dataset
        train_size = len(train_dataset)
        eval_size = len(eval_dataset)
    else:
        if dataset.format == 'base':
            train_pairs, eval_pairs = _make_base_pair_split(dataset.num_pairs, eval_ratio=eval_ratio, seed=seed)
            train_dataset = PairSubsetDataset(dataset, train_pairs)
            eval_dataset = PairSubsetDataset(dataset, eval_pairs)
            train_size = len(train_dataset)
            eval_size = len(eval_dataset)
        else:
            split = _make_pair_aware_split_teacher(dataset, eval_ratio=eval_ratio, seed=seed)
            if split:
                train_indices, eval_indices = split
                train_dataset = torch.utils.data.Subset(dataset, train_indices)
                eval_dataset = torch.utils.data.Subset(dataset, eval_indices)
                train_size = len(train_indices)
                eval_size = len(eval_indices)
            else:
                eval_size = max(1, int(len(dataset) * eval_ratio))
                train_size = len(dataset) - eval_size
                train_dataset, eval_dataset = random_split(
                    dataset, [train_size, eval_size], generator=torch.Generator().manual_seed(seed)
                )

    print(f"\nTrain size: {train_size}")
    print(f"Eval size: {eval_size}")

    def _make_loader(ds, *, shuffle: bool, drop_last: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=(pin_memory and torch_device.type == 'cuda'),
        )

    # Create eval loader once (optionally subsampled for speed)
    eval_dataset_for_epoch = eval_dataset
    if max_eval_pairs and max_eval_pairs > 0:
        rng = random.Random(seed)
        if isinstance(eval_dataset, PairSubsetDataset):
            eval_pair_count = len(eval_dataset.pair_indices)
            if max_eval_pairs < eval_pair_count:
                subset_pairs = rng.sample(eval_dataset.pair_indices, k=max_eval_pairs)
                eval_dataset_for_epoch = PairSubsetDataset(eval_dataset.base, subset_pairs)
        elif isinstance(eval_dataset, StoryEquationDataset) and eval_dataset.format == 'base':
            if max_eval_pairs < eval_dataset.num_pairs:
                subset_pairs = rng.sample(range(eval_dataset.num_pairs), k=max_eval_pairs)
                eval_dataset_for_epoch = PairSubsetDataset(eval_dataset, subset_pairs)

    eval_loader = _make_loader(eval_dataset_for_epoch, shuffle=False, drop_last=False)

    # Initialize model
    model = Seq2SeqTransformer(
        vocab_size=tokenizer.actual_vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        nhead=nhead,
        max_length=max_length,
    )
    model.to(torch_device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")

    # Setup training
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction='none')

    # Learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=lr * 0.01)

    print(f"\nTraining configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  LR Scheduler: CosineAnnealingWarmRestarts")
    print(f"  Seed: {seed}")
    print(f"  Max length: {max_length}")
    print(f"  Tokenizer mode: {tokenizer.mode}")
    print(f"  Compact equations: {compact_equations_final}")
    if resolved_tok_path:
        print(f"  Tokenizer path: {resolved_tok_path}")
    if resolved_ckpt_path:
        print(f"  Init checkpoint: {resolved_ckpt_path}")
        print(f"  Init mixed embeddings: {init_mixed_embeddings}")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  num_layers: {num_layers}")
    print(f"  nhead: {nhead}")
    print(f"  eval_ratio: {eval_ratio}")
    if max_train_pairs_per_epoch and max_train_pairs_per_epoch > 0:
        print(f"  max_train_pairs_per_epoch: {max_train_pairs_per_epoch}")
    if max_eval_pairs and max_eval_pairs > 0:
        print(f"  max_eval_pairs: {max_eval_pairs}")
    print(f"  grad_accum_steps: {grad_accum_steps}")
    print(f"  amp: {amp}")
    print(f"  num_workers: {num_workers}")
    print(f"  pin_memory: {pin_memory}")
    if early_stop_patience and early_stop_patience > 0:
        print(f"  early_stop_patience: {early_stop_patience} (min_delta={early_stop_min_delta})")
    if eval_data_path:
        print(f"  Eval data: {eval_data_path}")
    print(f"  Output dir: {output_dir}")

    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    metrics = {
        'epoch_losses': [],
        'eval_losses': [],
        'task_distribution': task_counts,
        'epoch_metrics': [],
        'config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'seed': seed,
            'max_length': max_length,
            'tokenizer_mode': tokenizer.mode,
            'compact_equations': compact_equations_final,
            'tokenizer_path': str(resolved_tok_path) if resolved_tok_path else None,
            'init_checkpoint': str(resolved_ckpt_path) if resolved_ckpt_path else None,
            'init_mixed_embeddings': bool(init_mixed_embeddings),
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'nhead': nhead,
            'device': str(torch_device),
            'eval_ratio': eval_ratio,
            'max_train_pairs_per_epoch': max_train_pairs_per_epoch,
            'max_eval_pairs': max_eval_pairs,
            'log_every': log_every,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'amp': amp,
            'grad_accum_steps': grad_accum_steps,
            'early_stop_patience': early_stop_patience,
            'early_stop_min_delta': early_stop_min_delta,
        },
    }

    amp = (amp or "none").lower()
    if amp not in {"none", "fp16", "bf16"}:
        raise ValueError("amp must be one of: none, fp16, bf16")
    if torch_device.type != 'cuda' and amp != "none":
        print("[warn] AMP requested but CUDA is not active; disabling AMP.")
        amp = "none"

    autocast_ctx = nullcontext()
    amp_dtype = None
    scaler = None
    if amp == "fp16":
        autocast_ctx = torch.cuda.amp.autocast
        amp_dtype = torch.float16
        scaler = torch.cuda.amp.GradScaler()
    elif amp == "bf16":
        autocast_ctx = torch.cuda.amp.autocast
        amp_dtype = torch.bfloat16

    best_eval_token_loss: Optional[float] = None
    best_epoch: Optional[int] = None
    epochs_no_improve = 0

    def _load_checkpoint_into_model(path: Path) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state_dict, dict):
            raise SystemExit(f"Unsupported checkpoint format at {path}")

        model_state = model.state_dict()
        filtered = {}
        skipped = []
        for k, v in state_dict.items():
            if k in model_state and hasattr(v, "shape") and model_state[k].shape == v.shape:
                filtered[k] = v
            else:
                skipped.append(k)

        incompatible = model.load_state_dict(filtered, strict=False)
        print(f"Loaded init checkpoint weights: {path}")
        print(f"  loaded_keys: {len(filtered)}")
        print(f"  skipped_keys: {len(skipped)}")
        if incompatible.missing_keys:
            print(f"  missing_keys: {len(incompatible.missing_keys)}")
        if incompatible.unexpected_keys:
            print(f"  unexpected_keys: {len(incompatible.unexpected_keys)}")

    def _init_mixed_tokens_from_chars() -> None:
        # Initialize multi-character token embeddings/output rows from their character components.
        t2i = tokenizer.token_to_id
        special = {"<PAD>", "<UNK>", "<BOS>", "<EOS>"}
        multi_tokens = [t for t in t2i.keys() if len(t) > 1 and t not in special]
        if not multi_tokens:
            return

        def _components(tok: str) -> List[str]:
            if tok in {"+T", "-T", "*T", "/T"}:
                return [tok[0], "T"]
            if tok == "->":
                return ["-", ">"]
            if tok == "<-":
                return ["<", "-"]
            return list(tok)

        updated = 0
        skipped = 0
        with torch.no_grad():
            for tok in multi_tokens:
                tid = t2i.get(tok)
                comps = _components(tok)
                comp_ids = [t2i.get(c) for c in comps]
                if tid is None or any(cid is None for cid in comp_ids):
                    skipped += 1
                    continue
                comp_ids_int = [int(cid) for cid in comp_ids]  # type: ignore[arg-type]

                enc_avg = model.encoder_embedding.weight.data[comp_ids_int].mean(dim=0)
                dec_avg = model.decoder_embedding.weight.data[comp_ids_int].mean(dim=0)
                out_avg = model.output.weight.data[comp_ids_int].mean(dim=0)
                model.encoder_embedding.weight.data[int(tid)] = enc_avg
                model.decoder_embedding.weight.data[int(tid)] = dec_avg
                model.output.weight.data[int(tid)] = out_avg
                if getattr(model.output, "bias", None) is not None:
                    b_avg = model.output.bias.data[comp_ids_int].mean(dim=0)
                    model.output.bias.data[int(tid)] = b_avg
                updated += 1

        print(f"Initialized {updated} multi-character tokens from character components (skipped={skipped})")

    if resolved_ckpt_path:
        _load_checkpoint_into_model(resolved_ckpt_path)

        if init_mixed_embeddings and tokenizer.mode == "mixed":
            # If the checkpoint already used mixed tokenization, re-init is likely harmful.
            ckpt_tok_mode = None
            if resolved_tok_path and resolved_tok_path.exists():
                try:
                    ckpt_tok_mode = (json.loads(resolved_tok_path.read_text(encoding="utf-8")) or {}).get("mode")
                except Exception:
                    ckpt_tok_mode = None
            if str(ckpt_tok_mode or "char").lower() != "mixed":
                _init_mixed_tokens_from_chars()
            else:
                print("[info] Skipping init-mixed-embeddings (checkpoint tokenizer already mixed).")

    for epoch in range(epochs):
        model.train()
        epoch_t0 = time.monotonic()

        train_epoch_dataset = train_dataset
        if max_train_pairs_per_epoch and max_train_pairs_per_epoch > 0:
            rng = random.Random(seed + epoch)
            if isinstance(train_dataset, PairSubsetDataset):
                train_pair_count = len(train_dataset.pair_indices)
                if max_train_pairs_per_epoch < train_pair_count:
                    subset_pairs = rng.sample(train_dataset.pair_indices, k=max_train_pairs_per_epoch)
                    train_epoch_dataset = PairSubsetDataset(train_dataset.base, subset_pairs)
            elif isinstance(train_dataset, StoryEquationDataset) and train_dataset.format == 'base':
                if max_train_pairs_per_epoch < train_dataset.num_pairs:
                    subset_pairs = rng.sample(range(train_dataset.num_pairs), k=max_train_pairs_per_epoch)
                    train_epoch_dataset = PairSubsetDataset(train_dataset, subset_pairs)

        train_loader = _make_loader(train_epoch_dataset, shuffle=True, drop_last=True)

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 70)

        # Train accumulators
        train_sample_loss_sum = 0.0
        train_token_loss_sum = 0.0
        train_token_count = 0
        train_correct = 0
        train_seq_exact = 0
        train_sample_count = 0

        train_by_dir: Dict[str, Dict[str, float]] = {}

        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(train_loader):
            encoder_input = batch['encoder_input'].to(torch_device, non_blocking=True)
            decoder_input = batch['decoder_input'].to(torch_device, non_blocking=True)
            decoder_target = batch['decoder_target'].to(torch_device, non_blocking=True)
            directions = batch.get('direction', ['unknown'] * encoder_input.size(0))

            with autocast_ctx(dtype=amp_dtype) if amp != "none" else nullcontext():
                logits = model(encoder_input, decoder_input)
                batch_size, seq_len = decoder_target.shape
                loss_tokens = loss_fn(logits.view(-1, logits.size(-1)), decoder_target.view(-1)).view(batch_size, seq_len)
                mask = (decoder_target != 0)
                mask_f = mask.float()

                sample_token_counts = mask_f.sum(dim=1).clamp(min=1.0)
                sample_losses = (loss_tokens * mask_f).sum(dim=1) / sample_token_counts
                loss_sample = sample_losses.mean()

            loss_for_step = loss_sample / max(grad_accum_steps, 1)
            if scaler is not None:
                scaler.scale(loss_for_step).backward()
            else:
                loss_for_step.backward()

            # Metrics (use full-precision detached tensors)
            with torch.no_grad():
                token_loss_sum = (loss_tokens * mask_f).sum().item()
                token_count = int(mask.sum().item())
                preds = logits.argmax(dim=-1)
                correct = int(((preds == decoder_target) & mask).sum().item())
                seq_exact = ((preds == decoder_target) | (~mask)).all(dim=1)
                seq_exact_count = int(seq_exact.sum().item())

                train_token_loss_sum += token_loss_sum
                train_token_count += token_count
                train_correct += correct
                train_seq_exact += seq_exact_count
                train_sample_loss_sum += float(sample_losses.sum().item())
                train_sample_count += int(batch_size)

                # Per-direction aggregates
                for i in range(batch_size):
                    d = directions[i]
                    if d not in train_by_dir:
                        train_by_dir[d] = {
                            'token_loss_sum': 0.0,
                            'token_count': 0,
                            'correct': 0,
                            'seq_exact': 0,
                            'samples': 0,
                        }
                    sample_mask = mask[i]
                    sample_token_count = int(sample_mask.sum().item())
                    if sample_token_count == 0:
                        continue
                    train_by_dir[d]['token_loss_sum'] += float((loss_tokens[i][sample_mask]).sum().item())
                    train_by_dir[d]['token_count'] += sample_token_count
                    train_by_dir[d]['correct'] += int(((preds[i] == decoder_target[i]) & sample_mask).sum().item())
                    train_by_dir[d]['seq_exact'] += int(bool(seq_exact[i].item()))
                    train_by_dir[d]['samples'] += 1

            # Optimizer step (grad accumulation)
            if (batch_idx + 1) % max(grad_accum_steps, 1) == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if log_every and (batch_idx % log_every == 0 or batch_idx == len(train_loader) - 1):
                current_lr = optimizer.param_groups[0]['lr']
                avg_tok_loss = (train_token_loss_sum / train_token_count) if train_token_count else 0.0
                avg_ppl = math.exp(avg_tok_loss) if avg_tok_loss < 50 else float('inf')
                avg_acc = (train_correct / train_token_count) if train_token_count else 0.0
                print(
                    f"  Step {batch_idx}/{len(train_loader)} "
                    f"loss_s={loss_sample.item():.4f} "
                    f"train_ppl={avg_ppl:.2f} "
                    f"train_acc={avg_acc:.3f} "
                    f"lr={current_lr:.6f}"
                )

        # Final optimizer step if needed (leftover grads)
        if len(train_loader) % max(grad_accum_steps, 1) != 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        train_sample_loss = train_sample_loss_sum / max(train_sample_count, 1)
        train_token_loss = train_token_loss_sum / max(train_token_count, 1)
        train_ppl = math.exp(train_token_loss) if train_token_loss < 50 else float('inf')
        train_acc = train_correct / max(train_token_count, 1)
        train_seq_exact_rate = train_seq_exact / max(train_sample_count, 1)

        metrics['epoch_losses'].append({'epoch': epoch + 1, 'loss': train_sample_loss})
        print(
            f"\n  Train: loss_s={train_sample_loss:.4f} "
            f"loss_t={train_token_loss:.4f} ppl={train_ppl:.2f} "
            f"acc={train_acc:.3f} seq_exact={train_seq_exact_rate:.3f} "
            f"tokens={train_token_count:,}"
        )
        for d, m in sorted(train_by_dir.items()):
            d_loss = (m['token_loss_sum'] / m['token_count']) if m['token_count'] else 0.0
            d_ppl = math.exp(d_loss) if d_loss < 50 else float('inf')
            d_acc = (m['correct'] / m['token_count']) if m['token_count'] else 0.0
            d_seq = (m['seq_exact'] / m['samples']) if m['samples'] else 0.0
            print(f"    {d:12s}: ppl={d_ppl:7.2f} acc={d_acc:.3f} seq_exact={d_seq:.3f} samples={m['samples']:,}")

        # Evaluate
        model.eval()
        eval_sample_loss_sum = 0.0
        eval_token_loss_sum = 0.0
        eval_token_count = 0
        eval_correct = 0
        eval_seq_exact = 0
        eval_sample_count = 0
        eval_by_dir: Dict[str, Dict[str, float]] = {}

        with torch.no_grad():
            for batch in eval_loader:
                encoder_input = batch['encoder_input'].to(torch_device)
                decoder_input = batch['decoder_input'].to(torch_device)
                decoder_target = batch['decoder_target'].to(torch_device)
                directions = batch.get('direction', ['unknown'] * encoder_input.size(0))

                with autocast_ctx(dtype=amp_dtype) if amp != "none" else nullcontext():
                    logits = model(encoder_input, decoder_input)
                    batch_size, seq_len = decoder_target.shape
                    loss_tokens = loss_fn(logits.view(-1, logits.size(-1)), decoder_target.view(-1)).view(batch_size, seq_len)
                    mask = (decoder_target != 0)
                    mask_f = mask.float()

                    sample_token_counts = mask_f.sum(dim=1).clamp(min=1.0)
                    sample_losses = (loss_tokens * mask_f).sum(dim=1) / sample_token_counts

                token_loss_sum = (loss_tokens * mask_f).sum().item()
                token_count = int(mask.sum().item())
                preds = logits.argmax(dim=-1)
                correct = int(((preds == decoder_target) & mask).sum().item())
                seq_exact = ((preds == decoder_target) | (~mask)).all(dim=1)
                seq_exact_count = int(seq_exact.sum().item())

                eval_token_loss_sum += token_loss_sum
                eval_token_count += token_count
                eval_correct += correct
                eval_seq_exact += seq_exact_count
                eval_sample_loss_sum += float(sample_losses.sum().item())
                eval_sample_count += int(batch_size)

                for i in range(batch_size):
                    d = directions[i]
                    if d not in eval_by_dir:
                        eval_by_dir[d] = {
                            'token_loss_sum': 0.0,
                            'token_count': 0,
                            'correct': 0,
                            'seq_exact': 0,
                            'samples': 0,
                        }
                    sample_mask = mask[i]
                    sample_token_count = int(sample_mask.sum().item())
                    if sample_token_count == 0:
                        continue
                    eval_by_dir[d]['token_loss_sum'] += float((loss_tokens[i][sample_mask]).sum().item())
                    eval_by_dir[d]['token_count'] += sample_token_count
                    eval_by_dir[d]['correct'] += int(((preds[i] == decoder_target[i]) & sample_mask).sum().item())
                    eval_by_dir[d]['seq_exact'] += int(bool(seq_exact[i].item()))
                    eval_by_dir[d]['samples'] += 1

        eval_sample_loss = eval_sample_loss_sum / max(eval_sample_count, 1)
        eval_token_loss = eval_token_loss_sum / max(eval_token_count, 1)
        eval_ppl = math.exp(eval_token_loss) if eval_token_loss < 50 else float('inf')
        eval_acc = eval_correct / max(eval_token_count, 1)
        eval_seq_exact_rate = eval_seq_exact / max(eval_sample_count, 1)

        metrics['eval_losses'].append({'epoch': epoch + 1, 'loss': eval_sample_loss})
        print(
            f"  Eval : loss_s={eval_sample_loss:.4f} "
            f"loss_t={eval_token_loss:.4f} ppl={eval_ppl:.2f} "
            f"acc={eval_acc:.3f} seq_exact={eval_seq_exact_rate:.3f} "
            f"tokens={eval_token_count:,}"
        )
        for d, m in sorted(eval_by_dir.items()):
            d_loss = (m['token_loss_sum'] / m['token_count']) if m['token_count'] else 0.0
            d_ppl = math.exp(d_loss) if d_loss < 50 else float('inf')
            d_acc = (m['correct'] / m['token_count']) if m['token_count'] else 0.0
            d_seq = (m['seq_exact'] / m['samples']) if m['samples'] else 0.0
            print(f"    {d:12s}: ppl={d_ppl:7.2f} acc={d_acc:.3f} seq_exact={d_seq:.3f} samples={m['samples']:,}")

        epoch_time = time.monotonic() - epoch_t0
        tokens_per_sec = train_token_count / max(epoch_time, 1e-6)
        print(f"  Epoch time: {epoch_time:.1f}s | train tok/s: {tokens_per_sec:,.0f}")

        epoch_metrics = {
            'epoch': epoch + 1,
            'lr': optimizer.param_groups[0]['lr'],
            'time_sec': epoch_time,
            'train': {
                'sample_loss': train_sample_loss,
                'token_loss': train_token_loss,
                'perplexity': train_ppl,
                'token_accuracy': train_acc,
                'sequence_exact': train_seq_exact_rate,
                'tokens': train_token_count,
                'samples': train_sample_count,
                'by_direction': {},
            },
            'eval': {
                'sample_loss': eval_sample_loss,
                'token_loss': eval_token_loss,
                'perplexity': eval_ppl,
                'token_accuracy': eval_acc,
                'sequence_exact': eval_seq_exact_rate,
                'tokens': eval_token_count,
                'samples': eval_sample_count,
                'by_direction': {},
            },
        }

        for d, m in train_by_dir.items():
            epoch_metrics['train']['by_direction'][d] = {
                'token_loss': (m['token_loss_sum'] / m['token_count']) if m['token_count'] else None,
                'perplexity': (math.exp(m['token_loss_sum'] / m['token_count']) if (m['token_count'] and (m['token_loss_sum'] / m['token_count']) < 50) else None),
                'token_accuracy': (m['correct'] / m['token_count']) if m['token_count'] else None,
                'sequence_exact': (m['seq_exact'] / m['samples']) if m['samples'] else None,
                'tokens': m['token_count'],
                'samples': m['samples'],
            }
        for d, m in eval_by_dir.items():
            epoch_metrics['eval']['by_direction'][d] = {
                'token_loss': (m['token_loss_sum'] / m['token_count']) if m['token_count'] else None,
                'perplexity': (math.exp(m['token_loss_sum'] / m['token_count']) if (m['token_count'] and (m['token_loss_sum'] / m['token_count']) < 50) else None),
                'token_accuracy': (m['correct'] / m['token_count']) if m['token_count'] else None,
                'sequence_exact': (m['seq_exact'] / m['samples']) if m['samples'] else None,
                'tokens': m['token_count'],
                'samples': m['samples'],
            }

        metrics['epoch_metrics'].append(epoch_metrics)

        # Save best checkpoint by eval token loss
        improved = best_eval_token_loss is None or (eval_token_loss < best_eval_token_loss - early_stop_min_delta)
        if improved:
            best_eval_token_loss = eval_token_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            best_path = output_path / "best_model.pt"
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'vocab_size': tokenizer.actual_vocab_size,
                    'config': {
                        'model_type': 'transformer_seq2seq',
                        'hidden_dim': hidden_dim,
                        'num_layers': num_layers,
                        'nhead': nhead,
                        'max_length': tokenizer.max_length,
                        'tokenizer_mode': tokenizer.mode,
                    },
                    'best_eval_token_loss': best_eval_token_loss,
                    'best_epoch': best_epoch,
                },
                best_path,
            )
            print(f"  [best] Saved {best_path} (eval_token_loss={best_eval_token_loss:.4f})")
        else:
            epochs_no_improve += 1

        if early_stop_patience and epochs_no_improve >= early_stop_patience:
            print(f"  [early-stop] No improvement for {epochs_no_improve} epochs; stopping.")
            break

        # Step scheduler
        scheduler.step()
        print(f"  Next epoch LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / "final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': tokenizer.actual_vocab_size,
        'config': {
            'model_type': 'transformer_seq2seq',
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'nhead': nhead,
            'max_length': tokenizer.max_length,
            'tokenizer_mode': tokenizer.mode,
        }
    }, model_path)
    print(f"\nSaved model to: {model_path}")

    # Save tokenizer config
    tokenizer_path = output_path / "tokenizer.json"
    with open(tokenizer_path, 'w') as f:
        json.dump({
            'vocab_size': tokenizer.vocab_size,
            'max_length': tokenizer.max_length,
            'actual_vocab_size': tokenizer.actual_vocab_size,
            'mode': tokenizer.mode,
            'token_to_id': tokenizer.token_to_id,
        }, f, indent=2)
    print(f"Saved tokenizer to: {tokenizer_path}")

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
    if best_epoch is not None:
        print(f"Best eval token loss: {best_eval_token_loss:.4f} @ epoch {best_epoch}")
    print(f"\nModel saved to: {output_path}")

    return metrics

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train story↔equation bidirectional translation model')
    parser.add_argument('--data', required=True, help='Path to training data (JSONL)')
    parser.add_argument('--output-dir', default='output/teacher_model_story', help='Output directory for model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--eval-data', default=None, help='Optional holdout JSONL for evaluation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max-length', type=int, default=512, help='Max sequence length')
    parser.add_argument('--tokenizer', default=None, help='Optional tokenizer.json (defaults to --init-checkpoint sibling when provided)')
    parser.add_argument('--tokenizer-mode', type=str, default='mixed', choices=['char', 'mixed'], help='Tokenizer mode (mixed enables atomic TKS tokens)')
    parser.add_argument('--init-checkpoint', default=None, help='Optional checkpoint (.pt or directory) to warm-start from')
    parser.add_argument('--init-mixed-embeddings', action='store_true', help='When warm-starting a char-token model into mixed mode, init composite token weights from character components')
    parser.add_argument('--compact-equations', action=argparse.BooleanOptionalAction, default=None, help='Strip whitespace from equations in prompts/targets (default: True for mixed tokenizer)')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Transformer hidden dimension')
    parser.add_argument('--num-layers', type=int, default=4, help='Number of encoder/decoder layers')
    parser.add_argument('--nhead', type=int, default=8, help='Attention heads')
    parser.add_argument('--device', default='auto', help='auto|cpu|cuda|cuda:0')
    parser.add_argument('--eval-ratio', type=float, default=0.1, help='Eval ratio when --eval-data not provided')
    parser.add_argument('--max-train-pairs-per-epoch', type=int, default=0, help='For base datasets, cap pairs per epoch (0=all)')
    parser.add_argument('--max-eval-pairs', type=int, default=0, help='For base datasets, cap eval pairs (0=all)')
    parser.add_argument('--log-every', type=int, default=10, help='Log every N steps (0=disable)')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader workers (0 recommended on WSL /mnt/c)')
    parser.add_argument('--pin-memory', action=argparse.BooleanOptionalAction, default=True, help='Pin CPU memory for faster H2D copies')
    parser.add_argument('--amp', type=str, default='none', choices=['none', 'fp16', 'bf16'], help='Mixed precision')
    parser.add_argument('--grad-accum-steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--early-stop-patience', type=int, default=0, help='Stop if eval doesn\'t improve for N epochs (0=off)')
    parser.add_argument('--early-stop-min-delta', type=float, default=0.0, help='Minimum eval improvement to reset patience')

    args = parser.parse_args()
    train_model(
        args.data,
        args.output_dir,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        eval_data_path=args.eval_data,
        seed=args.seed,
        max_length=args.max_length,
        tokenizer_mode=args.tokenizer_mode,
        tokenizer_path=args.tokenizer,
        init_checkpoint=args.init_checkpoint,
        init_mixed_embeddings=args.init_mixed_embeddings,
        compact_equations=args.compact_equations,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        nhead=args.nhead,
        device=args.device,
        eval_ratio=args.eval_ratio,
        max_train_pairs_per_epoch=args.max_train_pairs_per_epoch,
        max_eval_pairs=args.max_eval_pairs,
        log_every=args.log_every,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        amp=args.amp,
        grad_accum_steps=args.grad_accum_steps,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
    )
