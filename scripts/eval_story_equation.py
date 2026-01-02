"""
Story↔Equation Evaluation Script with Round-Trip Testing

This script evaluates a dual-task model trained on story_to_equation and
equation_to_story tasks, including:
1. Standard metrics (accuracy, loss) per task direction
2. Round-trip consistency testing (story→equation→story)
3. Semantic similarity metrics

Usage:
    python scripts/eval_story_equation.py \
        --checkpoint output/teacher_model_story/final_model.pt \
        --data output/teacher_story_eq_holdout.jsonl \
        --output output/teacher_model_story/eval_results.json

Author: Track 2 Agent H
Date: 2025-12-15
"""

import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from difflib import SequenceMatcher

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Simple tokenizer for seq2seq (copied from train_story_equation_lite.py)
class Seq2SeqTokenizer:
    def __init__(self, vocab_size=1000, max_length=256, *, mode: str = "char"):
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

        # Add characters
        for c in ' \\n.,!?;:\'"()-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789^_*/<>=':
            if c not in self.token_to_id:
                self.token_to_id[c] = next_id
                next_id += 1

        self.actual_vocab_size = next_id
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self._rebuild_multi_token_index()

    @classmethod
    def from_json(cls, path: Path) -> "Seq2SeqTokenizer":
        data = json.loads(path.read_text(encoding='utf-8'))
        tok = cls(
            vocab_size=int(data.get('vocab_size', 1000)),
            max_length=int(data.get('max_length', 256)),
            mode=str(data.get("mode", "char") or "char"),
        )
        tok.token_to_id = {k: int(v) for k, v in data['token_to_id'].items()}
        tok.id_to_token = {int(v): k for k, v in tok.token_to_id.items()}
        tok.max_length = int(data.get('max_length', tok.max_length))
        tok.actual_vocab_size = int(data.get('actual_vocab_size', max(tok.id_to_token.keys()) + 1))
        tok.mode = str(data.get("mode", tok.mode) or tok.mode).lower()
        tok._rebuild_multi_token_index()
        return tok

    def _rebuild_multi_token_index(self) -> None:
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
            tokens.append(2)  # BOS
        tokens.extend(core)
        if add_eos:
            tokens.append(3)  # EOS
        while len(tokens) < self.max_length:
            tokens.append(0)  # PAD
        return tokens[:self.max_length]

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
    """Memory-light JSONL random access via byte offsets (safe for DataLoader workers)."""

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


# Simplified Seq2Seq model (encoder-decoder LSTM)
class SimpleLSTMSeq2Seq(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Embeddings
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # Encoder
        self.encoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)

        # Decoder
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)

        # Output
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, encoder_input, decoder_input):
        # Encode
        enc_emb = self.embedding(encoder_input)
        _, (hidden, cell) = self.encoder(enc_emb)

        # Decode
        dec_emb = self.embedding(decoder_input)
        dec_out, _ = self.decoder(dec_emb, (hidden, cell))

        # Project to vocab
        logits = self.output(dec_out)
        return logits


# Seq2Seq Transformer model (copied from train_story_equation.py)
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


class StoryEquationDataset(Dataset):
    """
    Dataset supporting:
      - teacher format: {input, target, task_type, direction, metadata.pair_id, ...}
      - base-pair format: {story, equation, expr_elements, expr_ops, pair_id, ...}

    For base format, each pair expands to 2 records (story→eq and eq→story).
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

    def __getitem__(self, idx: int) -> Dict[str, Any]:
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
            pair_id = entry.get('pair_id', f'pair_{pair_idx:07d}')

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
                'input_text': input_text,
                'target_text': target_text,
            }

        # teacher format
        entry = self.reader.get(idx)

        input_text = entry.get('input', '')
        target_text = entry.get('target', '')
        task_type = entry.get('task_type', 'unknown')
        direction = entry.get('direction', task_type)
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
            'input_text': input_text,
            'target_text': target_text,
        }


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    device: torch.device = None,
) -> str:
    """
    Generate text from the seq2seq model given a prompt.

    Args:
        model: Trained seq2seq model
        tokenizer: Tokenizer instance
        prompt: Input prompt text
        max_length: Maximum generation length
        device: Device to run on

    Returns:
        Generated text
    """
    if device is None:
        device = torch.device('cpu')

    model.eval()

    # Tokenize encoder input and trim padding to reduce compute
    encoder_input_ids = tokenizer.tokenize(prompt, add_bos=True, add_eos=True)
    try:
        enc_len = encoder_input_ids.index(0)
    except ValueError:
        enc_len = len(encoder_input_ids)
    enc_len = max(enc_len, 1)
    encoder_tensor = torch.tensor([encoder_input_ids[:enc_len]], dtype=torch.long).to(device)

    max_seq_len = getattr(tokenizer, 'max_length', 256)

    # Start decoder with BOS token
    decoder_input = [2]  # BOS

    # Greedy decoding (incremental; no full-length padding each step)
    for _ in range(max_new_tokens):
        decoder_tensor = torch.tensor([decoder_input], dtype=torch.long).to(device)

        logits = model(encoder_tensor, decoder_tensor)
        next_token = torch.argmax(logits[0, -1, :]).item()

        # Stop if EOS token (3) or PAD token (0)
        if next_token == 3 or next_token == 0:
            break

        decoder_input.append(next_token)

        # Limit generation length by model sequence window
        if len(decoder_input) >= max_seq_len:
            break

    # Decode to text
    text = tokenizer.decode(decoder_input)
    return text.strip()


@torch.no_grad()
def evaluate_dual_task(
    model: nn.Module,
    dataloader: DataLoader,
    tokenizer: Seq2SeqTokenizer,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Evaluate model on dual-task data (story_to_equation and equation_to_story).

    Args:
        model: Trained seq2seq model
        dataloader: DataLoader for evaluation data
        tokenizer: Tokenizer instance
        device: Device to run on

    Returns:
        Dict with evaluation metrics per task direction
    """
    model.eval()

    loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction='none')

    direction_metrics: Dict[str, Dict[str, float]] = {}

    total_token_loss_sum = 0.0
    total_sample_loss_sum = 0.0
    total_correct = 0
    total_tokens = 0
    total_samples = 0
    num_batches = 0

    for batch in dataloader:
        encoder_input = batch['encoder_input'].to(device)
        decoder_input = batch['decoder_input'].to(device)
        decoder_target = batch['decoder_target'].to(device)
        directions = batch['direction']

        # Forward pass
        logits = model(encoder_input, decoder_input)
        batch_size, seq_len = decoder_target.shape
        loss_tokens = loss_fn(logits.view(-1, logits.size(-1)), decoder_target.view(-1)).view(batch_size, seq_len)
        mask = (decoder_target != 0)
        mask_f = mask.float()
        sample_token_counts = mask_f.sum(dim=1).clamp(min=1.0)
        sample_losses = (loss_tokens * mask_f).sum(dim=1) / sample_token_counts

        token_loss_sum = float((loss_tokens * mask_f).sum().item())
        token_count = int(mask.sum().item())

        # Compute accuracy
        preds = logits.argmax(dim=-1)
        correct = int(((preds == decoder_target) & mask).sum().item())
        total_correct += correct
        total_tokens += token_count
        total_token_loss_sum += token_loss_sum
        total_sample_loss_sum += float(sample_losses.sum().item())
        total_samples += int(batch_size)

        # Track per-direction metrics
        for i in range(batch_size):
            direction = directions[i]

            if direction not in direction_metrics:
                direction_metrics[direction] = {
                    'token_loss_sum': 0.0,
                    'correct': 0,
                    'total_tokens': 0,
                    'count': 0,
                }

            sample_mask = mask[i]
            sample_correct = int(((preds[i] == decoder_target[i]) & sample_mask).sum().item())
            sample_tokens = int(sample_mask.sum().item())

            if sample_tokens > 0:
                direction_metrics[direction]['token_loss_sum'] += float(loss_tokens[i][sample_mask].sum().item())
                direction_metrics[direction]['correct'] += sample_correct
                direction_metrics[direction]['total_tokens'] += sample_tokens
            direction_metrics[direction]['count'] += 1

        num_batches += 1

    # Compute overall metrics
    avg_token_loss = total_token_loss_sum / max(total_tokens, 1)
    avg_sample_loss = total_sample_loss_sum / max(total_samples, 1)
    overall_accuracy = total_correct / max(total_tokens, 1)
    perplexity = math.exp(avg_token_loss) if avg_token_loss < 50 else float('inf')

    # Compute per-direction metrics
    direction_results = {}
    for direction, metrics in direction_metrics.items():
        if metrics['count'] > 0:
            d_loss = (metrics['token_loss_sum'] / metrics['total_tokens']) if metrics['total_tokens'] else None
            direction_results[direction] = {
                'loss': d_loss,
                'accuracy': metrics['correct'] / max(metrics['total_tokens'], 1),
                'perplexity': (math.exp(d_loss) if (d_loss is not None and d_loss < 50) else None),
                'count': metrics['count'],
                'tokens': metrics['total_tokens'],
            }

    return {
        'overall': {
            'loss': avg_token_loss,
            'sample_loss': avg_sample_loss,
            'accuracy': overall_accuracy,
            'perplexity': perplexity,
            'num_batches': num_batches,
            'total_tokens': total_tokens,
            'total_samples': total_samples,
        },
        'by_direction': direction_results,
    }


def test_roundtrip_consistency(
    model: nn.Module,
    dataset: StoryEquationDataset,
    tokenizer: Seq2SeqTokenizer,
    device: torch.device,
    num_samples: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Test round-trip consistency: story → equation → story.

    Args:
        model: Trained model
        dataset: Dataset with paired story/equation samples
        tokenizer: Tokenizer instance
        device: Device to run on
        num_samples: Number of samples to test

    Returns:
        Dict with round-trip test results
    """
    model.eval()

    results = []

    element_re = re.compile(r'[ABCD](?:10|[1-9])(?:\^[1-9])?(?:_d[1-7])?')
    norm_eq = lambda s: re.sub(r'\s+', '', (s or '').strip()).upper()

    rng = random.Random(seed)

    if dataset.format == 'base':
        pair_indices = list(range(dataset.num_pairs))
        rng.shuffle(pair_indices)
        pair_indices = pair_indices[: min(num_samples, len(pair_indices))]
        print(f"\nRound-trip testing (base): sampling {len(pair_indices)} pairs")

        for pair_idx in pair_indices:
            entry = dataset.reader.get(pair_idx)
            pair_id = entry.get('pair_id', f'pair_{pair_idx:07d}')
            original_story = entry.get('story', '')
            original_equation = entry.get('equation', '')

            story_prompt = dataset._build_story_to_eq_prompt(original_story)
            generated_equation = generate_text(model, tokenizer, story_prompt, max_new_tokens=128, device=device)

            elements = element_re.findall(generated_equation)
            equation_prompt = dataset._build_eq_to_story_prompt(generated_equation, elements)
            regenerated_story = generate_text(model, tokenizer, equation_prompt, max_new_tokens=256, device=device)

            similarity = SequenceMatcher(None, original_story.strip().lower(), regenerated_story.strip().lower()).ratio()

            results.append({
                'pair_id': pair_id,
                'original_story': original_story,
                'original_equation': original_equation,
                'generated_equation': generated_equation,
                'regenerated_story': regenerated_story,
                'similarity': similarity,
                'equation_match': norm_eq(generated_equation) == norm_eq(original_equation),
            })
    else:
        pairs: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for i in range(len(dataset)):
            entry = dataset.reader.get(i)
            pair_id = entry.get('metadata', {}).get('pair_id', None)
            if not pair_id:
                continue
            pairs.setdefault(pair_id, {})
            direction = entry.get('direction', entry.get('task_type', 'unknown'))
            pairs[pair_id][direction] = entry

        complete_pairs = {
            pid: p for pid, p in pairs.items()
            if 'story_to_eq' in p and 'eq_to_story' in p
        }
        pair_ids = list(complete_pairs.keys())
        rng.shuffle(pair_ids)
        pair_ids = pair_ids[: min(num_samples, len(pair_ids))]
        print(f"\nRound-trip testing (teacher): sampling {len(pair_ids)} of {len(complete_pairs)} complete pairs")

        for pair_id in pair_ids:
            pair_data = complete_pairs[pair_id]

            story_entry = pair_data['story_to_eq']
            original_story_input = story_entry.get('input', '')
            if "TKS narrative:\n\n" in original_story_input:
                original_story = original_story_input.split("TKS narrative:\n\n")[1].split("\n\nTranslate")[0]
            else:
                original_story = story_entry.get('target', '')

            original_equation = story_entry.get('target', '')

            story_prompt = story_entry.get('input', '')
            generated_equation = generate_text(model, tokenizer, story_prompt, max_new_tokens=128, device=device)

            elements = element_re.findall(generated_equation)
            elements_text = ', '.join(elements) if elements else generated_equation
            equation_prompt = (
                f"Given the TKS equation: {generated_equation}\n\n"
                f"Elements: {elements_text}\n\n"
                f"Generate a natural language narrative describing this TKS working."
            )
            regenerated_story = generate_text(model, tokenizer, equation_prompt, max_new_tokens=256, device=device)

            similarity = SequenceMatcher(None, original_story.strip().lower(), regenerated_story.strip().lower()).ratio()

            results.append({
                'pair_id': pair_id,
                'original_story': original_story,
                'original_equation': original_equation,
                'generated_equation': generated_equation,
                'regenerated_story': regenerated_story,
                'similarity': similarity,
                'equation_match': norm_eq(generated_equation) == norm_eq(original_equation),
            })

    # Compute aggregate statistics
    avg_similarity = sum(r['similarity'] for r in results) / max(len(results), 1)
    equation_match_rate = sum(1 for r in results if r['equation_match']) / max(len(results), 1)

    return {
        'num_tested': len(results),
        'avg_similarity': avg_similarity,
        'equation_match_rate': equation_match_rate,
        'samples': results,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Story↔Equation Evaluation with Round-Trip Testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to holdout JSONL data file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for evaluation report (JSON)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for evaluation (default: 8)')
    parser.add_argument('--roundtrip-samples', type=int, default=20,
                       help='Number of samples for round-trip testing (default: 20)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--tokenizer', type=str, default=None,
                       help='Optional tokenizer.json path (default: checkpoint sibling tokenizer.json)')
    parser.add_argument('--tokenizer-mode', type=str, default=None, choices=['char', 'mixed'],
                       help='Override tokenizer mode (default: mode stored in tokenizer.json, else char)')
    parser.add_argument('--compact-equations', action=argparse.BooleanOptionalAction, default=None,
                       help='Strip whitespace from equations in prompts/targets (default: True for mixed tokenizer)')
    parser.add_argument('--device', type=str, default='auto',
                       help='auto|cpu|cuda|cuda:0')

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Device setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise SystemExit("CUDA requested but torch.cuda.is_available() is False. Run `python scripts/check_cuda.py`.")

    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Initialize tokenizer
    checkpoint_path = Path(args.checkpoint)
    tokenizer_path = Path(args.tokenizer) if args.tokenizer else (checkpoint_path.parent / "tokenizer.json")
    if tokenizer_path.exists():
        tokenizer = Seq2SeqTokenizer.from_json(tokenizer_path)
        print(f"Tokenizer loaded: {tokenizer_path}")
    else:
        tokenizer = Seq2SeqTokenizer(max_length=256)
        print("Tokenizer: default (tokenizer.json not found)")

    if args.tokenizer_mode:
        tokenizer.mode = args.tokenizer_mode.lower()
        tokenizer._rebuild_multi_token_index()
        print(f"Tokenizer mode override: {tokenizer.mode}")
    print(f"Tokenizer vocabulary: {tokenizer.actual_vocab_size} (max_length={tokenizer.max_length})")

    compact_equations = (tokenizer.mode == "mixed") if args.compact_equations is None else bool(args.compact_equations)
    print(f"Compact equations: {compact_equations}")

    # Load dataset
    print(f"\nLoading data from: {args.data}")
    dataset = StoryEquationDataset(args.data, tokenizer, compact_equations=compact_equations)
    if dataset.format == 'base':
        print(f"Total: {dataset.num_pairs:,} base pairs ({len(dataset):,} bidirectional records)")
    else:
        print(f"Total entries: {len(dataset):,}")

    task_counts = dataset.task_distribution()
    print(f"\nTask distribution:")
    for task_type, count in sorted(task_counts.items()):
        print(f"  {task_type}: {count:,}")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Load checkpoint
    print(f"\nLoading model from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        config = checkpoint.get('config', {}) or {}
    else:
        state_dict = checkpoint
        config = {}

    # Infer architecture from config/state_dict
    model_type = config.get('model_type')
    state_keys = list(state_dict.keys()) if hasattr(state_dict, 'keys') else []
    if not model_type:
        if any(k.startswith('encoder_embedding.') for k in state_keys):
            model_type = 'transformer_seq2seq'
        elif any(k.startswith('encoder.') for k in state_keys) and any(k.startswith('decoder.') for k in state_keys):
            model_type = 'lstm_seq2seq'

    if model_type == 'transformer_seq2seq':
        model = Seq2SeqTransformer(
            vocab_size=tokenizer.actual_vocab_size,
            hidden_dim=int(config.get('hidden_dim', 256)),
            num_layers=int(config.get('num_layers', 4)),
            nhead=int(config.get('nhead', 8)),
            max_length=int(config.get('max_length', tokenizer.max_length)),
        )
    elif model_type == 'lstm_seq2seq':
        model = SimpleLSTMSeq2Seq(
            vocab_size=tokenizer.actual_vocab_size,
            hidden_dim=int(config.get('hidden_dim', 128)),
        )
    else:
        raise ValueError(
            "Unsupported checkpoint for this evaluator. "
            f"Expected seq2seq LSTM/Transformer, got model_type={model_type!r}."
        )

    model.to(device)
    model.load_state_dict(state_dict)
    print(f"Model loaded successfully (model_type={model_type})")

    # Run evaluation
    print("\n" + "=" * 70)
    print("STORY<->EQUATION EVALUATION RESULTS")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.data}")

    # Standard evaluation
    print("\nEvaluating model performance...")
    eval_metrics = evaluate_dual_task(model, dataloader, tokenizer, device)

    print(f"\nOverall Performance:")
    print(f"  Loss: {eval_metrics['overall']['loss']:.4f}")
    print(f"  Sample loss: {eval_metrics['overall']['sample_loss']:.4f}")
    print(f"  Accuracy: {eval_metrics['overall']['accuracy']:.4f}")
    print(f"  Perplexity: {eval_metrics['overall']['perplexity']:.2f}")
    print(f"  Batches: {eval_metrics['overall']['num_batches']}")
    print(f"  Tokens: {eval_metrics['overall']['total_tokens']}")

    print(f"\nPer-Direction Performance:")
    for direction, metrics in sorted(eval_metrics['by_direction'].items()):
        loss = metrics.get('loss')
        ppl = metrics.get('perplexity')
        print(f"  {direction}:")
        print(f"    Loss: {loss:.4f}" if loss is not None else "    Loss: n/a")
        print(f"    Perplexity: {ppl:.2f}" if ppl is not None else "    Perplexity: n/a")
        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    Samples: {metrics['count']}")
        print(f"    Tokens: {metrics['tokens']}")

    # Round-trip testing
    print("\n" + "=" * 70)
    print("ROUND-TRIP CONSISTENCY TESTING")
    print("=" * 70)

    roundtrip_results = test_roundtrip_consistency(
        model, dataset, tokenizer, device, num_samples=args.roundtrip_samples
    )

    print(f"\nRound-Trip Results:")
    print(f"  Samples tested: {roundtrip_results['num_tested']}")
    print(f"  Avg similarity: {roundtrip_results['avg_similarity']:.4f}")
    print(f"  Equation match rate: {roundtrip_results['equation_match_rate']:.2%}")

    # Show a few examples
    print(f"\nSample Round-Trips:")
    for i, sample in enumerate(roundtrip_results['samples'][:5]):
        print(f"\n  Example {i+1} (pair_id: {sample['pair_id']}):")
        print(f"    Original equation: {sample['original_equation']}")
        print(f"    Generated equation: {sample['generated_equation']}")
        print(f"    Equation match: {sample['equation_match']}")
        print(f"    Similarity: {sample['similarity']:.2%}")

    # Generate report
    report = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'checkpoint': args.checkpoint,
            'data': args.data,
            'batch_size': args.batch_size,
            'roundtrip_samples': args.roundtrip_samples,
            'seed': args.seed,
        },
        'evaluation': eval_metrics,
        'roundtrip': roundtrip_results,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nReport saved to: {output_path}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
