"""
Lightweight training script for bidirectional story-equation translation
Reduced complexity for faster training and testing
"""
import json
import random
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Simple tokenizer for seq2seq
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

        # Add characters (including ^/_ plus operator glyphs like /, *, <, > used in canon)
        for c in ' \\n.,!?;:\'"()-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789^_*/<>=':
            if c not in self.token_to_id:
                self.token_to_id[c] = next_id
                next_id += 1

        self.actual_vocab_size = next_id
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self._rebuild_multi_token_index()

    def _rebuild_multi_token_index(self) -> None:
        special = {"<PAD>", "<UNK>", "<BOS>", "<EOS>"}
        multi = [t for t in self.token_to_id.keys() if len(t) > 1 and t not in special]
        multi.sort(key=len, reverse=True)
        by_first = {}
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

    def _tokenize_mixed_core(self, text: str):
        ids = []
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

        tokens = []
        if add_bos:
            tokens.append(2)
        tokens.extend(core)
        if add_eos:
            tokens.append(3)
        if len(tokens) < self.max_length:
            tokens.extend([0] * (self.max_length - len(tokens)))
        return tokens[: self.max_length]

# Dataset
class StoryEquationDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.tokenizer = tokenizer
        self.entries = []

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    self.entries.append(entry)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        input_text = entry.get('input', '')
        target_text = entry.get('target', '')

        input_ids = self.tokenizer.tokenize(input_text, add_bos=True, add_eos=True)
        target_ids = self.tokenizer.tokenize(target_text, add_bos=True, add_eos=True)

        decoder_target = target_ids[1:] + [0]

        return {
            'encoder_input': torch.tensor(input_ids, dtype=torch.long),
            'decoder_input': torch.tensor(target_ids, dtype=torch.long),
            'decoder_target': torch.tensor(decoder_target, dtype=torch.long),
            'task_type': entry.get('task_type', 'story_to_equation'),
            'direction': entry.get('direction', entry.get('task_type', 'unknown')),
            'pair_id': entry.get('metadata', {}).get('pair_id', 'unknown'),
        }

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


def _make_pair_aware_split(entries, eval_ratio: float, seed: int):
    """Split indices by pair_id so both directions stay together."""
    pair_to_indices = {}
    for idx, entry in enumerate(entries):
        pair_id = entry.get('metadata', {}).get('pair_id', None)
        if not pair_id:
            return None
        pair_to_indices.setdefault(pair_id, []).append(idx)

    pair_ids = list(pair_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(pair_ids)

    eval_pair_count = max(1, int(len(pair_ids) * eval_ratio))
    eval_pairs = set(pair_ids[:eval_pair_count])

    eval_indices = [i for pid in pair_ids if pid in eval_pairs for i in pair_to_indices[pid]]
    train_indices = [i for pid in pair_ids if pid not in eval_pairs for i in pair_to_indices[pid]]

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
    max_length: int = 256,
    hidden_dim: int = 128,
    device: str = "auto",
):
    print("="*70)
    print("TRACK 2 AGENT G: STORY-EQUATION BIDIRECTIONAL TRAINING (LITE)")
    print("="*70)

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
    tokenizer = Seq2SeqTokenizer(max_length=max_length)
    dataset = StoryEquationDataset(data_path, tokenizer)
    print(f"Loaded {len(dataset)} training samples")

    # Count task types
    task_counts = {}
    for entry in dataset.entries:
        task_type = entry.get('task_type', 'unknown')
        task_counts[task_type] = task_counts.get(task_type, 0) + 1

    print(f"\nTask distribution:")
    for task_type, count in sorted(task_counts.items()):
        print(f"  {task_type}: {count}")

    # Split
    if eval_data_path:
        eval_dataset = StoryEquationDataset(eval_data_path, tokenizer)
        train_dataset = dataset
        train_size = len(train_dataset)
        eval_size = len(eval_dataset)
    else:
        eval_ratio = 0.1
        split = _make_pair_aware_split(dataset.entries, eval_ratio=eval_ratio, seed=seed)
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

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = SimpleLSTMSeq2Seq(vocab_size=tokenizer.actual_vocab_size, hidden_dim=hidden_dim)
    model.to(torch_device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")

    # Training setup
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction='none')

    print(f"\nTraining configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Seed: {seed}")
    print(f"  Max length: {max_length}")
    if eval_data_path:
        print(f"  Eval data: {eval_data_path}")
    print(f"  Output dir: {output_dir}")

    # Training loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)

    metrics = {
        'epoch_losses': [],
        'eval_losses': [],
        'task_distribution': task_counts,
    }

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-"*70)

        for batch_idx, batch in enumerate(train_loader):
            encoder_input = batch['encoder_input'].to(torch_device)
            decoder_input = batch['decoder_input'].to(torch_device)
            decoder_target = batch['decoder_target'].to(torch_device)

            optimizer.zero_grad()
            logits = model(encoder_input, decoder_input)
            loss = _loss_sample_avg(loss_fn, logits, decoder_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1:
                print(f"  Step {batch_idx}/{len(train_loader)}: loss={loss.item():.4f}")

        avg_loss = epoch_loss / max(num_batches, 1)
        metrics['epoch_losses'].append({'epoch': epoch + 1, 'loss': avg_loss})
        print(f"\n  Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        # Evaluate
        model.eval()
        eval_loss = 0.0
        eval_batches = 0

        with torch.no_grad():
            eval_losses_by_direction = {}
            eval_counts_by_direction = {}
            for batch in eval_loader:
                encoder_input = batch['encoder_input'].to(torch_device)
                decoder_input = batch['decoder_input'].to(torch_device)
                decoder_target = batch['decoder_target'].to(torch_device)

                logits = model(encoder_input, decoder_input)
                loss = _loss_sample_avg(loss_fn, logits, decoder_target)
                eval_loss += loss.item()
                eval_batches += 1

                # Direction breakdown (optional)
                if 'direction' in batch:
                    directions = batch['direction']
                    batch_size_ = decoder_target.size(0)
                    token_losses = loss_fn(logits.view(-1, logits.size(-1)), decoder_target.view(-1)).view(batch_size_, -1)
                    mask = (decoder_target != 0).float()
                    sample_losses = (token_losses * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                    for i in range(batch_size_):
                        d = directions[i]
                        eval_losses_by_direction[d] = eval_losses_by_direction.get(d, 0.0) + sample_losses[i].item()
                        eval_counts_by_direction[d] = eval_counts_by_direction.get(d, 0) + 1

        avg_eval_loss = eval_loss / max(eval_batches, 1)
        metrics['eval_losses'].append({'epoch': epoch + 1, 'loss': avg_eval_loss})
        print(f"  Eval loss: {avg_eval_loss:.4f}")
        if eval_counts_by_direction:
            for d in sorted(eval_counts_by_direction.keys()):
                avg_d = eval_losses_by_direction[d] / eval_counts_by_direction[d]
                print(f"    {d} eval loss: {avg_d:.4f} ({eval_counts_by_direction[d]} samples)")

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / "final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': tokenizer.actual_vocab_size,
        'config': {
            'model_type': 'lstm_seq2seq',
            'hidden_dim': hidden_dim,
            'max_length': tokenizer.max_length,
        }
    }, model_path)
    print(f"\nSaved model to: {model_path}")

    # Save tokenizer
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

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nFinal train loss: {metrics['epoch_losses'][-1]['loss']:.4f}")
    print(f"Final eval loss: {metrics['eval_losses'][-1]['loss']:.4f}")

    return metrics

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output-dir', default='output/teacher_model_story')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--eval-data', default=None, help='Optional holdout JSONL for evaluation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max-length', type=int, default=256, help='Max sequence length')
    parser.add_argument('--hidden-dim', type=int, default=128, help='LSTM hidden dimension')
    parser.add_argument('--device', default='auto', help='auto|cpu|cuda|cuda:0')

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
        hidden_dim=args.hidden_dim,
        device=args.device,
    )
