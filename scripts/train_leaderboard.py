#!/usr/bin/env python3
"""TKS Training with Real-Time Leaderboard - Direct integration."""

import sys
import os

# Force unbuffered output for real-time display
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
import json
import random
import time
from pathlib import Path
from datetime import datetime

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Any, Optional

# Check GPU
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    DEVICE = 'cuda'
else:
    print("FATAL ERROR: CUDA not available.")
    print("To protect the CPU, training will not start.")
    sys.exit(1)

from src.tks.core.tks_llm_core_v4 import TKSNoeticLM, TKSNoeticLMConfig

# Standard separator - used everywhere: dataset, generation, eval
SEP = " => "


class TKSDataset(Dataset):
    """TKS training dataset with variable-length batching support."""

    def __init__(self, data_path: str, tokenizer_vocab: dict, max_seq_len: int = 256):
        self.max_seq_len = max_seq_len
        self.vocab = tokenizer_vocab
        self.pad_id = tokenizer_vocab.get('<PAD>', 0)
        self.bos_id = tokenizer_vocab.get('<BOS>', 2)
        self.eos_id = tokenizer_vocab.get('<EOS>', 3)
        self.unk_id = tokenizer_vocab.get('<UNK>', 1)

        # Load data
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        self.data.append(json.loads(line))
                    except:
                        continue
        print(f"Loaded {len(self.data)} training samples")

    def _tokenize(self, text: str) -> List[int]:
        ids = [self.bos_id]
        for c in text[:self.max_seq_len - 2]:
            ids.append(self.vocab.get(c, self.unk_id))
        ids.append(self.eos_id)
        return ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Get prompt and target
        if 'input' in item and 'output' in item:
            prompt = item['input']
            target = item['output']
        elif 'prompt' in item and 'target' in item:
            prompt = item['prompt']
            target = item['target']
        elif 'input' in item and 'target' in item:
            prompt = item['input']
            target = item['target']
        else:
            prompt = ""
            target = item.get('text', '')

        # Build full text: prompt + SEP + target
        full_text = prompt + SEP + target if prompt else target
        input_ids = self._tokenize(full_text)

        # Calculate prompt length for label masking (prompt + SEP)
        # We want to mask all tokens up to and including SEP
        prompt_with_sep = prompt + SEP if prompt else ""
        prompt_tokens = self._tokenize(prompt_with_sep) if prompt_with_sep else [self.bos_id]
        prompt_len = len(prompt_tokens) - 1  # -1 because we don't count EOS from prompt tokenization

        # Build labels: -100 for prompt, next token for target
        labels = []
        for i in range(len(input_ids)):
            if i < prompt_len:
                labels.append(-100)
            else:
                next_tok = input_ids[i + 1] if i + 1 < len(input_ids) else self.eos_id
                labels.append(next_tok)

        return {'input_ids': input_ids, 'labels': labels}


def collate_fn(batch, pad_id=0):
    """Collate with variable-length padding."""
    max_len = max(len(x['input_ids']) for x in batch)

    input_ids = []
    labels = []
    attention_mask = []

    for item in batch:
        ids = item['input_ids']
        labs = item['labels']
        pad_len = max_len - len(ids)

        input_ids.append(ids + [pad_id] * pad_len)
        labels.append(labs + [-100] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)

    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
    }


class Leaderboard:
    """Real-time training leaderboard."""

    def __init__(self):
        self.history = []

    def evaluate(self, model, vocab, device, test_samples):
        """Evaluate model on generation quality."""
        model.eval()
        id2tok = {v: k for k, v in vocab.items()}
        pad_id = vocab.get('<PAD>', 0)
        eos_id = vocab.get('<EOS>', 3)
        bos_id = vocab.get('<BOS>', 2)

        def generate(prompt, max_tokens=30):
            # Truncate prompt if too long
            max_prompt_len = 256 - max_tokens - 10
            if len(prompt) > max_prompt_len:
                prompt = prompt[:max_prompt_len]
                
            # Add SEP to match training format: prompt + SEP + target
            prompt_with_sep = prompt.rstrip() + SEP
            ids = [bos_id] + [vocab.get(c, vocab.get('<UNK>', 1)) for c in prompt_with_sep]
            
            # Ensure ids don't exceed limit
            if len(ids) > 256 - max_tokens:
                ids = ids[-(256 - max_tokens):]
                # Ensure BOS is first if we cut it
                if ids[0] != bos_id:
                    ids[0] = bos_id
            
            generated = torch.tensor([ids], dtype=torch.long, device=device)

            tokens = []
            with torch.no_grad():
                for _ in range(max_tokens):
                    if generated.shape[1] >= 256: break
                    
                    mask = torch.ones_like(generated)
                    out = model(generated, attention_mask=mask)
                    logits = out.get('logits', out.get('output')) if isinstance(out, dict) else out

                    next_logits = logits[:, -1, :].clone()
                    next_logits[:, pad_id] = float('-inf')

                    next_tok = next_logits.argmax(dim=-1).item()
                    if next_tok == eos_id:
                        break
                    tokens.append(id2tok.get(next_tok, '?'))
                    generated = torch.cat([generated, torch.tensor([[next_tok]], device=device)], dim=1)

            return ''.join(tokens)

        def generate_topk(prompt, expected, k=3):
            """Generate and check if correct output is in top-k at each position."""
            # Truncate prompt
            max_tokens = len(expected) + 5
            max_prompt_len = 256 - max_tokens - 10
            if len(prompt) > max_prompt_len:
                prompt = prompt[:max_prompt_len]
                
            prompt_with_sep = prompt.rstrip() + SEP
            ids = [bos_id] + [vocab.get(c, vocab.get('<UNK>', 1)) for c in prompt_with_sep]
            
            # Ensure ids don't exceed limit
            if len(ids) > 256 - max_tokens:
                ids = ids[-(256 - max_tokens):]
                if ids[0] != bos_id:
                    ids[0] = bos_id
            
            generated = torch.tensor([ids], dtype=torch.long, device=device)

            # Check first few target tokens
            target_ids = [vocab.get(c, vocab.get('<UNK>', 1)) for c in expected[:5]]
            topk_hits = 0

            with torch.no_grad():
                for i, target_id in enumerate(target_ids):
                    if generated.shape[1] >= 256: break
                    
                    mask = torch.ones_like(generated)
                    out = model(generated, attention_mask=mask)
                    logits = out.get('logits', out.get('output')) if isinstance(out, dict) else out

                    next_logits = logits[:, -1, :]
                    topk_ids = next_logits.topk(k).indices[0].tolist()

                    if target_id in topk_ids:
                        topk_hits += 1

                    # Use greedy for next step
                    next_tok = next_logits.argmax(dim=-1).item()
                    if next_tok == eos_id:
                        break
                    generated = torch.cat([generated, torch.tensor([[next_tok]], device=device)], dim=1)

            return topk_hits, len(target_ids)

        # Evaluate
        syntax_ok = 0
        exact = 0
        struct_ok = 0
        topk_total_hits = 0
        topk_total_positions = 0
        samples_out = []

        for sample in test_samples:
            prompt = sample.get('input', sample.get('prompt', ''))
            expected = sample.get('output', sample.get('target', ''))
            if not prompt:
                continue

            output = generate(prompt, max_tokens=len(expected) + 10)

            if output.startswith('('):
                syntax_ok += 1
            if output.strip() == expected.strip():
                exact += 1
            if '(' in output and ')' in output:
                struct_ok += 1

            # Top-K check
            hits, positions = generate_topk(prompt, expected)
            topk_total_hits += hits
            topk_total_positions += positions

            if len(samples_out) < 3:
                samples_out.append((prompt[:25], expected[:15], output[:15]))

        n = len(test_samples)
        topk_pct = 100 * topk_total_hits / max(topk_total_positions, 1)

        return {
            'syntax_pct': 100 * syntax_ok / max(n, 1),
            'exact_pct': 100 * exact / max(n, 1),
            'struct_pct': 100 * struct_ok / max(n, 1),
            'topk_pct': topk_pct,
            'samples': samples_out,
        }

    def update(self, epoch, train_loss, eval_loss, metrics):
        self.history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'eval_loss': eval_loss,
            **metrics,
        })

    def display(self):
        print("\n" + "=" * 100)
        print("                              REAL-TIME LEADERBOARD")
        print("=" * 100)
        print(f"{'Epoch':>6} | {'Train':>9} | {'Eval':>9} | {'Syntax%':>8} | {'Exact%':>8} | {'Top3%':>7} | {'Struct%':>8}")
        print("-" * 100)

        for e in self.history:
            print(f"{e['epoch']:>6} | {e['train_loss']:>9.4f} | {e['eval_loss']:>9.4f} | "
                  f"{e['syntax_pct']:>7.1f}% | {e['exact_pct']:>7.1f}% | {e.get('topk_pct', 0):>6.1f}% | {e['struct_pct']:>7.1f}%")

        if self.history:
            print("-" * 100)
            best = max(self.history, key=lambda x: x['syntax_pct'] + x['exact_pct'] + x.get('topk_pct', 0))
            print(f"BEST: Epoch {best['epoch']} - Syntax {best['syntax_pct']:.1f}%, Exact {best['exact_pct']:.1f}%, Top3 {best.get('topk_pct', 0):.1f}%")

            latest = self.history[-1]
            if latest.get('samples'):
                print("\nLatest Outputs:")
                for p, exp, out in latest['samples']:
                    print(f"  '{p}...' -> '{out}' (exp: '{exp}')")

        print("=" * 100 + "\n")


def train(args):
    device = DEVICE

    # Load tokenizer
    tokenizer_path = 'checkpoints/v2_fixed/tokenizer.json'
    with open(tokenizer_path, 'r') as f:
        vocab = json.load(f)
    print(f"Vocab size: {len(vocab)}")

    # Create model
    config = TKSNoeticLMConfig(vocab_size=len(vocab))
    model = TKSNoeticLM(config).to(device)

    # Resume if checkpoint exists
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1

    # Create dataset
    dataset = TKSDataset(args.data, vocab)
    pad_id = vocab.get('<PAD>', 0)

    # Split for eval
    eval_size = min(500, len(dataset) // 10)
    train_size = len(dataset) - eval_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
        num_workers=0,
        pin_memory=USE_CUDA,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id),
        num_workers=0,
    )

    # Get test samples for leaderboard
    test_samples = random.sample(dataset.data, min(50, len(dataset.data)))

    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    scaler = torch.amp.GradScaler(device) if USE_CUDA else None

    # Leaderboard
    leaderboard = Leaderboard()

    # Output dir
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"LR: {args.lr}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    print("=" * 70 + "\n")

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # TRAIN
        model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            optimizer.zero_grad()

            if USE_CUDA:
                with torch.amp.autocast('cuda'):
                    output = model(input_ids, attention_mask=attention_mask, return_full_trace=True)
                    logits = output.get('logits', output.get('output')) if isinstance(output, dict) else output
                    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(input_ids, attention_mask=attention_mask, return_full_trace=True)
                logits = output.get('logits', output.get('output')) if isinstance(output, dict) else output
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        train_loss = total_loss / num_batches

        # EVAL
        model.eval()
        eval_loss = 0
        eval_batches = 0
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                if USE_CUDA:
                    with torch.amp.autocast('cuda'):
                        output = model(input_ids, attention_mask=attention_mask)
                        logits = output.get('logits', output.get('output')) if isinstance(output, dict) else output
                        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                else:
                    output = model(input_ids, attention_mask=attention_mask)
                    logits = output.get('logits', output.get('output')) if isinstance(output, dict) else output
                    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

                eval_loss += loss.item()
                eval_batches += 1

        eval_loss = eval_loss / max(eval_batches, 1)

        # Update leaderboard
        gen_metrics = leaderboard.evaluate(model, vocab, device, test_samples)
        leaderboard.update(epoch + 1, train_loss, eval_loss, gen_metrics)
        leaderboard.display()

        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch+1} completed in {elapsed:.1f}s\n")

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config.__dict__,
            'tokenizer': vocab,
            'metrics': {
                'train_losses': [e['train_loss'] for e in leaderboard.history],
                'eval_losses': [e['eval_loss'] for e in leaderboard.history],
            }
        }, output_path / 'latest_model.pt')

        # Save best
        if gen_metrics['syntax_pct'] + gen_metrics['exact_pct'] >= max(
            e['syntax_pct'] + e['exact_pct'] for e in leaderboard.history
        ):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config.__dict__,
                'tokenizer': vocab,
            }, output_path / 'best_model.pt')
            print(f"  Saved best model (epoch {epoch+1})")

    # Final save
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
        'tokenizer': vocab,
    }, output_path / 'final_model.pt')

    # Save history
    with open(output_path / 'leaderboard.json', 'w') as f:
        json.dump(leaderboard.history, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    best = max(leaderboard.history, key=lambda x: x['syntax_pct'] + x['exact_pct'])
    print(f"Best: Epoch {best['epoch']} - Syntax {best['syntax_pct']:.1f}%, Exact {best['exact_pct']:.1f}%")
    print(f"Checkpoint: {output_path / 'best_model.pt'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--data', default='datasets/jsonl/tks_v1_nl2tks_mix.jsonl')
    parser.add_argument('--output-dir', default='checkpoints/v3_leaderboard')
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()

    train(args)
