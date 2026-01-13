#!/usr/bin/env python3
"""TKS Training with Real-Time Leaderboard - Shows generation quality each epoch."""

import sys
import os
import json
import random
import time
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

# Check GPU availability
if not torch.cuda.is_available():
    print("ERROR: CUDA not available! This script requires GPU.")
    print("Please run on a machine with CUDA-capable GPU.")
    sys.exit(1)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

from src.tks.core.tks_llm_core_v4 import TKSNoeticLM, TKSNoeticLMConfig


class RealTimeLeaderboard:
    """Tracks and displays training progress with generation quality metrics."""

    def __init__(self, test_prompts, test_expected, data_path=None):
        self.test_prompts = test_prompts
        self.test_expected = test_expected
        self.data_path = data_path
        self.history = []

        # Load some training samples for exact match testing
        self.train_samples = []
        if data_path and os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                all_samples = [json.loads(line) for line in f if line.strip()]
            self.train_samples = random.sample(all_samples, min(20, len(all_samples)))

    def evaluate_generation(self, model, tokenizer, device):
        """Run generation tests and return metrics."""
        model.eval()

        pad_id = tokenizer.vocab.get('<PAD>', 0)
        eos_id = tokenizer.vocab.get('<EOS>', 3)
        bos_id = tokenizer.vocab.get('<BOS>', 2)

        def encode(text):
            # Add trailing space to match training format
            text = text.rstrip() + ' '
            ids = [bos_id]
            for c in text:
                ids.append(tokenizer.vocab.get(c, tokenizer.vocab.get('<UNK>', 1)))
            return ids

        def decode(ids):
            result = []
            for i in ids:
                tok = tokenizer.id2tok.get(i, '?')
                if tok not in ['<PAD>', '<BOS>', '<EOS>']:
                    result.append(tok)
            return ''.join(result)

        def generate(prompt, max_tokens=30):
            input_ids = encode(prompt)
            generated = torch.tensor([input_ids], dtype=torch.long, device=device)
            tokens = []

            with torch.no_grad():
                for _ in range(max_tokens):
                    attention_mask = torch.ones_like(generated)
                    output = model(generated, attention_mask=attention_mask)
                    logits = output.get('logits', output.get('output')) if isinstance(output, dict) else output

                    next_logits = logits[:, -1, :].clone()
                    next_logits[:, pad_id] = float('-inf')

                    next_token = next_logits.argmax(dim=-1, keepdim=True)
                    token_id = next_token.item()

                    if token_id == eos_id:
                        break

                    tokens.append(tokenizer.id2tok.get(token_id, '?'))
                    generated = torch.cat([generated, next_token], dim=1)

            return ''.join(tokens)

        # Test A: Syntax validity (starts with '(')
        syntax_valid = 0
        for prompt in self.test_prompts:
            output = generate(prompt)
            if output.startswith('('):
                syntax_valid += 1
        syntax_pct = 100 * syntax_valid / len(self.test_prompts)

        # Test B: Exact match on training samples
        exact_matches = 0
        close_matches = 0
        for sample in self.train_samples:
            prompt = sample.get('input', sample.get('prompt', ''))
            expected = sample.get('output', sample.get('target', ''))
            if not prompt or not expected:
                continue

            output = generate(prompt, max_tokens=len(expected) + 10)

            if output.strip() == expected.strip():
                exact_matches += 1
            elif expected.strip() in output or output.strip() in expected.strip():
                close_matches += 1

        exact_pct = 100 * exact_matches / max(len(self.train_samples), 1)
        close_pct = 100 * close_matches / max(len(self.train_samples), 1)

        # Test C: Structure validity (has parens, operators, sephirot)
        struct_valid = 0
        for prompt in self.test_prompts:
            output = generate(prompt)
            has_parens = '(' in output and ')' in output
            has_sephirot = any(s in output for s in ['A', 'B', 'C', 'D'])
            if has_parens and has_sephirot:
                struct_valid += 1
        struct_pct = 100 * struct_valid / len(self.test_prompts)

        # Get sample outputs for display
        sample_outputs = []
        for i, prompt in enumerate(self.test_prompts[:3]):
            output = generate(prompt)
            expected = self.test_expected[i] if i < len(self.test_expected) else "?"
            sample_outputs.append((prompt[:30], expected, output[:30]))

        return {
            'syntax_valid_pct': syntax_pct,
            'exact_match_pct': exact_pct,
            'close_match_pct': close_pct,
            'structure_valid_pct': struct_pct,
            'sample_outputs': sample_outputs,
        }

    def update(self, epoch, train_loss, eval_loss, model, tokenizer, device):
        """Update leaderboard with new epoch results."""
        gen_metrics = self.evaluate_generation(model, tokenizer, device)

        entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'eval_loss': eval_loss,
            'timestamp': datetime.now().isoformat(),
            **gen_metrics
        }
        self.history.append(entry)

        return entry

    def display(self):
        """Display the leaderboard."""
        print("\n" + "=" * 90)
        print("                         REAL-TIME TRAINING LEADERBOARD")
        print("=" * 90)
        print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Eval Loss':>10} | {'Syntax%':>8} | {'Exact%':>7} | {'Struct%':>8}")
        print("-" * 90)

        for entry in self.history:
            print(f"{entry['epoch']:>6} | {entry['train_loss']:>10.4f} | {entry['eval_loss']:>10.4f} | "
                  f"{entry['syntax_valid_pct']:>7.1f}% | {entry['exact_match_pct']:>6.1f}% | {entry['structure_valid_pct']:>7.1f}%")

        print("-" * 90)

        # Show best epoch
        if self.history:
            best = max(self.history, key=lambda x: x['syntax_valid_pct'] + x['exact_match_pct'])
            print(f"BEST: Epoch {best['epoch']} (Syntax: {best['syntax_valid_pct']:.1f}%, Exact: {best['exact_match_pct']:.1f}%)")

        # Show sample outputs from latest
        if self.history:
            latest = self.history[-1]
            print("\nSample Outputs (Latest):")
            for prompt, expected, output in latest['sample_outputs']:
                print(f"  '{prompt}...' -> '{output}' (expected: '{expected}')")

        print("=" * 90 + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='TKS Training with Real-Time Leaderboard')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--data', type=str, default='datasets/jsonl/tks_v1_nl2tks_mix.jsonl')
    parser.add_argument('--output-dir', type=str, default='checkpoints/v2_leaderboard')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()

    # Test prompts for leaderboard
    test_prompts = [
        "Goal: A0.1 Divine Blueprint",
        "Goal: A1.2 Material Form",
        "Map concept: B2.1 Positive Belief",
        "Map concept: C5.1 Emotional Receptivity",
        "Map concept: D2.1 Physical Health",
        "Calculate TKS: A1.1 joining with B2.1",
        "Goal: B4.1 Thought Intensity",
        "TKS Operation: C3.1 combined with D4.1",
    ]

    test_expected = [
        "(A0:B2)",
        "(A1:D3)",
        "(B2^0)",
        "(C5^0)",
        "(D2^0)",
        "(A1^?) + (B2^?)",
        "(B4:B7)",
        "(C3^?) + (D4^?)",
    ]

    print("\n" + "=" * 70)
    print("TKS TRAINING WITH REAL-TIME LEADERBOARD")
    print("=" * 70)
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output_dir}")
    print("=" * 70 + "\n")

    # Build training command
    cmd = [
        sys.executable, 'scripts/train_cuda.py',
        '--data', args.data,
        '--output-dir', args.output_dir,
        '--epochs', '1',  # Train 1 epoch at a time for leaderboard updates
        '--batch-size', str(args.batch_size),
        '--model-type', 'noetic_v4',
        '--learning-rate', str(args.lr),
        '--disable-auxiliary-losses',
        '--num-workers', '0',
    ]

    # Initialize leaderboard
    leaderboard = RealTimeLeaderboard(test_prompts, test_expected, args.data)

    # Training loop with leaderboard updates
    checkpoint_path = args.resume

    for epoch in range(args.epochs):
        print(f"\n>>> EPOCH {epoch + 1}/{args.epochs} <<<\n")

        # Build command for this epoch
        epoch_cmd = cmd.copy()
        if checkpoint_path and os.path.exists(checkpoint_path):
            epoch_cmd.extend(['--resume', checkpoint_path])

        # Run training for 1 epoch
        start_time = time.time()
        result = subprocess.run(epoch_cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time

        if result.returncode != 0:
            print(f"Training failed! Error:\n{result.stderr[-2000:]}")
            break

        # Parse training output for losses
        train_loss = 0.0
        eval_loss = 0.0
        for line in result.stdout.split('\n'):
            if 'train_loss' in line.lower() or 'loss:' in line.lower():
                try:
                    # Try to extract loss value
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if 'loss' in p.lower() and i + 1 < len(parts):
                            val = parts[i + 1].strip(',:')
                            train_loss = float(val)
                            break
                except:
                    pass
            if 'eval' in line.lower() and 'loss' in line.lower():
                try:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if 'loss' in p.lower() and i + 1 < len(parts):
                            val = parts[i + 1].strip(',:')
                            eval_loss = float(val)
                            break
                except:
                    pass

        # Update checkpoint path for next epoch
        checkpoint_path = os.path.join(args.output_dir, 'final_model.pt')

        # Load model for evaluation
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location='cuda', weights_only=False)

            # Load tokenizer
            tokenizer_data = ckpt.get('tokenizer')
            if tokenizer_data is None:
                tokenizer_path = os.path.join(args.output_dir, 'tokenizer.json')
                if os.path.exists(tokenizer_path):
                    with open(tokenizer_path, 'r') as f:
                        tokenizer_data = json.load(f)

            class SimpleTokenizer:
                def __init__(self, vocab):
                    self.vocab = vocab
                    self.id2tok = {v: k for k, v in vocab.items()}

            tokenizer = SimpleTokenizer(tokenizer_data)

            # Load model
            config = TKSNoeticLMConfig(vocab_size=len(tokenizer.vocab))
            model = TKSNoeticLM(config)
            model.load_state_dict(ckpt['model_state_dict'])
            model = model.cuda().eval()

            # Get metrics from checkpoint
            metrics = ckpt.get('metrics', {})
            train_loss = metrics.get('train_losses', [train_loss])[-1] if metrics.get('train_losses') else train_loss
            eval_loss = metrics.get('eval_losses', [eval_loss])[-1] if metrics.get('eval_losses') else eval_loss

            # Update leaderboard
            entry = leaderboard.update(epoch + 1, train_loss, eval_loss, model, tokenizer, 'cuda')

            # Display updated leaderboard
            leaderboard.display()

            print(f"Epoch {epoch + 1} completed in {elapsed:.1f}s")

            # Clean up
            del model
            torch.cuda.empty_cache()
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")

    # Save leaderboard history
    history_path = os.path.join(args.output_dir, 'leaderboard_history.json')
    with open(history_path, 'w') as f:
        json.dump(leaderboard.history, f, indent=2)
    print(f"\nLeaderboard history saved to: {history_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    if leaderboard.history:
        best = max(leaderboard.history, key=lambda x: x['syntax_valid_pct'] + x['exact_match_pct'])
        print(f"Best Epoch: {best['epoch']}")
        print(f"  Syntax Valid: {best['syntax_valid_pct']:.1f}%")
        print(f"  Exact Match: {best['exact_match_pct']:.1f}%")
        print(f"  Structure Valid: {best['structure_valid_pct']:.1f}%")
    print(f"\nCheckpoint: {checkpoint_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
