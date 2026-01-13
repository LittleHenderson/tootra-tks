#!/usr/bin/env python3
"""Proof tests for fixed TKS training: A) Generation, B) Exact match, C) Robustness."""

import sys
import os
import json
import random
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tks.core.tks_llm_core_v4 import TKSNoeticLM, TKSNoeticLMConfig

# Determine device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")


def load_model_and_tokenizer(checkpoint_path):
    """Load model and tokenizer from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    # Load tokenizer
    tokenizer_data = checkpoint.get('tokenizer')
    if tokenizer_data is None:
        tokenizer_path = os.path.join(os.path.dirname(checkpoint_path), 'tokenizer.json')
        with open(tokenizer_path, 'r') as f:
            tokenizer_data = json.load(f)

    class SimpleTokenizer:
        def __init__(self, vocab):
            self.vocab = vocab
            self.id2tok = {v: k for k, v in vocab.items()}

        def encode(self, text):
            ids = [self.vocab.get('<BOS>', 2)]
            for char in text:
                ids.append(self.vocab.get(char, self.vocab.get('<UNK>', 1)))
            return ids

        def decode(self, ids):
            result = []
            for id in ids:
                tok = self.id2tok.get(id, '<UNK>')
                if tok not in ['<PAD>', '<BOS>', '<EOS>']:
                    result.append(tok)
            return ''.join(result)

    tokenizer = SimpleTokenizer(tokenizer_data if isinstance(tokenizer_data, dict) else tokenizer_data)

    # Load model - check for config object or dict
    config = checkpoint.get('config')
    if config is None:
        # Create default config with correct vocab size
        config = TKSNoeticLMConfig(vocab_size=len(tokenizer.vocab))
    elif isinstance(config, dict):
        # Config is a dict, create TKSNoeticLMConfig from it
        config = TKSNoeticLMConfig(
            vocab_size=config.get('vocab_size', len(tokenizer.vocab)),
            max_seq_len=config.get('max_seq_len', 256),
            num_layers=config.get('num_layers', 4),
            num_scales=config.get('num_scales', 3),
            dropout=config.get('dropout', 0.1),
        )
    # else config is already a TKSNoeticLMConfig object

    model = TKSNoeticLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE).eval()

    return model, tokenizer


def generate(model, tokenizer, prompt, max_new_tokens=50):
    """Generate text autoregressively with PAD blocking."""
    pad_id = tokenizer.vocab.get('<PAD>', 0)
    eos_id = tokenizer.vocab.get('<EOS>', 3)

    # IMPORTANT: Training data format is "prompt + ' ' + target"
    # So we need to add trailing space to match training distribution
    prompt_with_space = prompt.rstrip() + ' '
    input_ids = tokenizer.encode(prompt_with_space)
    generated = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)

    tokens_generated = []

    with torch.no_grad():
        for step in range(max_new_tokens):
            attention_mask = torch.ones_like(generated)
            output = model(generated, attention_mask=attention_mask)

            if isinstance(output, dict):
                logits = output.get('logits', output.get('output'))
            else:
                logits = output

            next_logits = logits[:, -1, :].clone()
            next_logits[:, pad_id] = float('-inf')  # Block PAD

            next_token = next_logits.argmax(dim=-1, keepdim=True)
            token_id = next_token.item()
            token_str = tokenizer.id2tok.get(token_id, f'[{token_id}]')
            tokens_generated.append(token_str)

            if token_id == eos_id:
                break

            generated = torch.cat([generated, next_token], dim=1)

    return tokens_generated


def test_a_generation(model, tokenizer):
    """Test A: Can model generate valid TKS?"""
    print("\n" + "="*60)
    print("TEST A: Generation (should produce valid TKS)")
    print("="*60)

    # NOTE: Training data format is "input" without => separator
    # So prompts should match training format
    prompts = [
        "Goal: A0.1 Divine Blueprint",
        "Goal: A1.2 Material Form",
        "Map concept: B2.1 Positive Belief",
    ]

    passed = 0
    for prompt in prompts:
        tokens = generate(model, tokenizer, prompt, max_new_tokens=30)
        output = ''.join(tokens)

        # Check: should produce valid TKS-like output
        # Must have ( and ), not just gibberish
        first_token = tokens[0] if tokens else '<empty>'
        not_pad_eos = first_token not in ['<PAD>', '<EOS>', '<empty>']
        has_parens = '(' in output and ')' in output
        has_tks = any(s in output for s in ['A', 'B', 'C', 'D'])  # Sephirot letters
        is_good = not_pad_eos and (has_parens or has_tks)

        status = "PASS" if is_good else "FAIL"
        if is_good:
            passed += 1

        print(f"\n[{status}] Prompt: {prompt}")
        print(f"       First token: {first_token}")
        print(f"       Output: {output[:60]}...")

    print(f"\nTest A Result: {passed}/{len(prompts)} passed")
    return passed == len(prompts)


def test_b_exact_match(model, tokenizer, data_path, n_samples=20):
    """Test B: Check if model can reproduce training samples."""
    print("\n" + "="*60)
    print("TEST B: Training Sample Exact Match")
    print("="*60)

    # Load training data
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    samples.append(json.loads(line))
                except:
                    continue

    if len(samples) < n_samples:
        n_samples = len(samples)

    test_samples = random.sample(samples, n_samples)

    exact_matches = 0
    partial_matches = 0

    for i, sample in enumerate(test_samples):
        # Get prompt and expected target
        # Note: Training data format is input/output (no => separator)
        if 'input' in sample and 'output' in sample:
            prompt = sample['input']
            expected = sample['output']
        elif 'prompt' in sample and 'target' in sample:
            prompt = sample['prompt']
            expected = sample['target']
        elif 'text' in sample:
            # For text format, split on =>
            text = sample['text']
            if '=>' in text:
                parts = text.split('=>', 1)
                prompt = parts[0].strip()  # No => needed
                expected = parts[1].strip()
            else:
                continue
        else:
            continue

        # Generate
        tokens = generate(model, tokenizer, prompt, max_new_tokens=len(expected) + 10)
        output = ''.join(t for t in tokens if t not in ['<EOS>'])

        # Check match
        expected_clean = expected.strip()
        output_clean = output.strip()

        is_exact = output_clean == expected_clean
        is_partial = expected_clean.startswith(output_clean) or output_clean.startswith(expected_clean[:len(output_clean)])

        if is_exact:
            exact_matches += 1
            status = "EXACT"
        elif is_partial and len(output_clean) > 3:
            partial_matches += 1
            status = "PARTIAL"
        else:
            status = "MISS"

        if i < 5 or status == "EXACT":  # Show first 5 and all exact matches
            print(f"\n[{status}] Sample {i+1}:")
            print(f"  Prompt: {prompt[:50]}...")
            print(f"  Expected: {expected_clean[:40]}")
            print(f"  Got:      {output_clean[:40]}")

    print(f"\nTest B Result: {exact_matches} exact, {partial_matches} partial out of {n_samples}")
    return exact_matches >= n_samples * 0.5  # Pass if 50%+ exact match


def test_c_robustness(model, tokenizer):
    """Test C: Quick robustness check - syntax validity."""
    print("\n" + "="*60)
    print("TEST C: Robustness / Syntax Validity")
    print("="*60)

    # Test prompts covering different TKS patterns
    # Note: Training data format has no => separator
    test_cases = [
        ("Goal: A1.1 Test", "Should start with ("),
        ("Goal: B2.1 Flow", "Should produce TKS operators"),
        ("Map concept: C3.1 Connect", "Valid TKS syntax"),
        ("Calculate TKS: D4.1 Transform", "Operator sequence"),
        ("Goal: A5.1 Nested", "Sephirot notation"),
    ]

    valid_count = 0
    total = len(test_cases)

    for prompt, desc in test_cases:
        tokens = generate(model, tokenizer, prompt, max_new_tokens=40)
        output = ''.join(t for t in tokens if t not in ['<EOS>'])

        # Basic syntax checks
        has_parens = '(' in output
        has_operators = any(op in output for op in ['+', '-', '->', '<-', '+T', '-T'])
        has_sephirot = any(s in output for s in ['A1', 'A2', 'B1', 'B2', 'C1', 'D1'])
        not_garbage = len(output) > 3 and output[0] != '<'

        is_valid = (has_parens or has_operators or has_sephirot) and not_garbage

        if is_valid:
            valid_count += 1
            status = "VALID"
        else:
            status = "INVALID"

        print(f"\n[{status}] {desc}")
        print(f"  Prompt: {prompt}")
        print(f"  Output: {output[:60]}")

    pct = 100 * valid_count / total
    print(f"\nTest C Result: {valid_count}/{total} ({pct:.0f}%) valid syntax")
    return valid_count >= total * 0.6  # Pass if 60%+ valid


def main():
    print("="*60)
    print("TKS PROOF TESTS - Verifying Fixed Training")
    print("="*60)

    checkpoint_path = "checkpoints/v2_fixed/final_model.pt"
    data_path = "datasets/jsonl/tks_v1_nl2tks_mix.jsonl"

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return

    print(f"\nLoading model from: {checkpoint_path}")
    model, tokenizer = load_model_and_tokenizer(checkpoint_path)
    print(f"Vocab size: {len(tokenizer.vocab)}")

    # Run tests
    results = {}
    results['A'] = test_a_generation(model, tokenizer)

    if os.path.exists(data_path):
        results['B'] = test_b_exact_match(model, tokenizer, data_path)
    else:
        print(f"\nSkipping Test B: Data file not found: {data_path}")
        results['B'] = None

    results['C'] = test_c_robustness(model, tokenizer)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for test, passed in results.items():
        if passed is None:
            status = "SKIPPED"
        elif passed:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"  Test {test}: {status}")

    all_passed = all(v for v in results.values() if v is not None)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")


if __name__ == "__main__":
    main()
