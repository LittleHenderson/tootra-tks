#!/usr/bin/env python3
"""
Test tokenizer coverage for dual-task story-equation conversion.

Verifies that the tokenizer can handle all required tokens:
- Base elements: A1-A10, B1-B10, C1-C10, D1-D10
- Extended notation senses: ^1-^9
- Extended notation foundations: _d1-_d7
- Operators: +, -, +T, -T, ->, <-, *T, /T, o
- Direction tags for bidirectional training

Usage:
    python scripts/test_tokenizer_coverage.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Simple tokenizer (from quick_train.py)
class SimpleTokenizer:
    def __init__(self, vocab_size=1000, max_length=256):
        self.vocab_size = vocab_size
        self.max_length = max_length
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

    def tokenize(self, text):
        tokens = [2]  # BOS
        for c in text[:self.max_length-2]:
            tokens.append(self.token_to_id.get(c, 1))
        tokens.append(3)  # EOS
        while len(tokens) < self.max_length:
            tokens.append(0)  # PAD
        return tokens[:self.max_length]

    def decode(self, token_ids):
        """Decode token IDs back to text."""
        chars = []
        for tid in token_ids:
            if tid == 0:  # PAD
                continue
            elif tid == 2:  # BOS
                continue
            elif tid == 3:  # EOS
                break
            else:
                chars.append(self.id_to_token.get(tid, '<UNK>'))
        return ''.join(chars)


def test_base_elements():
    """Test coverage of base elements (A1-D10)."""
    print("\n" + "="*70)
    print("TEST 1: Base Elements Coverage")
    print("="*70)

    tokenizer = SimpleTokenizer()
    worlds = ['A', 'B', 'C', 'D']
    noetics = range(1, 11)

    missing = []
    for world in worlds:
        for noetic in noetics:
            element = f"{world}{noetic}"
            if element not in tokenizer.token_to_id:
                missing.append(element)

    if missing:
        print(f"[FAIL] Missing {len(missing)} base elements: {missing[:10]}...")
        return False
    else:
        print(f"[PASS] All 40 base elements present in tokenizer")
        return True


def test_operators():
    """Test coverage of TKS operators."""
    print("\n" + "="*70)
    print("TEST 2: Operator Coverage")
    print("="*70)

    tokenizer = SimpleTokenizer()
    operators = ['+', '-', '+T', '-T', '->', '<-', '*T', '/T', 'o']

    missing = []
    for op in operators:
        if op not in tokenizer.token_to_id:
            missing.append(op)

    if missing:
        print(f"[FAIL] Missing {len(missing)} operators: {missing}")
        return False
    else:
        print(f"[PASS] All 9 operators present in tokenizer")
        return True


def test_extended_notation():
    """Test coverage of extended notation (senses and foundations)."""
    print("\n" + "="*70)
    print("TEST 3: Extended Notation Coverage (^k and _dF)")
    print("="*70)

    tokenizer = SimpleTokenizer()

    # Test senses (^1-^9)
    missing_chars = []
    if '^' not in tokenizer.token_to_id:
        missing_chars.append('^')

    # Test foundations (_d1-_d7) - need _, d, and digits
    for char in ['_', 'd'] + list('1234567'):
        if char not in tokenizer.token_to_id:
            missing_chars.append(char)

    if missing_chars:
        print(f"[FAIL] Missing extended notation chars: {missing_chars}")
        return False
    else:
        print(f"[PASS] All extended notation characters present (^, _, d, digits)")

        # Test actual extended notation parsing
        test_elements = [
            "A1^3",      # Sense only
            "B4_d5",     # Foundation only
            "C7^9_d7",   # Both sense and foundation
            "D10^1_d1",  # With noetic 10
        ]

        print("\n  Testing extended notation elements:")
        for elem in test_elements:
            tokens = tokenizer.tokenize(elem)
            decoded = tokenizer.decode(tokens)
            match = elem == decoded
            status = "[OK]" if match else "[X]"
            print(f"    {status} {elem:15} -> {decoded}")

        return True


def test_equation_roundtrip():
    """Test roundtrip encoding/decoding of TKS equations."""
    print("\n" + "="*70)
    print("TEST 4: Equation Roundtrip")
    print("="*70)

    tokenizer = SimpleTokenizer()

    test_equations = [
        "A1 + B2",                    # Simple
        "C4^2 -T C9^7_d1",           # Extended notation
        "A1^1 /T B7^4 +T C10 -> D8", # Complex
        "B5^7_d7 - B9 -> B4_d6 o B6", # Multiple foundations
    ]

    all_pass = True
    print("\n  Testing equation roundtrip:")
    for eq in test_equations:
        tokens = tokenizer.tokenize(eq)
        decoded = tokenizer.decode(tokens)
        match = eq == decoded
        status = "[OK]" if match else "[X]"
        if not match:
            all_pass = False
        print(f"    {status} {eq:30} -> {decoded}")

    if all_pass:
        print(f"\n[PASS] All equations encode/decode correctly")
    else:
        print(f"\n[FAIL] Some equations failed roundtrip")

    return all_pass


def test_direction_tags():
    """Test that direction tags can be encoded."""
    print("\n" + "="*70)
    print("TEST 5: Direction Tags")
    print("="*70)

    tokenizer = SimpleTokenizer()

    direction_tags = [
        "story_to_eq",
        "eq_to_story",
        "story_to_equation",
        "equation_to_story",
    ]

    all_pass = True
    print("\n  Testing direction tag encoding:")
    for tag in direction_tags:
        tokens = tokenizer.tokenize(tag)
        decoded = tokenizer.decode(tokens)
        match = tag == decoded
        status = "[OK]" if match else "[X]"
        if not match:
            all_pass = False
        print(f"    {status} {tag:20} -> {decoded}")

    if all_pass:
        print(f"\n[PASS] All direction tags encode correctly")
    else:
        print(f"\n[FAIL] Some direction tags failed")

    return all_pass


def test_vocab_size():
    """Report actual vocabulary size."""
    print("\n" + "="*70)
    print("TEST 6: Vocabulary Size")
    print("="*70)

    tokenizer = SimpleTokenizer()

    print(f"\n  Configured vocab size:  {tokenizer.vocab_size}")
    print(f"  Actual vocab size:      {tokenizer.actual_vocab_size}")
    print(f"  Unused capacity:        {tokenizer.vocab_size - tokenizer.actual_vocab_size}")

    # Breakdown
    special_tokens = 4  # PAD, UNK, BOS, EOS
    base_elements = 40  # A1-A10, B1-B10, C1-C10, D1-D10
    operators = 9       # +, -, +T, -T, ->, <-, *T, /T, o

    print(f"\n  Token breakdown:")
    print(f"    Special tokens:       {special_tokens}")
    print(f"    Base elements:        {base_elements}")
    print(f"    Operators:            {operators}")
    print(f"    Characters:           ~{tokenizer.actual_vocab_size - special_tokens - base_elements - operators}")

    if tokenizer.actual_vocab_size <= tokenizer.vocab_size:
        print(f"\n[PASS] Actual vocab size fits within configured size")
        return True
    else:
        print(f"\n[FAIL] Actual vocab size exceeds configured size!")
        return False


def main():
    """Run all tokenizer coverage tests."""
    print("\n" + "="*70)
    print("TKS TOKENIZER COVERAGE TEST")
    print("Testing dual-task story-equation conversion requirements")
    print("="*70)

    results = []

    # Run all tests
    results.append(("Base Elements", test_base_elements()))
    results.append(("Operators", test_operators()))
    results.append(("Extended Notation", test_extended_notation()))
    results.append(("Equation Roundtrip", test_equation_roundtrip()))
    results.append(("Direction Tags", test_direction_tags()))
    results.append(("Vocabulary Size", test_vocab_size()))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status:10} {test_name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print(f"\n  Total: {passed}/{total} tests passed")
    print("="*70)

    return 0 if all(p for _, p in results) else 1


if __name__ == '__main__':
    sys.exit(main())
