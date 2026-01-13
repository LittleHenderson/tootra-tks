"""
TKS Hybrid Tokenizer
====================

Combines BPE for natural language with atomic TKS special tokens.

Key Design Principles:
1. TKS tokens are NEVER split (e.g., "A0", "(A0:B2)" stay atomic)
2. Natural language uses standard BPE
3. Seamless encoding/decoding between NL and TKS
4. Preserves TKS structure for model to learn patterns

Usage:
    tokenizer = TKSHybridTokenizer.from_pretrained("path/to/tokenizer")
    tokens = tokenizer.encode("The expression (A0:B2) means connection")
    text = tokenizer.decode(tokens)
"""

import json
import re
import os
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, field
import hashlib


@dataclass
class TKSHybridTokenizerConfig:
    """Configuration for the hybrid tokenizer."""

    # Base BPE vocab size (for natural language)
    base_vocab_size: int = 8000

    # Special control tokens
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
    bos_token: str = "<BOS>"
    eos_token: str = "<EOS>"
    sep_token: str = " => "  # TKS separator (kept as string for compatibility)
    mask_token: str = "<MASK>"

    # TKS notation settings
    max_sephirot_index: int = 10  # A0-A9, B0-B9, etc.
    max_noetic_level: int = 9     # ^0 to ^9
    worlds: str = "ABCD"          # The four worlds


class TKSHybridTokenizer:
    """
    Hybrid tokenizer that handles both natural language (BPE) and TKS notation.

    TKS tokens are detected via regex and mapped to special token IDs.
    Everything else goes through the BPE vocabulary.
    """

    # Regex patterns for TKS notation
    TKS_PATTERNS = {
        # Full sephirot with optional noetic level: A0, B5^3, C2.1
        'sephirot': re.compile(r'\b([ABCD])(\d+)(?:\.(\d+))?(?:\^(\d+))?\b'),

        # TKS operators
        'operators': re.compile(r'(\+T|-T|\*T|/T|->|<-|\+|-|\*|/)'),

        # Structural: parentheses, path marker
        'structural': re.compile(r'(\(|\)|:|\^)'),

        # Full TKS expression in parentheses
        'expression': re.compile(r'\([ABCD]\d+(?:\.\d+)?(?:\^?\d+)?(?::[ABCD]\d+(?:\.\d+)?(?:\^?\d+)?)*\)'),
    }

    def __init__(self, config: Optional[TKSHybridTokenizerConfig] = None):
        self.config = config or TKSHybridTokenizerConfig()

        # Initialize vocabularies
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        # Track special token ranges
        self.special_token_ids: Set[int] = set()
        self.tks_token_ids: Set[int] = set()

        # Build initial vocabulary
        self._build_vocabulary()

    def _build_vocabulary(self):
        """Build the complete vocabulary with special tokens first."""
        idx = 0

        # 1. Control tokens (0-5)
        control_tokens = [
            self.config.pad_token,
            self.config.unk_token,
            self.config.bos_token,
            self.config.eos_token,
            "<SEP>",  # Explicit separator token
            self.config.mask_token,
        ]
        for token in control_tokens:
            self.vocab[token] = idx
            self.special_token_ids.add(idx)
            idx += 1

        # 2. World markers
        for world in self.config.worlds:
            token = f"<WORLD_{world}>"
            self.vocab[token] = idx
            self.tks_token_ids.add(idx)
            idx += 1

        # 3. Sephirot tokens (A0-A9, B0-B9, C0-C9, D0-D9)
        for world in self.config.worlds:
            for i in range(self.config.max_sephirot_index):
                token = f"{world}{i}"
                self.vocab[token] = idx
                self.tks_token_ids.add(idx)
                idx += 1

        # 4. Extended sephirot (A0.1-A9.9, etc.) - common sub-indices
        for world in self.config.worlds:
            for i in range(self.config.max_sephirot_index):
                for sub in range(1, 10):  # .1 to .9
                    token = f"{world}{i}.{sub}"
                    self.vocab[token] = idx
                    self.tks_token_ids.add(idx)
                    idx += 1

        # 5. Noetic level markers
        for level in range(self.config.max_noetic_level + 1):
            token = f"^{level}"
            self.vocab[token] = idx
            self.tks_token_ids.add(idx)
            idx += 1

        # 6. TKS Operators
        operators = ["+", "-", "+T", "-T", "*T", "/T", "->", "<-", "*", "/"]
        for op in operators:
            self.vocab[op] = idx
            self.tks_token_ids.add(idx)
            idx += 1

        # 7. Structural tokens
        structural = ["(", ")", ":", "=>"]
        for s in structural:
            self.vocab[s] = idx
            self.tks_token_ids.add(idx)
            idx += 1

        # 8. RPM markers
        rpm_tokens = ["<RPM_DESIRE>", "<RPM_WISDOM>", "<RPM_POWER>"]
        for token in rpm_tokens:
            self.vocab[token] = idx
            self.tks_token_ids.add(idx)
            idx += 1

        self.tks_vocab_end = idx

        # 9. Character-level fallback for now (will be BPE later)
        # Add ASCII printable characters
        for c in range(32, 127):
            char = chr(c)
            if char not in self.vocab:
                self.vocab[char] = idx
                idx += 1

        # Add common punctuation and whitespace
        for char in ['\n', '\t', ' ']:
            if char not in self.vocab:
                self.vocab[char] = idx
                idx += 1

        # Build reverse mapping
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        # Store vocab size
        self.vocab_size = len(self.vocab)

    def _tokenize_tks(self, text: str) -> List[str]:
        """
        Tokenize text, treating TKS notation as atomic tokens.

        Strategy:
        1. Find all TKS patterns and mark their positions
        2. Split text at TKS boundaries
        3. Tokenize non-TKS parts character by character (or BPE later)
        4. Reassemble with TKS tokens intact
        """
        tokens = []
        pos = 0

        # Find all TKS tokens with their positions
        tks_matches = []

        # Match sephirot (with optional sub-index and noetic level)
        for match in re.finditer(r'[ABCD]\d+(?:\.\d+)?(?:\^\d+)?', text):
            tks_matches.append((match.start(), match.end(), match.group()))

        # Match operators
        for match in re.finditer(r'\+T|-T|\*T|/T|->|<-|=>|\+|-|\*|/', text):
            tks_matches.append((match.start(), match.end(), match.group()))

        # Match structural
        for match in re.finditer(r'[():]', text):
            tks_matches.append((match.start(), match.end(), match.group()))

        # Match noetic levels standalone
        for match in re.finditer(r'\^\d+', text):
            tks_matches.append((match.start(), match.end(), match.group()))

        # Sort by position
        tks_matches.sort(key=lambda x: x[0])

        # Remove overlapping matches (keep longest)
        filtered_matches = []
        for start, end, token in tks_matches:
            if not filtered_matches or start >= filtered_matches[-1][1]:
                filtered_matches.append((start, end, token))

        # Build token list
        for start, end, tks_token in filtered_matches:
            # Add characters before this TKS token
            if pos < start:
                for char in text[pos:start]:
                    tokens.append(char)

            # Add the TKS token
            tokens.append(tks_token)
            pos = end

        # Add remaining characters
        if pos < len(text):
            for char in text[pos:]:
                tokens.append(char)

        return tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text (can contain TKS notation)
            add_special_tokens: Whether to add BOS/EOS

        Returns:
            List of token IDs
        """
        tokens = self._tokenize_tks(text)

        ids = []
        if add_special_tokens:
            ids.append(self.vocab[self.config.bos_token])

        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                # Try character-by-character
                for char in token:
                    ids.append(self.vocab.get(char, self.vocab[self.config.unk_token]))

        if add_special_tokens:
            ids.append(self.vocab[self.config.eos_token])

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.

        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip PAD, BOS, EOS, etc.

        Returns:
            Decoded text string
        """
        tokens = []
        skip_ids = {
            self.vocab[self.config.pad_token],
            self.vocab[self.config.bos_token],
            self.vocab[self.config.eos_token],
        } if skip_special_tokens else set()

        for id in ids:
            if id in skip_ids:
                continue
            token = self.id_to_token.get(id, self.config.unk_token)
            tokens.append(token)

        return ''.join(tokens)

    def encode_tks_expression(self, expr: str) -> List[int]:
        """
        Encode a pure TKS expression (optimized path).

        Args:
            expr: TKS expression like "(A0:B2)" or "(C5^3) + (D2^3)"

        Returns:
            List of token IDs
        """
        return self.encode(expr, add_special_tokens=False)

    def get_tks_token_ids(self) -> Set[int]:
        """Return the set of all TKS-specific token IDs."""
        return self.tks_token_ids.copy()

    def is_tks_token(self, token_id: int) -> bool:
        """Check if a token ID is a TKS token."""
        return token_id in self.tks_token_ids

    def save(self, path: str):
        """Save tokenizer to directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save vocab
        with open(path / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=2, ensure_ascii=False)

        # Save config
        config_dict = {
            "base_vocab_size": self.config.base_vocab_size,
            "pad_token": self.config.pad_token,
            "unk_token": self.config.unk_token,
            "bos_token": self.config.bos_token,
            "eos_token": self.config.eos_token,
            "sep_token": self.config.sep_token,
            "max_sephirot_index": self.config.max_sephirot_index,
            "max_noetic_level": self.config.max_noetic_level,
            "worlds": self.config.worlds,
            "vocab_size": self.vocab_size,
            "tks_vocab_end": self.tks_vocab_end,
        }
        with open(path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)

        # Save hash for integrity
        vocab_str = json.dumps(self.vocab, sort_keys=True)
        vocab_hash = hashlib.sha256(vocab_str.encode()).hexdigest()
        with open(path / "vocab_hash.txt", "w") as f:
            f.write(vocab_hash)

    @classmethod
    def from_pretrained(cls, path: str) -> "TKSHybridTokenizer":
        """Load tokenizer from directory."""
        path = Path(path)

        # Load config
        with open(path / "config.json", "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        config = TKSHybridTokenizerConfig(
            base_vocab_size=config_dict.get("base_vocab_size", 8000),
            pad_token=config_dict.get("pad_token", "<PAD>"),
            unk_token=config_dict.get("unk_token", "<UNK>"),
            bos_token=config_dict.get("bos_token", "<BOS>"),
            eos_token=config_dict.get("eos_token", "<EOS>"),
            sep_token=config_dict.get("sep_token", " => "),
            max_sephirot_index=config_dict.get("max_sephirot_index", 10),
            max_noetic_level=config_dict.get("max_noetic_level", 9),
            worlds=config_dict.get("worlds", "ABCD"),
        )

        tokenizer = cls(config)

        # Override vocab if saved
        if (path / "vocab.json").exists():
            with open(path / "vocab.json", "r", encoding="utf-8") as f:
                tokenizer.vocab = json.load(f)
                tokenizer.id_to_token = {v: k for k, v in tokenizer.vocab.items()}
                tokenizer.vocab_size = len(tokenizer.vocab)

        return tokenizer

    def __len__(self) -> int:
        return self.vocab_size


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_tokenizer_for_phase1() -> TKSHybridTokenizer:
    """Create a tokenizer configured for Phase 1 training."""
    config = TKSHybridTokenizerConfig(
        base_vocab_size=8000,
        max_sephirot_index=10,
        max_noetic_level=9,
    )
    return TKSHybridTokenizer(config)


def demo_tokenizer():
    """Demonstrate tokenizer functionality."""
    tokenizer = create_tokenizer_for_phase1()

    print("=" * 60)
    print("TKS HYBRID TOKENIZER DEMO")
    print("=" * 60)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"TKS tokens: {len(tokenizer.tks_token_ids)}")
    print()

    test_cases = [
        "The expression (A0:B2) represents divine connection.",
        "Goal: A0.1 Divine Blueprint => (A0:B2)",
        "(C5^3) + (D2^3) shows emotional-physical transformation",
        "Map concept: B2.1 Positive Belief",
        "Use +T for forward transformation and -T for reverse.",
    ]

    for text in test_cases:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        tks_count = sum(1 for id in ids if tokenizer.is_tks_token(id))

        print(f"Input:   {text}")
        print(f"Tokens:  {len(ids)} ({tks_count} TKS)")
        print(f"IDs:     {ids[:15]}{'...' if len(ids) > 15 else ''}")
        print(f"Decoded: {decoded}")
        print()


if __name__ == "__main__":
    demo_tokenizer()
