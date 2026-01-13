"""
TKS Deterministic Tokenizer

Ensures consistent token-to-id mapping across Training and Evaluation.
"""

import json
import os
import torch
from typing import List, Dict, Optional

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

class TKSTokenizer:
    def __init__(self, vocab: Dict[str, int] = None, max_length: int = 256):
        self.max_length = max_length
        if vocab:
            self.vocab = vocab
            self.id_to_token = {v: k for k, v in vocab.items()}
        else:
            # Initialize with Defaults (TKS Specific)
            self.vocab = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
            next_id = len(SPECIAL_TOKENS)
            
            # TKS Elements
            for world in ['A', 'B', 'C', 'D']:
                for noetic in range(1, 11):
                    self.vocab[f"{world}{noetic}"] = next_id
                    next_id += 1
            
            # Operators
            for op in ['+', '-', '+T', '-T', '->', '<-', '*T', '/T', 'o']:
                self.vocab[op] = next_id
                next_id += 1
                
            # Characters
            chars = ' .,!?;:\'"()-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789^_*/<>='
            for c in chars:
                if c not in self.vocab:
                    self.vocab[c] = next_id
                    next_id += 1
                    
            self.id_to_token = {v: k for k, v in self.vocab.items()}
            
    @property
    def actual_vocab_size(self) -> int:
        return len(self.vocab)
    
    @property
    def pad_token_id(self) -> int:
        return self.vocab["<PAD>"]

    def tokenize(self, text: str) -> List[int]:
        # Simple character-level tokenization for now, 
        # relying on the vocab containing chars.
        # Future improvement: Regex split for TKS symbols.
        ids = [self.vocab["<BOS>"]]
        for char in text[:self.max_length-2]:
            ids.append(self.vocab.get(char, self.vocab["<UNK>"]))
        ids.append(self.vocab["<EOS>"])
        
        # Pad
        while len(ids) < self.max_length:
            ids.append(self.vocab["<PAD>"])
            
        return ids[:self.max_length]

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        chars = []
        for i in ids:
            if isinstance(i, torch.Tensor):
                i = i.item()
            token = self.id_to_token.get(i, "")
            if skip_special and token in SPECIAL_TOKENS:
                continue
            if skip_special and i == 0: # PAD
                continue
            chars.append(token)
        return "".join(chars)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> "TKSTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both formats:
        # 1. New format: plain dict {token: id}
        # 2. Old format: {"token_to_id": {token: id}, "vocab_size": ..., ...}
        if "token_to_id" in data:
            vocab = data["token_to_id"]
            max_length = data.get("max_length", 256)
        else:
            vocab = data
            max_length = 256

        instance = cls(vocab=vocab, max_length=max_length)
        return instance