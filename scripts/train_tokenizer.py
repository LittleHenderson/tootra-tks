"""
Train BPE Tokenizer for TKS General Model.
Combines JSONL datasets and Manuals for a rich vocabulary.
"""

import os
import glob
import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def get_text_iterator():
    # 1. Manuals
    manuals = glob.glob("*.md") + glob.glob("*.txt")
    for m in manuals:
        print(f"Reading {m}...")
        try:
            with open(m, 'r', encoding='utf-8') as f:
                yield f.read()
        except:
            pass

    # 2. Datasets
    datasets = [
        "datasets/jsonl/tks_mass_v2.jsonl",
        "output/teacher_grounding_curriculum.jsonl",
        "output/rebalanced_mix.jsonl"
    ]
    
    for d in datasets:
        if os.path.exists(d):
            print(f"Reading {d}...")
            with open(d, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if 'input' in item: yield item['input']
                        if 'output' in item: yield item['output']
                        if 'prompt' in item: yield item['prompt']
                        if 'target' in item: yield item['target']
                        if 'story' in item: yield item['story']
                    except:
                        pass

def train_tokenizer():
    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = Whitespace()
    
    trainer = BpeTrainer(
        vocab_size=16384, 
        special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>"]
    )
    
    print("Training tokenizer...")
    tokenizer.train_from_iterator(get_text_iterator(), trainer)
    
    output_path = "tokenizer_v5.json"
    tokenizer.save(output_path)
    print(f"Tokenizer saved to {output_path}")

if __name__ == "__main__":
    train_tokenizer()
