
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tokenizers import Tokenizer
from tks_llm_core_v7 import TKSGeneralLMv7, TKSGeneralConfigV7

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def diagnose_model():
    tokenizer = Tokenizer.from_file("tokenizer_v5.json")
    print(f"Tokenizer Vocab Size: {tokenizer.get_vocab_size()}")
    
    # Check special tokens
    specials = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
    for s in specials:
        tid = tokenizer.token_to_id(s)
        print(f"Token '{s}': {tid}")

    # Load model
    checkpoint_path = "checkpoints/v7_reset/v7_reset_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Check weight shapes
    emb_weight = checkpoint.get("token_emb.weight")
    head_weight = checkpoint.get("lm_head.weight")
    
    if emb_weight is not None:
        print(f"Model Embedding Shape: {emb_weight.shape}")
    if head_weight is not None:
        print(f"Model Head Shape: {head_weight.shape}")

    # Test encoding
    test_str = "Input: Test\nOutput:"
    ids = tokenizer.encode(test_str).ids
    print(f"Test encode '{test_str}': {ids}")
    
    # Test decoding of loop pattern
    loop_ids = tokenizer.encode(": : -> ->").ids
    print(f"Loop IDs: {loop_ids}")

if __name__ == "__main__":
    diagnose_model()
