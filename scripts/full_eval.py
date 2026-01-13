"""
Full Model Evaluation - All Task Types
"""

import argparse
import sys
import os
import json
import torch
from tokenizers import Tokenizer
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from tks_llm_core_v5 import TKSGeneralLM, TKSGeneralConfig
from eval.rpm_validator import RPMPathValidator, build_rpm_token_filter

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(checkpoint_path, tokenizer):
    config = TKSGeneralConfig(
        vocab_size=tokenizer.get_vocab_size(),
        hidden_dim=384,
        num_layers=12,
        num_scales=4
    )
    model = TKSGeneralLM(config).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model

def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=60,
    allowed_token_ids=None,
    require_colons=0,
):
    sep_id = tokenizer.token_to_id("<SEP>")
    eos_id = tokenizer.token_to_id("<EOS>")
    bos_id = tokenizer.token_to_id("<BOS>")
    pad_id = tokenizer.token_to_id("<PAD>")

    enc = tokenizer.encode(prompt + " <SEP> ")
    input_ids = [bos_id] + enc.ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)

    allowed_ids = None
    if allowed_token_ids:
        allowed_set = set(allowed_token_ids)
        allowed_set.update({eos_id})
        allowed_ids = torch.tensor(
            sorted(allowed_set),
            device=input_tensor.device,
            dtype=torch.long,
        )

    colon_count = 0
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(input_tensor)
            logits = out['logits'][:, -1, :]
            if allowed_ids is not None:
                mask = torch.full_like(logits, -float("inf"))
                mask[:, allowed_ids] = 0.0
                logits = logits + mask
            if pad_id is not None:
                logits[:, pad_id] = -1e9
            if require_colons and eos_id is not None and colon_count < require_colons:
                logits[:, eos_id] = -1e9

            next_token = logits.argmax(dim=-1).item()

            if next_token == eos_id or (pad_id is not None and next_token == pad_id):
                break

            input_tensor = torch.cat([
                input_tensor,
                torch.tensor([[next_token]], device=DEVICE)
            ], dim=1)
            if require_colons:
                try:
                    decoded = tokenizer.decode([next_token])
                except Exception:
                    decoded = ""
                if decoded:
                    colon_count += decoded.count(":")

    all_ids = input_tensor[0].tolist()
    try:
        sep_pos = all_ids.index(sep_id)
        output_ids = all_ids[sep_pos+1:]
    except ValueError:
        output_ids = all_ids[len(input_ids):]

    return tokenizer.decode(output_ids).strip()

def main():
    parser = argparse.ArgumentParser(description="Full model evaluation")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/v5_reconciled/epoch_60.pt",
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--tokenizer",
        default="tokenizer_v5.json",
        help="Tokenizer JSON path",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(f"FULL MODEL EVALUATION - {args.checkpoint}")
    print("=" * 70)

    tokenizer = Tokenizer.from_file(args.tokenizer)
    model = load_model(args.checkpoint, tokenizer)
    print(f"Model loaded on {DEVICE}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test Categories
    tests = {
        "GROUNDING - Elements": [
            "Define TKS Element A0",
            "Define TKS Element A5",
            "Define TKS Element B3",
            "Define TKS Element C7",
            "Define TKS Element D2",
            "What is B6 in TKS?",
        ],
        "GROUNDING - Noetics": [
            "Define TKS Noetic ^0",
            "Define TKS Noetic ^3",
            "Define TKS Noetic ^6",
            "Define TKS Noetic ^9",
        ],
        "GROUNDING - SubFoundations": [
            "Define TKS Sub-Foundation 1a",
            "Define TKS Sub-Foundation 2b",
            "Define TKS Sub-Foundation 4c",
            "Define TKS Sub-Foundation 7d",
        ],
        "OPERATOR Tasks": [
            "What is the TKS operator for addition?",
            "What is the TKS operator for transformation?",
            "Explain the + operator in TKS",
        ],
        "FM REGULATOR Tasks": [
            "FM regulator for: I feel anxious about tomorrow",
            "FM regulator for: My body is tired",
            "FM regulator for: I need to think clearly",
        ],
        "RPM Tasks": [
            "Create RPM for emotional balance",
            "What RPM targets mental clarity?",
        ],
        "INTERPRETATION Tasks": [
            "Interpret TKS: A3 + B5",
            "What does C2 - D1 mean?",
        ],
    }

    rpm_validator = RPMPathValidator(min_len=3, max_len=3)
    rpm_allowed_ids = build_rpm_token_filter(tokenizer)
    if not rpm_allowed_ids:
        rpm_allowed_ids = None
    rpm_total = 0
    rpm_valid = 0

    for category, prompts in tests.items():
        print(f"\n{'='*70}")
        print(f"{category}")
        print("=" * 70)
        for prompt in prompts:
            is_rpm = category.lower().startswith("rpm")
            output = generate(
                model,
                tokenizer,
                prompt,
                allowed_token_ids=rpm_allowed_ids if is_rpm else None,
                require_colons=2 if is_rpm else 0,
            )
            print(f"\nQ: {prompt}")
            print(f"A: {output[:120]}{'...' if len(output) > 120 else ''}")     
            if is_rpm:
                rpm_total += 1
                rpm_result = rpm_validator.validate(output, allow_extract=False)
                rpm_valid += 1 if rpm_result.is_valid else 0
                if rpm_result.candidate and rpm_result.candidate != output:
                    print(f"RPM path: {rpm_result.candidate}")
                print(f"RPM path valid: {rpm_result.is_valid}")

    if rpm_total:
        rate = 100 * rpm_valid / rpm_total
        print(f"\nRPM PATH VALIDITY: {rpm_valid}/{rpm_total} ({rate:.1f}%)")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
