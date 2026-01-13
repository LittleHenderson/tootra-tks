"""
Compare Original vs Reconciled Model on Grounding + Equation Tasks.
Canonical eval entry point for quick model comparisons.
"""

import sys
import os
import json
import torch
from tokenizers import Tokenizer
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_DIR = Path(__file__).parent / "eval"
sys.path.insert(0, str(EVAL_DIR))
from tks_llm_core_v5 import TKSGeneralLM, TKSGeneralConfig
from metrics import compare_tks_expressions, normalize_text, partial_text_match

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

def generate(model, tokenizer, prompt, max_new_tokens=50):
    sep_id = tokenizer.token_to_id("<SEP>")
    eos_id = tokenizer.token_to_id("<EOS>")
    bos_id = tokenizer.token_to_id("<BOS>")
    pad_id = tokenizer.token_to_id("<PAD>")

    enc = tokenizer.encode(prompt + " <SEP> ")
    input_ids = [bos_id] + enc.ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(input_tensor)
            logits = out['logits'][:, -1, :]
            if pad_id is not None:
                logits[:, pad_id] = -1e9
            next_token = logits.argmax(dim=-1).item()

            if next_token == eos_id or (pad_id is not None and next_token == pad_id):
                break

            input_tensor = torch.cat([
                input_tensor,
                torch.tensor([[next_token]], device=DEVICE)
            ], dim=1)

    all_ids = input_tensor[0].tolist()
    try:
        sep_pos = all_ids.index(sep_id)
        output_ids = all_ids[sep_pos+1:]
    except ValueError:
        output_ids = all_ids[len(input_ids):]

    return tokenizer.decode(output_ids).strip()

def main():
    print("=" * 70)
    print("MODEL COMPARISON: Original vs Reconciled")
    print("=" * 70)

    # Load tokenizer
    tokenizer = Tokenizer.from_file("tokenizer_v5.json")

    # Load both models
    print("\nLoading models...")
    original = load_model("checkpoints/v5_general/epoch_10.pt", tokenizer)
    print("  Original model loaded (v5_general/epoch_10.pt)")

    reconciled = load_model("checkpoints/v5_reconciled/epoch_60.pt", tokenizer)
    print("  Reconciled model loaded (v5_reconciled/epoch_60.pt)")

    # Load canonical definitions for ground truth
    canon_dir = Path("canon/normalized")

    with open(canon_dir / "canon_elements_40.json", encoding="utf-8") as f:
        elements = {e["symbol"]: e["definition"] for e in json.load(f)}

    with open(canon_dir / "canon_noetics_10.json", encoding="utf-8") as f:
        noetics = {n["id"]: f"{n['name']}: {n['meaning']}" for n in json.load(f)}

    with open(canon_dir / "canon_subfoundations_28.json", encoding="utf-8") as f:
        subfoundations = {sf["id"]: sf["name"] for sf in json.load(f)}

    # Test cases
    test_cases = [
        # Elements (grounding)
        {"task": "grounding", "prompt": "Define TKS Element A0", "expected": elements.get("A0", "N/A")},
        {"task": "grounding", "prompt": "Define TKS Element B5", "expected": elements.get("B5", "N/A")},
        {"task": "grounding", "prompt": "Define TKS Element C3", "expected": elements.get("C3", "N/A")},
        {"task": "grounding", "prompt": "Define TKS Element D7", "expected": elements.get("D7", "N/A")},
        # Noetics (grounding)
        {"task": "grounding", "prompt": "Define TKS Noetic ^1", "expected": noetics.get("^1", "N/A")},
        {"task": "grounding", "prompt": "Define TKS Noetic ^5", "expected": noetics.get("^5", "N/A")},
        {"task": "grounding", "prompt": "Define TKS Noetic ^8", "expected": noetics.get("^8", "N/A")},
        # Subfoundations (grounding)
        {"task": "grounding", "prompt": "Define TKS Sub-Foundation 1a", "expected": subfoundations.get("1a", "N/A")},
        {"task": "grounding", "prompt": "Define TKS Sub-Foundation 1d", "expected": subfoundations.get("1d", "N/A")},
        {"task": "grounding", "prompt": "Define TKS Sub-Foundation 3b", "expected": subfoundations.get("3b", "N/A")},
        # Equation checks (optional, keep small and canonical-friendly)
        {"task": "equation", "prompt": "Normalize TKS: A1 + B2", "expected": "A1 +T B2"},
        {"task": "equation", "prompt": "Normalize TKS: C3 * D4", "expected": "C3 *T D4"},
    ]

    print("\n" + "=" * 70)
    print("GROUNDING COMPARISON")
    print("=" * 70)

    orig_grounding = {"exact": 0, "partial": 0, "total": 0}
    recon_grounding = {"exact": 0, "partial": 0, "total": 0}
    orig_equation = {"syntax_valid": 0, "structure_match": 0, "id_exact": 0, "full_exact": 0, "total": 0}
    recon_equation = {"syntax_valid": 0, "structure_match": 0, "id_exact": 0, "full_exact": 0, "total": 0}

    def _accum_eq(bucket, result):
        bucket["total"] += 1
        bucket["syntax_valid"] += int(result.syntax_valid)
        bucket["structure_match"] += int(result.structure_match)
        bucket["id_exact"] += int(result.id_exact)
        bucket["full_exact"] += int(result.full_exact)

    for case in test_cases:
        prompt = case["prompt"]
        expected = case["expected"]
        task = case.get("task", "grounding")

        orig_out = generate(original, tokenizer, prompt)
        recon_out = generate(reconciled, tokenizer, prompt)

        if task == "equation":
            orig_eval = compare_tks_expressions(orig_out, expected)
            recon_eval = compare_tks_expressions(recon_out, expected)
            _accum_eq(orig_equation, orig_eval)
            _accum_eq(recon_equation, recon_eval)

            print(f"\n{prompt}")
            print(f"  Expected:   {expected}")
            print(f"  Original:   {orig_out}")
            print(
                "  Original metrics: "
                f"syntax_valid={orig_eval.syntax_valid} "
                f"struct={orig_eval.structure_match} "
                f"id_exact={orig_eval.id_exact} "
                f"full_exact={orig_eval.full_exact}"
            )
            print(f"  Reconciled: {recon_out}")
            print(
                "  Reconciled metrics: "
                f"syntax_valid={recon_eval.syntax_valid} "
                f"struct={recon_eval.structure_match} "
                f"id_exact={recon_eval.id_exact} "
                f"full_exact={recon_eval.full_exact}"
            )
            continue

        norm_expected = normalize_text(expected)
        norm_orig = normalize_text(orig_out)
        norm_recon = normalize_text(recon_out)

        orig_match = "EXACT" if norm_orig == norm_expected else ("PARTIAL" if partial_text_match(orig_out, expected) else "MISS")
        recon_match = "EXACT" if norm_recon == norm_expected else ("PARTIAL" if partial_text_match(recon_out, expected) else "MISS")

        orig_grounding["total"] += 1
        recon_grounding["total"] += 1
        if orig_match == "EXACT":
            orig_grounding["exact"] += 1
        elif orig_match == "PARTIAL":
            orig_grounding["partial"] += 1

        if recon_match == "EXACT":
            recon_grounding["exact"] += 1
        elif recon_match == "PARTIAL":
            recon_grounding["partial"] += 1

        print(f"\n{prompt}")
        print(f"  Expected:   {expected[:60]}...")
        print(f"  Original:   [{orig_match:7}] {orig_out[:60]}...")
        print(f"  Reconciled: [{recon_match:7}] {recon_out[:60]}...")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if orig_grounding["total"]:
        print(f"\nOriginal Model (grounding):")
        print(
            f"  Exact matches:   {orig_grounding['exact']}/{orig_grounding['total']} "
            f"({100*orig_grounding['exact']/orig_grounding['total']:.1f}%)"
        )
        print(
            f"  Partial matches: {orig_grounding['partial']}/{orig_grounding['total']} "
            f"({100*orig_grounding['partial']/orig_grounding['total']:.1f}%)"
        )
        print(
            f"  Total usable:    {orig_grounding['exact'] + orig_grounding['partial']}/{orig_grounding['total']} "
            f"({100*(orig_grounding['exact'] + orig_grounding['partial'])/orig_grounding['total']:.1f}%)"
        )

    if recon_grounding["total"]:
        print(f"\nReconciled Model (grounding):")
        print(
            f"  Exact matches:   {recon_grounding['exact']}/{recon_grounding['total']} "
            f"({100*recon_grounding['exact']/recon_grounding['total']:.1f}%)"
        )
        print(
            f"  Partial matches: {recon_grounding['partial']}/{recon_grounding['total']} "
            f"({100*recon_grounding['partial']/recon_grounding['total']:.1f}%)"
        )
        print(
            f"  Total usable:    {recon_grounding['exact'] + recon_grounding['partial']}/{recon_grounding['total']} "
            f"({100*(recon_grounding['exact'] + recon_grounding['partial'])/recon_grounding['total']:.1f}%)"
        )

    if orig_equation["total"]:
        print(f"\nOriginal Model (equations):")
        print(
            f"  Syntax valid:    {orig_equation['syntax_valid']}/{orig_equation['total']} "
            f"({100*orig_equation['syntax_valid']/orig_equation['total']:.1f}%)"
        )
        print(
            f"  Structure match: {orig_equation['structure_match']}/{orig_equation['total']} "
            f"({100*orig_equation['structure_match']/orig_equation['total']:.1f}%)"
        )
        print(
            f"  ID exact:        {orig_equation['id_exact']}/{orig_equation['total']} "
            f"({100*orig_equation['id_exact']/orig_equation['total']:.1f}%)"
        )
        print(
            f"  Full exact:      {orig_equation['full_exact']}/{orig_equation['total']} "
            f"({100*orig_equation['full_exact']/orig_equation['total']:.1f}%)"
        )

    if recon_equation["total"]:
        print(f"\nReconciled Model (equations):")
        print(
            f"  Syntax valid:    {recon_equation['syntax_valid']}/{recon_equation['total']} "
            f"({100*recon_equation['syntax_valid']/recon_equation['total']:.1f}%)"
        )
        print(
            f"  Structure match: {recon_equation['structure_match']}/{recon_equation['total']} "
            f"({100*recon_equation['structure_match']/recon_equation['total']:.1f}%)"
        )
        print(
            f"  ID exact:        {recon_equation['id_exact']}/{recon_equation['total']} "
            f"({100*recon_equation['id_exact']/recon_equation['total']:.1f}%)"
        )
        print(
            f"  Full exact:      {recon_equation['full_exact']}/{recon_equation['total']} "
            f"({100*recon_equation['full_exact']/recon_equation['total']:.1f}%)"
        )

    if orig_grounding["total"] or recon_grounding["total"]:
        orig_score = orig_grounding["exact"] + 0.5 * orig_grounding["partial"]
        recon_score = recon_grounding["exact"] + 0.5 * recon_grounding["partial"]
        improvement = recon_score - orig_score

        print(f"\nImprovement Score (grounding): {improvement:+.1f} points")
        if improvement > 0:
            print("  => Reconciled model shows improvement")
        elif improvement < 0:
            print("  => Reconciled model shows regression (may need more training)")
        else:
            print("  => No significant change")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
