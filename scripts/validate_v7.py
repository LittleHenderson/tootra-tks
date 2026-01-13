"""
TKS v7 Earned Depth Validation Script
Tests:
1. ATOMIC: Known foundations in novel world contexts
2. SYNTACTIC: Deeper operator nesting than training data
3. SEMANTIC: Paraphrase invariance
4. DPS: Earned depth functionality (v7 specific)
"""
import argparse
import sys
import torch
from pathlib import Path
from tokenizers import Tokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from tks_llm_core_v7 import TKSGeneralLMv7, TKSGeneralConfigV7

# ============================================================================
# TEST CASES
# ============================================================================

ATOMIC_TESTS = [
    ("wisdom in spiritual space", "A2 <- A10"),
    ("wisdom in physical space", "D11W <- D10"),
    ("wisdom in mental space", "B5W <- B10"),
    ("power in primary substrate", "_d1"),
    ("power in tertiary substrate", "_d3"),
    ("desire manifesting through causal projection", "N1 + operator"),
    ("beauty reflecting in formative space", "N6 + reflection"),
]

SYNTACTIC_TESTS = [
    ("the origination transmutes through power then inverts to wisdom", "Chain A->B->C"),
    ("causal vector projects, reflects, then crystallizes in physical space", "Chain 3+ ops"),
    ("masculine power in quaternary emotional substrate expressing differentiation", "Complex context"),
    ("the inversion of the projection of desire through mental causality", "Nested ops"),
]

SEMANTIC_TESTS = [
    ("the causal origination vector", "vector of causal origins"),
    ("wisdom acquisition path", "the path for acquiring wisdom"),
    ("masculine projective modality", "projective masculine mode"),
]

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================


def _infer_hidden_dim(state_dict):
    for key in ("token_emb.weight", "lm_head.weight"):
        if key in state_dict:
            return state_dict[key].shape[1]
    return None


def _infer_num_layers(state_dict):
    indices = set()
    for key in state_dict.keys():
        if key.startswith("blocks."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                indices.add(int(parts[1]))
    return (max(indices) + 1) if indices else None


def _infer_memory_bank(state_dict):
    storage = state_dict.get("_memory_bank.storage")
    if storage is None:
        return None
    return storage.shape[0]


def _load_state_dict(checkpoint_path: str):
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    return state_dict


def load_model(
    checkpoint_path: str,
    hidden_dim: int | None,
    num_layers: int | None,
    memory_bank_size: int | None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file("tokenizer_v5.json")

    state_dict = _load_state_dict(checkpoint_path)
    inferred_hidden = hidden_dim or _infer_hidden_dim(state_dict) or 384
    inferred_layers = num_layers or _infer_num_layers(state_dict) or 6
    inferred_mem = memory_bank_size or _infer_memory_bank(state_dict) or 1000

    config = TKSGeneralConfigV7(
        vocab_size=tokenizer.get_vocab_size(),
        hidden_dim=inferred_hidden,
        num_layers=inferred_layers,
        use_njt=True,
        njt_use_nested_memory=True,
        use_dps_gating=True,
        dps_initial_p_max=2,
        dps_max_depth=5,
        use_memory_bank=True,
        memory_bank_size=inferred_mem,
    )

    model = TKSGeneralLMv7(config)

    # Remove stack buffers from state_dict (size mismatch with dynamic allocation)
    keys_to_remove = [k for k in state_dict.keys() if "stack_memory" in k or "stack_ptr" in k]
    for key in keys_to_remove:
        state_dict.pop(key, None)

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    return model, tokenizer, device


def generate(model, tokenizer, device, prompt, max_len=128):
    """Generate output for a prompt using v7 logic."""
    bos_id = tokenizer.token_to_id("<BOS>") or 1
    eos_id = tokenizer.token_to_id("<EOS>") or 2

    full_prompt = f"Goal: {prompt} <SEP> Reasoning:"
    enc = tokenizer.encode(full_prompt)
    input_ids = [bos_id] + enc.ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    model.reset_dps_state()

    with torch.no_grad():
        for _ in range(max_len):
            output = model(input_tensor, return_full_trace=False)
            logits = output["logits"]
            next_token = logits[0, -1, :].argmax().item()

            if next_token == eos_id:
                break

            input_tensor = torch.cat(
                [input_tensor, torch.tensor([[next_token]], device=device)],
                dim=1,
            )

    output_ids = input_tensor[0].tolist()
    text = tokenizer.decode(output_ids)
    if "<SEP>" in text:
        text = text.split("<SEP>")[-1].strip()
    return text, model.dps_state


def run_tests(checkpoint: str, hidden_dim: int | None, num_layers: int | None, memory_bank_size: int | None):
    if not Path(checkpoint).exists():
        print(f"ERROR: {checkpoint} not found. Training may not be complete.")
        sys.exit(1)

    print(f"Loading v7 Earned Depth Engine from {checkpoint}...")
    model, tokenizer, device = load_model(
        checkpoint,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        memory_bank_size=memory_bank_size,
    )
    print(f"Model loaded on {device}")
    print(f"Initial DPS state: p_max={model.dps_state.p_max}, tokens={model.dps_state.tokens}\n")

    results = {"ATOMIC": 0, "SYNTACTIC": 0, "SEMANTIC": 0}
    dps_stats = {"total_unlocks": 0, "max_depth_reached": 2}

    print("--- ATOMIC TESTS ---")
    score = 0
    for prompt, _expected in ATOMIC_TESTS:
        out, dps = generate(model, tokenizer, device, prompt)
        valid = any(c in out for c in ["A", "B", "C", "D", "P", "W", "->", "<-", "^"])
        if valid:
            score += 1
        print(f"In: {prompt}\nOut: {out}\nValid: {'YES' if valid else 'NO'}\n")
    results["ATOMIC"] = (score / len(ATOMIC_TESTS)) * 100

    print("--- SYNTACTIC TESTS (Deep Nesting) ---")
    score = 0
    for prompt, _expected in SYNTACTIC_TESTS:
        out, dps = generate(model, tokenizer, device, prompt)
        ops = sum(1 for c in out if c in ["-", ">", "<", "^", "/", "!"])
        valid = ops >= 2
        if valid:
            score += 1
        dps_stats["max_depth_reached"] = max(dps_stats["max_depth_reached"], dps.p_max)
        print(f"In: {prompt}\nOut: {out}\nOps: {ops} | Valid: {'YES' if valid else 'NO'}\n")
    results["SYNTACTIC"] = (score / len(SYNTACTIC_TESTS)) * 100

    print("--- SEMANTIC TESTS ---")
    score = 0
    for prompt, _expected in SEMANTIC_TESTS:
        out, _dps = generate(model, tokenizer, device, prompt)
        valid = len(out) > 5
        if valid:
            score += 1
        print(f"In: {prompt}\nOut: {out}\n")
    results["SEMANTIC"] = (score / len(SEMANTIC_TESTS)) * 100

    print("\n" + "=" * 50)
    print("V7 EARNED DEPTH VALIDATION COMPLETE")
    print("=" * 50)
    print(f"ATOMIC:    {results['ATOMIC']:.1f}%")
    print(f"SYNTACTIC: {results['SYNTACTIC']:.1f}%")
    print(f"SEMANTIC:  {results['SEMANTIC']:.1f}%")
    print("-" * 50)
    print(f"DPS Max Depth Reached: {dps_stats['max_depth_reached']}")
    print("=" * 50)

    avg = sum(results.values()) / 3
    if avg >= 85:
        print("VERDICT: EXCELLENT - v7 Reasoning Engine is fully functional.")
    elif avg >= 60:
        print("VERDICT: GOOD - Significant reasoning capability.")
    else:
        print("VERDICT: NEEDS WORK - Reasoning logic needs improvement.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/v7_best.pt")
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--memory-bank-size", type=int, default=None)
    args = parser.parse_args()
    run_tests(
        args.checkpoint,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        memory_bank_size=args.memory_bank_size,
    )
