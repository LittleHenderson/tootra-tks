import sys
import os
import argparse
from collections import Counter
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tks_llm_core_v7 import TKSGeneralLMv7, TKSGeneralConfigV7
from tks_features.rtta_canonical import RTTACanonical

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# HYPERPARAMETERS
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.4
TOP_K = 40
TOP_P = 0.9
REPEAT_WINDOW = 64
REPEAT_PENALTY = 1.2
PROMPT_PENALTY = 0.5
NO_REPEAT_NGRAM = 3
MAX_SAME_TOKEN_RUN = 5
MIN_GENERATED_TOKENS = 15
MIN_ALNUM_CHARS = 10  # Relaxed for code heavy output
REPETITION_RATIO_LIMIT = 0.5
UNIQUE_RATIO_LIMIT = 0.2
MIN_WORDS = 6
MIN_KEYWORDS = 2
DSL_KEYWORDS = {"blueprint", "stage", "sequence", "attractor", "simulate"}
UNHELPFUL_RTTA_ANSWERS = {
    "unable to determine",
    "cannot determine (unrecognized problem type)",
}
PLAIN_DUP_RATIO_LIMIT = 0.18
PLAIN_TOP_WORD_RATIO_LIMIT = 0.25
PLAIN_MIN_WORDS_FOR_QUALITY = 8

def sample_next_token(logits: torch.Tensor) -> int:
    if TEMPERATURE <= 0:
        return logits.argmax(dim=-1).item()
    
    logits = logits / TEMPERATURE
    
    # Top-K
    if TOP_K > 0:
        indices_to_remove = logits < torch.topk(logits, TOP_K)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')

    # Top-P (Nucleus)
    if TOP_P > 0 and TOP_P < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > TOP_P
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        for b in range(logits.size(0)):
            indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
            logits[b, indices_to_remove] = float('-inf')
            
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()

def banned_tokens_for_prefix(output_ids, n):
    if len(output_ids) < n - 1:
        return set()
    prefix = tuple(output_ids[-(n - 1):])
    banned = set()
    for i in range(len(output_ids) - n + 1):
        if tuple(output_ids[i:i + n - 1]) == prefix:
            banned.add(output_ids[i + n - 1])
    return banned

def fallback_dsl():
    return (
        "blueprint OvertonShift {\n"
        "  attractor current_demographic\n"
        "  sequence Phi_shift = transfinite(3)\n"
        "  stage 1: extract(attractor)\n"
        "  stage 2: define(Phi_shift)\n"
        "  stage 3: simulate(OvertonShift)\n"
        "}\n"
    )

def fallback_plain():
    return (
        "I'm not confident in that one yet. "
        "Try rephrasing it or ask a structured reasoning question, and I'll take another pass."
    )

def is_dsl_like(text: str) -> bool:
    lowered = text.lower()
    hits = {kw for kw in DSL_KEYWORDS if kw in lowered}
    words = [w for w in lowered.split() if any(ch.isalnum() for ch in w)]
    if len(words) < MIN_WORDS:
        return False
    if len(hits) < MIN_KEYWORDS:
        return False
    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < UNIQUE_RATIO_LIMIT:
        return False
    return True

def normalize_prompt(prompt: str, mode: str) -> str:
    cleaned = prompt.strip()
    if mode == "dsl" and "respond with tks dsl only." not in cleaned.lower():
        cleaned = f"{cleaned}\nRespond with TKS DSL only."
    return cleaned

def build_input_text(prompt: str, mode: str) -> str:
    if mode == "dsl":
        return f"TASK\n{prompt}\nTKS_DSL\n"
    return f"QUESTION\n{prompt}\nANSWER\n"

def is_unhelpful_rtta_answer(answer: str) -> bool:
    if not answer:
        return True
    return answer.strip().lower() in UNHELPFUL_RTTA_ANSWERS

def is_plain_low_quality(text: str) -> bool:
    words = [w for w in text.split() if any(ch.isalnum() for ch in w)]
    if len(words) <= 3:
        return False
    if len(words) < PLAIN_MIN_WORDS_FOR_QUALITY:
        return False
    lowered = [w.lower() for w in words]
    counts = Counter(lowered)
    top_ratio = max(counts.values()) / len(lowered)
    adjacent_dupes = sum(
        1 for i in range(1, len(lowered)) if lowered[i] == lowered[i - 1]
    )
    dup_ratio = adjacent_dupes / max(1, len(lowered) - 1)
    return (
        top_ratio > PLAIN_TOP_WORD_RATIO_LIMIT
        or dup_ratio > PLAIN_DUP_RATIO_LIMIT
    )

def load_model_and_tokenizer(checkpoint_path=None):
    tokenizer = Tokenizer.from_file("tokenizer_v5.json")
    if not checkpoint_path:
        checkpoint_path = "checkpoints/v7_multidomain/v7_multidomain_model.pt"

    config = TKSGeneralConfigV7(
        vocab_size=tokenizer.get_vocab_size(),
        hidden_dim=256,
        num_layers=6,
        num_scales=4,
        use_dps_gating=True,
        dps_max_depth=8
    )

    model = TKSGeneralLMv7(config).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    filtered_state = {k: v for k, v in checkpoint.items() if "stack_memory" not in k and "stack_ptr" not in k}
    model.load_state_dict(filtered_state, strict=False)
    model.eval()
    return model, tokenizer

def run_advanced_operation(
    prompt: str,
    *,
    require_model_output: bool = False,
    mode: str = "dsl",
    model: TKSGeneralLMv7 = None,
    tokenizer: Tokenizer = None,
    use_rtta: bool = False,
    rtta: RTTACanonical = None
):
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer()

    prompt = normalize_prompt(prompt, mode)
    print(f"--- INITIATING ADVANCED OPERATION ---\n{prompt}\n")

    rtta_result = None
    rtta_answer = None
    rtta_ok = False
    if mode == "plain" and use_rtta and rtta is not None:
        rtta_result = rtta.reason(prompt)
        if isinstance(rtta_result, dict):
            rtta_answer = rtta_result.get("answer")
            problem_type = rtta_result.get("evidence", {}).get("derived", {}).get("problem_type")
            rtta_ok = (
                rtta_answer
                and not is_unhelpful_rtta_answer(rtta_answer)
                and problem_type
                and problem_type != "unknown"
            )

    if rtta_ok and not require_model_output:
        print("[System Info: Using RTTA-C for recognized reasoning task]")
        print(f"--- REASONING ENGINE OUTPUT ---\n{rtta_answer}")
        return

    input_text = build_input_text(prompt, mode)
    input_ids = torch.tensor(tokenizer.encode(input_text).ids).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output_ids = input_ids[0].tolist()
        prompt_ids = set(output_ids)
        start_len = len(output_ids)
        same_run = 0
        loop_detected = False
        eos_id = tokenizer.token_to_id("<EOS>")
        alnum_chars = 0

        for _ in range(MAX_NEW_TOKENS):
            inp = torch.tensor([output_ids], device=DEVICE)
            logits = model(inp)['logits'][:, -1, :]

            if PROMPT_PENALTY > 0 and prompt_ids:
                logits[0, list(prompt_ids)] -= PROMPT_PENALTY

            if REPEAT_WINDOW > 0 and len(output_ids) > start_len:
                window_start = max(start_len, len(output_ids) - REPEAT_WINDOW)
                recent = output_ids[window_start:]
                for r in set(recent):
                    logits[0, r] -= REPEAT_PENALTY

            if NO_REPEAT_NGRAM > 1:
                banned = banned_tokens_for_prefix(output_ids[start_len:], NO_REPEAT_NGRAM)
                if banned:
                    logits[0, list(banned)] = float("-inf")

            next_token = sample_next_token(logits)
            if next_token == eos_id:
                break

            if output_ids and next_token == output_ids[-1]:
                same_run += 1
            else:
                same_run = 0

            if same_run >= MAX_SAME_TOKEN_RUN:
                loop_detected = True
                break

            output_ids.append(next_token)
            token_text = tokenizer.decode([next_token])
            alnum_chars += sum(1 for c in token_text if c.isalnum())

        generated_ids = output_ids[start_len:]
        response = tokenizer.decode(generated_ids).strip()

        # Quality Checks
        counts = Counter(generated_ids)
        rep_ratio = (max(counts.values()) / len(generated_ids)) if generated_ids else 1.0
        unique_ratio = (len(counts) / len(generated_ids)) if generated_ids else 0.0
        dsl_ok = True if mode != "dsl" else is_dsl_like(response)

        plain_low_quality = False
        if mode == "plain":
            plain_low_quality = is_plain_low_quality(response)

        if not generated_ids:
            loop_detected = True
        elif mode == "dsl":
            if len(generated_ids) < MIN_GENERATED_TOKENS:
                loop_detected = True
            elif rep_ratio > REPETITION_RATIO_LIMIT or unique_ratio < UNIQUE_RATIO_LIMIT:
                loop_detected = True
        else:
            if len(generated_ids) >= MIN_GENERATED_TOKENS and (
                rep_ratio > REPETITION_RATIO_LIMIT or unique_ratio < UNIQUE_RATIO_LIMIT
            ):
                loop_detected = True
            elif plain_low_quality:
                loop_detected = True

        quality_failed = loop_detected
        if mode == "dsl":
            quality_failed = quality_failed or alnum_chars < MIN_ALNUM_CHARS or not dsl_ok
        else:
            quality_failed = quality_failed or plain_low_quality

        if quality_failed:
            if mode == "dsl" and not require_model_output:
                print("[System Info: Low quality output, using TKS fallback]")
                response = fallback_dsl()
            elif mode == "plain" and not require_model_output:
                used_rtta = False
                if use_rtta and rtta is not None:
                    if rtta_answer is None and isinstance(rtta_result, dict):
                        rtta_answer = rtta_result.get("answer")
                    if rtta_answer and not is_unhelpful_rtta_answer(rtta_answer):
                        print("[System Info: Low quality output, using RTTA-C fallback]")
                        response = rtta_answer
                        used_rtta = True
                if not used_rtta:
                    print("[System Info: Low quality output, using plain fallback]")
                    response = fallback_plain()
            else:
                print("[Warning: Low quality output]")

        print(f"--- REASONING ENGINE OUTPUT ---\n{response}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=None, help="Override the default prompt")
    parser.add_argument(
        "--mode",
        choices=["dsl", "plain"],
        default="dsl",
        help="Output mode: 'dsl' uses DSL quality gate and fallback"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode (type 'exit' to quit)"
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/v7_multidomain/v7_multidomain_model.pt",
        help="Model checkpoint path"
    )
    parser.add_argument(
        "--require-model-output",
        action="store_true",
        help="Disable fallback DSL when output is low quality"
    )
    parser.add_argument(
        "--use-rtta",
        action="store_true",
        help="Use RTTA-C fallback for plain mode when output is low quality"
    )
    args = parser.parse_args()

    if args.interactive:
        model, tokenizer = load_model_and_tokenizer(args.checkpoint)
        rtta = RTTACanonical() if args.use_rtta else None
        print("Interactive mode. Type 'exit' to quit.")
        while True:
            try:
                user_prompt = input("Q> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_prompt:
                continue
            if user_prompt.lower() in {"exit", "quit", ":q", ":quit"}:
                break
            run_advanced_operation(
                user_prompt,
                require_model_output=args.require_model_output,
                mode=args.mode,
                model=model,
                tokenizer=tokenizer,
                use_rtta=args.use_rtta,
                rtta=rtta
            )
        return

    model, tokenizer = load_model_and_tokenizer(args.checkpoint)
    rtta = RTTACanonical() if args.use_rtta else None
    prompt = args.prompt or (
        "TASK: Engineer a 3-stage Overton Window shift to normalize "
        "'Noetic Architecture' in urban planning.\n"
        "1. Extract current demographic attractor.\n"
        "2. Define the transfinite sequence Phi_shift.\n"
        "3. Generate TKS DSL code for the simulation.\n"
        "Respond with TKS DSL only."
    )

    run_advanced_operation(
        prompt,
        require_model_output=args.require_model_output,
        mode=args.mode,
        model=model,
        tokenizer=tokenizer,
        use_rtta=args.use_rtta,
        rtta=rtta
    )

if __name__ == "__main__":
    main()
