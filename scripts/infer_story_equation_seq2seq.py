#!/usr/bin/env python3
"""
Minimal inference CLI for the seq2seq Story↔Equation models.

Supports both checkpoints produced by:
  - scripts/train_story_equation_lite.py (LSTM seq2seq)
  - scripts/train_story_equation.py (Transformer seq2seq)

Examples:
  # Story -> equation
  python scripts/infer_story_equation_seq2seq.py \\
    --checkpoint output/teacher_model_story_lite_v1/final_model.pt \\
    --direction story_to_eq \\
    --text "The following transformation occurs: the repulsion of the physical world subtracts Emotional-Negative energy with sense 6 anchored in Companionship."

  # Equation -> story
  python scripts/infer_story_equation_seq2seq.py \\
    --checkpoint output/teacher_model_story_lite_v1/final_model.pt \\
    --direction eq_to_story \\
    --text "D3 - C3^6_d4"
"""

import argparse
import re
import sys
from pathlib import Path

import torch

# Import shared tokenizer/model/generation helpers from the evaluator module
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
import eval_story_equation as ese  # noqa: E402


def build_story_to_eq_prompt(story: str) -> str:
    return (
        "Given this TKS narrative:\n\n"
        f"{story}\n\n"
        "Translate this into a TKS equation using canonical notation (worlds A,B,C,D; noetics 1-10)."
    )


_ELEMENT_RE = re.compile(r'[ABCD](?:10|[1-9])(?:\^[1-9])?(?:_d[1-7])?')


def build_eq_to_story_prompt(equation: str) -> str:
    elements = _ELEMENT_RE.findall(equation)
    elements_text = ", ".join(elements) if elements else equation
    return (
        f"Given the TKS equation: {equation}\n\n"
        f"Elements: {elements_text}\n\n"
        "Generate a natural language narrative describing this TKS working."
    )


def load_seq2seq_model(checkpoint_path: Path, tokenizer: ese.Seq2SeqTokenizer, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        config = checkpoint.get('config', {}) or {}
    else:
        state_dict = checkpoint
        config = {}

    model_type = config.get('model_type')
    state_keys = list(state_dict.keys()) if hasattr(state_dict, 'keys') else []

    if not model_type:
        if any(k.startswith('encoder_embedding.') for k in state_keys):
            model_type = 'transformer_seq2seq'
        elif any(k.startswith('encoder.') for k in state_keys) and any(k.startswith('decoder.') for k in state_keys):
            model_type = 'lstm_seq2seq'

    if model_type == 'transformer_seq2seq':
        model = ese.Seq2SeqTransformer(
            vocab_size=tokenizer.actual_vocab_size,
            hidden_dim=int(config.get('hidden_dim', 256)),
            num_layers=int(config.get('num_layers', 4)),
            nhead=int(config.get('nhead', 8)),
            max_length=int(config.get('max_length', tokenizer.max_length)),
        )
    elif model_type == 'lstm_seq2seq':
        model = ese.SimpleLSTMSeq2Seq(
            vocab_size=tokenizer.actual_vocab_size,
            hidden_dim=int(config.get('hidden_dim', 128)),
        )
    else:
        raise ValueError(
            "Unsupported checkpoint for this inference script. "
            f"Expected seq2seq LSTM/Transformer, got model_type={model_type!r}."
        )

    model.to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model_type, model


def main() -> int:
    parser = argparse.ArgumentParser(description="Seq2Seq Story↔Equation inference")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to final_model.pt')
    parser.add_argument('--tokenizer', type=str, default=None, help='Optional tokenizer.json path')
    parser.add_argument('--tokenizer-mode', choices=['char', 'mixed'], default=None, help='Override tokenizer mode (char|mixed)')
    parser.add_argument('--direction', choices=['story_to_eq', 'eq_to_story'], required=True)
    parser.add_argument('--text', type=str, required=True, help='Story or equation (depending on direction)')
    parser.add_argument('--raw-prompt', action='store_true', help='Treat --text as the full prompt')
    parser.add_argument('--max-new-tokens', type=int, default=128, help='Max tokens to generate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)

    tokenizer_path = Path(args.tokenizer) if args.tokenizer else (checkpoint_path.parent / "tokenizer.json")
    if tokenizer_path.exists():
        tokenizer = ese.Seq2SeqTokenizer.from_json(tokenizer_path)
    else:
        tokenizer = ese.Seq2SeqTokenizer(max_length=256)

    if args.tokenizer_mode:
        tokenizer.mode = args.tokenizer_mode.lower()
        tokenizer._rebuild_multi_token_index()

    model_type, model = load_seq2seq_model(checkpoint_path, tokenizer, device)

    if args.raw_prompt:
        prompt = args.text
    elif args.direction == 'story_to_eq':
        prompt = build_story_to_eq_prompt(args.text)
    else:
        equation = args.text
        if getattr(tokenizer, "mode", "char") == "mixed":
            equation = re.sub(r"\s+", "", equation.strip())
        prompt = build_eq_to_story_prompt(equation)

    output = ese.generate_text(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens, device=device)

    print(f"Model: {model_type}")
    print(f"Direction: {args.direction}")
    print("-----")
    print(output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
