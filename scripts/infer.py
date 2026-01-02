"""
TKS Inference CLI - Deterministic equation/story operations.

Modes:
- equation: Validate a TKS equation
- eq-to-story: Convert TKS equation to narrative story (rule-based decoder)
- story-to-eq: Map story keywords to canonical TKS equation

Options:
- --use-model: Also run constrained model generation (greedy, short, validated)

Usage:
  .\\.venv311\\Scripts\\python.exe scripts/infer.py --mode equation --input "A1 + B2 -> C3"
  .\\.venv311\\Scripts\\python.exe scripts/infer.py --mode equation --input "A1 ->" --use-model
  .\\.venv311\\Scripts\\python.exe scripts/infer.py --mode eq-to-story --input "C3 -> D4"
  .\\.venv311\\Scripts\\python.exe scripts/infer.py --mode story-to-eq --input "Fear transforms into courage"
"""
import argparse
import json
import re
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent.parent))
from teacher.validator import CanonicalValidator
from tks_rules.foundations import FOUNDATIONS
from tks_rules.noetics import NOETICS
from tks_rules.operators import OPERATORS
from tks_rules.worlds import WORLDS
SENSES = {1: 'sight', 2: 'hearing', 3: 'smell', 4: 'taste', 5: 'touch', 6: 'intuition', 7: 'balance', 8: 'proprioception', 9: 'time-sense'}

def load_model(checkpoint_path: str, device: str='cuda'):
    """Load model and tokenizer from checkpoint."""
    from scripts.quick_train import SimpleTokenizer, SimpleTransformer
    tokenizer = SimpleTokenizer(max_length=256)
    model = SimpleTransformer(tokenizer.actual_vocab_size, hidden_dim=128, num_layers=2)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return (model, tokenizer)

def get_default_checkpoint() -> str:
    """Find default checkpoint."""
    config_path = Path('config/model_defaults.json')
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        ckpt = cfg.get('default_checkpoint') or cfg.get('checkpoint')
        if ckpt and Path(ckpt).exists():
            return ckpt
    for path in ['output/teacher_model_long_v4/final_model.pt', 'output/teacher_model_v3/final_model.pt']:
        if Path(path).exists():
            return path
    return None

def model_generate(model, tokenizer, prompt: str, validator, device: str, max_tokens: int=15) -> dict:
    """
    Constrained model generation:
    - Greedy decoding (no sampling)
    - Short max length
    - Stops when output becomes invalid
    """
    model.eval()
    tokens = [2]
    for c in prompt:
        tokens.append(tokenizer.token_to_id.get(c, 1))
    tokens = torch.tensor([tokens], dtype=torch.long, device=device)
    id_to_token = {v: k for k, v in tokenizer.token_to_id.items()}
    generated = []
    last_valid_output = prompt
    stopped_reason = 'max_tokens'
    with torch.no_grad():
        for step in range(max_tokens):
            logits = model(tokens)
            next_token = logits[0, -1, :].argmax().item()
            if next_token in [0, 3]:
                stopped_reason = 'eos'
                break
            generated.append(next_token)
            tokens = torch.cat([tokens, torch.tensor([[next_token]], device=device)], dim=1)
            current_output = prompt + ''.join((id_to_token.get(t, '?') for t in generated))
            if current_output.rstrip().endswith((' ', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0')):
                result = validator.validate(current_output.strip())
                if result.is_valid:
                    last_valid_output = current_output.strip()
                elif not result.is_valid and last_valid_output != prompt:
                    stopped_reason = 'invalid'
                    break
    final_output = prompt + ''.join((id_to_token.get(t, '?') for t in generated))
    final_result = validator.validate(final_output.strip())
    return {'raw_output': final_output.strip(), 'last_valid': last_valid_output, 'is_valid': final_result.is_valid, 'stopped_reason': stopped_reason, 'tokens_generated': len(generated)}

def parse_element(element: str) -> dict:
    """Parse TKS element like A1^2_d3."""
    match = re.match('^([ABCD])(10|[1-9])(\\^[1-9])?(_d[1-7])?$', element)
    if not match:
        return None
    return {'world': match.group(1), 'noetic': int(match.group(2)), 'sense': int(match.group(3)[1]) if match.group(3) else None, 'foundation': match.group(4) if match.group(4) else None}

def equation_to_story(equation: str) -> str:
    """Convert TKS equation to narrative using canonical mappings."""
    tokens = equation.split()
    parts = []
    for token in tokens:
        if token in OPERATORS:
            parts.append(OPERATORS[token]['name'])
        else:
            elem = parse_element(token)
            if elem:
                world_name = WORLDS[elem['world']]['name']
                noetic_name = NOETICS[elem['noetic']]['name']
                desc = f'the {noetic_name} of the {world_name} realm'
                if elem['sense']:
                    desc += f" (through {SENSES.get(elem['sense'], 'sense')})"
                if elem['foundation']:
                    desc += f" grounded in {FOUNDATIONS.get(elem['foundation'], 'foundation')}"
                parts.append(desc)
            else:
                parts.append(f'[{token}]')
    if len(parts) == 1:
        return f'This represents {parts[0]}.'
    return 'In this TKS working, ' + ' '.join(parts) + '.'

def story_to_equation(story: str) -> dict:
    """Extract TKS equation from story using keyword matching."""
    story_lower = story.lower()
    detected_worlds = []
    for code, info in WORLDS.items():
        if any((kw in story_lower for kw in info['keywords'])):
            detected_worlds.append(code)
    if not detected_worlds:
        detected_worlds = ['B', 'C']
    detected_noetics = []
    for num, info in NOETICS.items():
        if any((kw in story_lower for kw in info['keywords'])):
            detected_noetics.append(num)
    if not detected_noetics:
        detected_noetics = [1]
    detected_op = '->'
    for op, info in OPERATORS.items():
        if any((kw in story_lower for kw in info['keywords'])):
            detected_op = op
            break
    if len(detected_worlds) >= 2:
        eq = f'{detected_worlds[0]}{detected_noetics[0]} {detected_op} {detected_worlds[1]}{(detected_noetics[-1] if len(detected_noetics) > 1 else detected_noetics[0])}'
    else:
        eq = f'{detected_worlds[0]}{detected_noetics[0]}'
        if len(detected_noetics) > 1:
            eq += f' {detected_op} {detected_worlds[0]}{detected_noetics[1]}'
    return {'equation': eq, 'detected_worlds': detected_worlds, 'detected_noetics': detected_noetics, 'detected_operator': detected_op}

def main():
    parser = argparse.ArgumentParser(description='TKS Inference CLI - Deterministic equation/story operations', formatter_class=argparse.RawDescriptionHelpFormatter, epilog='\nExamples:\n  Validate equation:\n    python scripts/infer.py --mode equation --input "A1 + B2 -> C3"\n\n  Validate with model completion:\n    python scripts/infer.py --mode equation --input "A1 ->" --use-model\n\n  Equation to story:\n    python scripts/infer.py --mode eq-to-story --input "C3 -> D4"\n\n  Story to equation:\n    python scripts/infer.py --mode story-to-eq --input "Fear transforms into courage"\n')
    parser.add_argument('--mode', choices=['equation', 'eq-to-story', 'story-to-eq'], required=True, help='Operation mode')
    parser.add_argument('--input', required=True, help='Input equation or story')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    parser.add_argument('--use-model', action='store_true', help='Also run constrained model generation (greedy, short, validated)')
    parser.add_argument('--checkpoint', help='Model checkpoint (default: long_v4)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-tokens', type=int, default=15, help='Max tokens for model generation')
    args = parser.parse_args()
    validator = CanonicalValidator(strict_mode=True)
    model, tokenizer = (None, None)
    if args.use_model:
        ckpt = args.checkpoint or get_default_checkpoint()
        if not ckpt:
            sys.exit('No checkpoint found. Provide --checkpoint or ensure long_v4 exists.')
        print(f'Loading model: {ckpt}')
        print(f'Device: {args.device}')
        model, tokenizer = load_model(ckpt, args.device)
        print('Model loaded.\n')
    if args.mode == 'equation':
        result = validator.validate(args.input)
        print(f'Equation: {args.input}')
        print(f'Valid: {result.is_valid}')
        print(f'Canon Score: {result.canon_score}')
        if result.issues:
            print(f"Issues: {', '.join(result.issues)}")
        if result.is_valid and args.verbose:
            story = equation_to_story(args.input)
            print(f'Meaning: {story}')
        if args.use_model and model:
            print(f'\n--- Model Output (greedy, max {args.max_tokens} tokens) ---')
            gen = model_generate(model, tokenizer, args.input, validator, args.device, args.max_tokens)
            print(f"Raw: {gen['raw_output']}")
            print(f"Last Valid: {gen['last_valid']}")
            print(f"Stopped: {gen['stopped_reason']} ({gen['tokens_generated']} tokens)")
    elif args.mode == 'eq-to-story':
        result = validator.validate(args.input)
        story = equation_to_story(args.input)
        print(f'Equation: {args.input}')
        print(f'Valid: {result.is_valid}')
        print(f'Story: {story}')
    elif args.mode == 'story-to-eq':
        result = story_to_equation(args.input)
        validation = validator.validate(result['equation'])
        print(f'Story: {args.input}')
        print(f"Equation (rule-based): {result['equation']}")
        print(f'Valid: {validation.is_valid}')
        if args.verbose:
            print(f"Detected worlds: {result['detected_worlds']}")
            print(f"Detected noetics: {result['detected_noetics']}")
            print(f"Detected operator: {result['detected_operator']}")
        if args.use_model and model:
            print(f'\n--- Model Output (greedy, max {args.max_tokens} tokens) ---')
            gen = model_generate(model, tokenizer, result['equation'] + ' ', validator, args.device, args.max_tokens)
            print(f"Prompt: {result['equation']}")
            print(f"Model expansion: {gen['raw_output']}")
            print(f"Last Valid: {gen['last_valid']}")
            print(f"Stopped: {gen['stopped_reason']}")
if __name__ == '__main__':
    main()