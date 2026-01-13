"""Test model generation with temperature sampling."""
import torch
import sys
sys.path.insert(0, '.')

from tokenizers import Tokenizer
from tks_llm_core_v7 import TKSGeneralLMv7, TKSGeneralConfigV7

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load tokenizer
    tokenizer = Tokenizer.from_file('tokenizer_v5.json')
    vocab_size = tokenizer.get_vocab_size()

    # Config
    config = TKSGeneralConfigV7(
        vocab_size=vocab_size,
        hidden_dim=256,
        num_layers=6,
        num_scales=4,
        use_dps_gating=True,
        dps_max_depth=8,
        use_memory_bank=True,
        memory_bank_size=1000,
        use_impact_only_nw=True,
    )

    # Load model
    model = TKSGeneralLMv7(config)
    state_dict = torch.load('checkpoints/v7_math_code/v7_math_code_model.pt', map_location=device)
    state_dict = {k: v for k, v in state_dict.items() if 'stack_memory' not in k and 'stack_ptr' not in k}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    bos_id = tokenizer.token_to_id('<BOS>')
    eos_id = tokenizer.token_to_id('<EOS>')

    def generate(prompt, temperature=1.0, top_k=50, max_tokens=60):
        enc = tokenizer.encode(prompt)
        tokens = [bos_id] + enc.ids
        input_tensor = torch.tensor([tokens], device=device)

        generated_ids = []
        current = input_tensor

        with torch.no_grad():
            for _ in range(max_tokens):
                output = model(current)
                logits = output['logits'][0, -1, :] / max(temperature, 0.01)

                # Top-k filtering
                if top_k > 0:
                    top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    threshold = top_values[-1]
                    logits[logits < threshold] = float('-inf')

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                generated_ids.append(next_token.item())
                if next_token.item() == eos_id:
                    break
                current = torch.cat([current, next_token.unsqueeze(0)], dim=1)

        return tokenizer.decode(generated_ids)

    prompt = "Input: Explain the difference between a list and a tuple in Python.\n"

    print('\n' + '='*60)
    print('TEMPERATURE SAMPLING TEST')
    print('='*60)
    print(f'Prompt: {prompt[:50]}...')

    for temp in [0.3, 0.5, 0.7, 1.0, 1.5]:
        print(f'\nTemperature {temp}:')
        output = generate(prompt, temperature=temp, top_k=50)
        print(f'  "{output[:100]}..."')

    print('\n' + '='*60)
    print('TOP-K VARIATION TEST (temp=0.8)')
    print('='*60)

    for k in [5, 20, 50, 100]:
        print(f'\nTop-k={k}:')
        output = generate(prompt, temperature=0.8, top_k=k)
        print(f'  "{output[:100]}..."')

    print('\n' + '='*60)
    print('MULTIPLE SAMPLES (temp=0.9, top_k=40)')
    print('='*60)

    for i in range(5):
        output = generate(prompt, temperature=0.9, top_k=40)
        print(f'\nSample {i+1}: "{output[:80]}..."')

    # Test with a different prompt
    print('\n' + '='*60)
    print('DIFFERENT PROMPT TEST')
    print('='*60)

    prompt2 = "Input: What is 17 * 23?\n"
    print(f'Prompt: {prompt2}')
    for temp in [0.5, 0.8, 1.2]:
        output = generate(prompt2, temperature=temp, top_k=40)
        print(f'  temp={temp}: "{output[:80]}..."')

if __name__ == '__main__':
    main()
