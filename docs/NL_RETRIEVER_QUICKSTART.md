# NL Retriever Quick Start Guide

## What is it?

The NL Retriever allows the TKS LLM to process natural language inputs through the operator core by automatically finding matching TKS equations. It acts as a fallback when no explicit equation is detected in the input.

## Quick Setup

### 1. Enable NL Retriever

```python
from tks_llm_core_v4 import TKSNoeticLM, TKSNoeticLMConfig

config = TKSNoeticLMConfig(
    vocab_size=512,
    use_operator_core=True,      # Enable operator core
    use_nl_retriever=True,       # Enable NL retriever fallback
    nl_retriever_threshold=0.5,  # Similarity threshold (0.0-1.0)
)

model = TKSNoeticLM(config)
```

### 2. Required Files

Place these files in your project directory:

```
output/
  nl_retriever.pt              # Trained retriever model

data/equation_embeddings/
  equation_corpus.pt           # Equation embeddings [N, 384]
  corpus_meta.pt              # Metadata with triplet info
```

### 3. Run Inference

```python
import torch

# Natural language input (no explicit equation)
tokens = torch.randint(0, 512, (1, 20))

# Forward pass - retriever automatically finds matching equation
output = model(tokens)

# Check if operator core was used
if 'operator_core_output' in output:
    print("Operator core used (equation retrieved)")
else:
    print("Pure NL path (no equation match)")
```

## When to Use

### Use NL Retriever when:

- Input is natural language describing TKS concepts
- You want equation-aware processing without explicit equations
- Input semantically maps to known TKS equations
- You have a pre-built equation corpus

### Don't use when:

- Input contains explicit equations (use EquationDetector instead)
- You want pure language modeling (disable operator core)
- No suitable equation corpus exists
- Performance is critical (adds ~5% overhead)

## Threshold Tuning

### Threshold Guide

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| 0.1-0.3 | Aggressive | Exploration, broad matching |
| 0.5 (default) | Balanced | General use |
| 0.7-0.9 | Conservative | High precision required |
| 0.95+ | Strict | Only exact semantic matches |

### Example

```python
# Conservative: only high-confidence matches
config = TKSNoeticLMConfig(
    use_nl_retriever=True,
    nl_retriever_threshold=0.8,
)

# Aggressive: try to match most inputs
config = TKSNoeticLMConfig(
    use_nl_retriever=True,
    nl_retriever_threshold=0.2,
)
```

## Integration with EquationDetector

### Typical Pipeline

```python
from equation_detector import EquationDetector

detector = EquationDetector()
text = "Consider spiritual growth and mental clarity"

# 1. Try explicit detection first
triplets = detector.parse_single(text)

if triplets:
    # Explicit equation found
    triplet = triplets[0]
    equation_triplet = (
        torch.tensor([triplet.left_idx]),
        torch.tensor([triplet.operator_idx]),
        torch.tensor([triplet.right_idx]),
    )
    output = model(tokens, equation_triplet=equation_triplet)
else:
    # No explicit equation - retriever will be used as fallback
    output = model(tokens)
```

## File Format Specifications

### nl_retriever.pt

Checkpoint dictionary:
```python
{
    'model_state_dict': {...},  # Retriever model weights
    'config': {
        'nl_dim': 384,
        'eq_dim': 384,
        'hidden_dim': 256,
        'use_hidden': True,
        'dropout': 0.1
    }
}
```

### equation_corpus.pt

Tensor of shape `[num_equations, 384]`:
```python
torch.Tensor([
    [0.123, -0.456, ...],  # Equation 0 embedding
    [0.789, 0.234, ...],   # Equation 1 embedding
    ...
])
```

### corpus_meta.pt

List of metadata dictionaries:
```python
[
    {
        'left_idx': 0,      # A1
        'operator_idx': 0,  # +
        'right_idx': 13,    # B4
    },
    {
        'left_idx': 22,     # C3
        'operator_idx': 2,  # ×
        'right_idx': 36,    # D7
    },
    ...
]
```

## Debugging

### Check if Retriever is Loaded

```python
print(f"Retriever loaded: {model.nl_retriever is not None}")
print(f"Corpus loaded: {model.nl_retriever_corpus is not None}")
print(f"Metadata loaded: {model.nl_retriever_meta is not None}")
```

### Monitor Retrieval

```python
# Manual retrieval test
if model.nl_retriever is not None:
    # Get hidden states (after blocks, before operator core)
    with torch.no_grad():
        emb = model.embedding(tokens)
        x = emb["noetic"] + model.position(tokens.shape[1], tokens.device)

        for layer in model.blocks:
            x, _ = layer(x, processor=model.processor)

        x = model.final_norm(x)

        # Try retrieval
        triplet, similarity = model._retrieve_equation_for_nl(x)

        if triplet:
            print(f"Retrieved equation (similarity={similarity:.4f}):")
            print(f"  Left: {triplet[0].item()}")
            print(f"  Operator: {triplet[1].item()}")
            print(f"  Right: {triplet[2].item()}")
        else:
            print(f"No match (similarity={similarity:.4f} < threshold={model.config.nl_retriever_threshold})")
```

## Common Issues

### Issue: Retriever not loading

**Symptom**: `model.nl_retriever is None`

**Solutions**:
1. Check file paths are correct
2. Verify files exist
3. Check file permissions
4. Look for exceptions during initialization

### Issue: Operator core never used

**Symptom**: `'operator_core_output' not in output`

**Solutions**:
1. Lower threshold (try 0.1)
2. Check corpus quality (embeddings may be poor)
3. Verify operator core is enabled
4. Check input is compatible with corpus

### Issue: Performance degradation

**Symptom**: Inference too slow

**Solutions**:
1. Reduce corpus size (keep only high-quality equations)
2. Disable retriever for non-equation tasks
3. Use CPU corpus (avoid GPU transfer)
4. Cache retrieved triplets for repeated inputs

## Testing

Run comprehensive tests:

```bash
python scripts/test_nl_retriever_integration.py
```

Expected output:
```
✓ PASS: Explicit Equation Detection
✓ PASS: NL Retriever Fallback
✓ PASS: Pure NL (No Match)
✓ PASS: Threshold Behavior
✓ PASS: EquationDetector Integration

Total: 5/5 tests passed
🎉 All tests passed!
```

## Performance Tips

1. **Corpus Size**: Keep corpus under 10K equations for fast retrieval
2. **Device**: Keep corpus on CPU unless batch retrieval needed
3. **Caching**: Cache common retrievals in production
4. **Batching**: Process multiple inputs together for efficiency
5. **Threshold**: Start high (0.8) and lower if needed

## Summary

The NL Retriever integration is:
- **Easy to enable**: 2 config flags
- **Self-contained**: Gracefully degrades if files missing
- **Efficient**: Minimal overhead (~5%)
- **Flexible**: Tunable threshold for different use cases

For more details, see: `docs/NL_RETRIEVER_INTEGRATION.md`
