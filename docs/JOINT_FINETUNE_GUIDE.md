# Joint Fine-Tuning Guide: TKS v4 + NL Retriever

Complete workflow for training the TKS-LLM v4 model with NL bridge and operator core.

## Overview

The joint fine-tuning script (`scripts/joint_finetune_v4_nl.py`) trains:
1. **TKS v4 Model** - Main noetic language model
2. **NL Retriever** - Maps natural language to equations
3. **Operator Core** - Handles TKS equation composition (with freeze/unfreeze)

## Training Phases

### Phase 1: Frozen Operator Core (epochs 1-N)
- Operator core weights frozen to preserve pretrained symmetry/anti-symmetry
- Main model learns world classification and RPM gating
- NL retriever learns NL→equation alignment

### Phase 2: Unfrozen (epochs N+1 to end)
- Operator core unfrozen with **lower learning rate** (0.1x default)
- All components fine-tune jointly
- Lower LR on operator core prevents destroying symmetry constraints

## Quick Start

```bash
# 1. Minimal test (built-in data)
python scripts/joint_finetune_v4_nl.py --epochs 5 --freeze-epochs 2

# 2. Full training with corpus
python scripts/joint_finetune_v4_nl.py \
    --train-jsonl data/equation_embeddings/splits/train.jsonl \
    --nl-corpus data/nl_equation_pairs/nl_corpus_train.pt \
    --eq-corpus data/equation_embeddings/equation_corpus.pt \
    --operator-path output/operator_core_pretrained.pt \
    --output-path output/v4_joint_finetuned.pt \
    --epochs 40 --freeze-epochs 10

# 3. With pretrained retriever
python scripts/joint_finetune_v4_nl.py \
    --train-jsonl data/equation_embeddings/splits/train.jsonl \
    --nl-corpus data/nl_equation_pairs/nl_corpus_train.pt \
    --retriever-path output/nl_retriever.pt \
    --epochs 40 --freeze-epochs 10
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 40 | Total training epochs |
| `--freeze-epochs` | 10 | Epochs to freeze operator core |
| `--lr` | 3e-4 | Base learning rate |
| `--operator-lr-mult` | 0.1 | LR multiplier for operator core after unfreeze |
| `--retriever-lr-mult` | 0.5 | LR multiplier for retriever |
| `--gate-init` | 0.2 | Initial operator gate value |
| `--gate-final` | 0.1 | Final operator gate value (decays linearly) |

## Loss Weights

| Loss | Weight | Purpose |
|------|--------|---------|
| `--lm-weight` | 1.0 | Next-token prediction |
| `--world-weight` | 0.5 | World classification (A/B/C/D) |
| `--sym-weight` | 0.1 | Operator symmetry constraints |
| `--retriever-weight` | 0.3 | NL→equation alignment |

## Outputs

The script saves:
1. `output/v4_joint_finetuned.pt` - Full checkpoint (model + retriever)
2. `output/v4_joint_finetuned_retriever.pt` - Separate retriever checkpoint
3. `output/v4_joint_finetuned_history.json` - Training metrics per epoch

## Data Requirements

### Required Files
- `data/equation_embeddings/equation_corpus.pt` - Equation embeddings
- `data/nl_equation_pairs/nl_corpus_train.pt` - NL→equation training pairs (optional but recommended)

### Optional Files
- `data/equation_embeddings/splits/train.jsonl` - Equation JSONL with interpretations
- `output/operator_core_pretrained.pt` - Pretrained operator core
- `output/nl_retriever.pt` - Pretrained NL retriever

## Monitoring Progress

The training loop prints:
```
Epoch  1/40 | gate=0.20 [FROZEN] | loss=2.3456 (lm=1.2 world=0.5 sym=0.1 ret=0.5) | acc=75.0% (eq=90% nl=50%)
...
Epoch 11/40 | gate=0.15 [UNFROZEN] | loss=1.8765 ...
```

- `[FROZEN]` / `[UNFROZEN]` shows operator core state
- `gate` shows current operator gate value
- `acc` shows world classification accuracy (equation vs NL breakdown)

## Expected Results

After full training:
- World accuracy: >95% (equations), >60% (NL)
- Operator symmetry violation: <0.01
- NL retrieval Recall@1: >50%

## Integration

After training, load the joint model:

```python
from tks_llm_core_v4 import TKSNoeticLM, TKSNoeticLMConfig
from tks_nl_retriever import TKSNLRetriever

# Load checkpoint
checkpoint = torch.load('output/v4_joint_finetuned.pt')

# Initialize model
config = TKSNoeticLMConfig(
    vocab_size=checkpoint['config']['vocab_size'],
    use_operator_core=True,
    use_nl_retriever=True,
    nl_retriever_path='output/v4_joint_finetuned_retriever.pt',
)
model = TKSNoeticLM(config)
model.load_state_dict(checkpoint['model_state_dict'])
```

## Troubleshooting

### "No pretrained operator core" warning
Run `scripts/pretrain_operator_core.py` first, or ignore if training from scratch.

### Low NL accuracy
- Ensure NL corpus is loaded: check for "Loaded NL corpus: N embeddings" in output
- Increase `--retriever-weight`
- Use more diverse NL training samples

### Symmetry violations increasing after unfreeze
- Reduce `--operator-lr-mult` (try 0.05)
- Increase `--freeze-epochs`
- Increase `--sym-weight`
