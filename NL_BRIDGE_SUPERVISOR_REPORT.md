# TKS NL Bridge - Supervisor Final Report

**Date**: 2025-12-23
**Supervisor**: tks-supervisor
**Status**: SCAFFOLDING COMPLETE - READY FOR AGENT EXECUTION

---

## Executive Summary

The NL Bridge infrastructure for TKS-LLM v4 has been fully specified and scaffolded. Three parallel agent tasks are now ready for execution:

| Agent | Task | Status | Output |
|-------|------|--------|--------|
| **A** | NL Dataset | COMPLETE | 40,863 samples in `data/nl_bridge/` |
| **B** | Retriever | SCAFFOLD READY | `scripts/nl_bridge/nl_retriever.py` |
| **C** | v4 Integration | SCAFFOLD READY | `scripts/nl_bridge/v4_integration_patch.py` |

---

## Agent A: NL Dataset (COMPLETE)

### Deliverables
- `/mnt/c/Users/wakil/Downloads/Everthing-Tootra-TKS/data/nl_bridge/nl_train.jsonl` (32,702 samples)
- `/mnt/c/Users/wakil/Downloads/Everthing-Tootra-TKS/data/nl_bridge/nl_val.jsonl` (4,081 samples)
- `/mnt/c/Users/wakil/Downloads/Everthing-Tootra-TKS/data/nl_bridge/nl_test.jsonl` (4,080 samples)
- `/mnt/c/Users/wakil/Downloads/Everthing-Tootra-TKS/data/nl_bridge/index_mapping.json`

### Sample Format
```json
{
  "nl_text": "Spiritual Mind combined with Mental Positive elevates consciousness...",
  "left_idx": 0,
  "right_idx": 11,
  "operator_idx": 0,
  "equation_idx": 1,
  "source_weight": 1.0
}
```

### Statistics
- Total samples: 40,863 (6.4 variants per equation)
- Source: 96.8% original, 3.2% generated
- Operators: + (22.5%), - (27.4%), x (25.2%), / (24.8%)
- Text length: 32-2493 chars (mean 221)

### Verified Contracts
- Index mapping matches `equation_detector.py` (A1=0 through D10=39)
- Operator indices: {0:+, 1:-, 2:x, 3:/}
- `equation_idx` aligns with `equation_corpus.pt` rows

---

## Agent B: Linear NL Retriever (SCAFFOLD READY)

### Implementation File
`/mnt/c/Users/wakil/Downloads/Everthing-Tootra-TKS/scripts/nl_bridge/nl_retriever.py`

### Key Class: LinearNLRetriever
```python
class LinearNLRetriever(nn.Module):
    """
    PURE LINEAR - Single projection matrix, no hidden layers.

    Architecture:
        NL embedding (384D) -> Linear (384D) -> cosine sim -> top-k
    """
    def __init__(
        self,
        embedding_dim: int = 384,
        corpus_path: str = "data/equation_embeddings/equation_corpus.pt",
        top_k: int = 5,
        temperature_init: float = 0.07,
    ):
        # Single linear projection, identity-initialized
        self.projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        nn.init.eye_(self.projection.weight)

        # Learnable temperature
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(0.07)))

        # Frozen corpus embeddings as buffers
        self.register_buffer('corpus_embeddings', ...)  # [6400, 384]
```

### Training Command
```bash
python scripts/nl_bridge/nl_retriever.py train --epochs 50 --batch-size 64
```

### Expected Metrics
- Target: Recall@5 > 60%
- Current baseline: ~20% (random)
- Training time: ~30 min on GPU

### Verified Constraints
- PURE LINEAR: No MLP, no attention, no hidden layers
- Uses same MiniLM encoder as corpus embeddings
- Contrastive loss with source weighting (orig=1.0, gen=0.5)
- Learnable temperature for similarity scaling

---

## Agent C: v4 Integration (SCAFFOLD READY)

### Implementation File
`/mnt/c/Users/wakil/Downloads/Everthing-Tootra-TKS/scripts/nl_bridge/v4_integration_patch.py`

### Integration Options

**Option 1: Enhanced Model**
```python
from scripts.nl_bridge.v4_integration_patch import TKSNoeticLMWithNL

config = TKSNoeticLMConfigWithNL(
    use_operator_core=True,
    use_nl_retriever=True,
    nl_retriever_path="models/nl_retriever.pt",
    nl_confidence_threshold=0.5,
)
model = TKSNoeticLMWithNL(config)

out = model(tokens, raw_text=["spiritual mind combined with mental positive"])
print(out['triplet_source'])  # 'detector' or 'retriever' or None
```

**Option 2: Patch Existing Model**
```python
from scripts.nl_bridge.v4_integration_patch import add_nl_retriever

model = TKSNoeticLM(config)
add_nl_retriever(model, retriever_path="models/nl_retriever.pt")
```

### Flow Diagram
```
raw_text: "spiritual mind elevates consciousness"
    |
    v
EquationDetector.parse_batch()
    |
    +--> Found "A1 + B2"? --> Use detector triplet
    |
    +--> Not found? --> NL Retriever
                            |
                            v
                       MiniLM encode
                            |
                            v
                       LinearNLRetriever.forward()
                            |
                            v
                       confidence > 0.5?
                            |
                            +--> Yes: Use retriever triplet
                            |
                            +--> No: No triplet (graceful fallback)
```

### Config Additions
```python
@dataclass
class TKSNoeticLMConfigWithNL(TKSNoeticLMConfig):
    use_nl_retriever: bool = False
    nl_retriever_path: str = "models/nl_retriever.pt"
    nl_corpus_path: str = "data/equation_embeddings/equation_corpus.pt"
    nl_confidence_threshold: float = 0.5
    nl_top_k: int = 5
```

---

## Interface Contracts (VERIFIED)

### Contract 1: Dataset -> Retriever
| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `equation_idx` | int | 0-6399 | Row in corpus_embeddings |
| `left_idx` | int | 0-39 | Element index |
| `right_idx` | int | 0-39 | Element index |
| `operator_idx` | int | 0-3 | Operator index |
| `source_weight` | float | 0.5-1.0 | Loss weight |

### Contract 2: Retriever -> v4 Integration
| Output | Type | Shape | Description |
|--------|------|-------|-------------|
| `left_idx` | LongTensor | [batch] | Element 0-39 |
| `right_idx` | LongTensor | [batch] | Element 0-39 |
| `operator_idx` | LongTensor | [batch] | Operator 0-3 |
| `confidence` | FloatTensor | [batch] | Retrieval confidence |

### Contract 3: v4 Output Extension
```python
out = model(tokens, raw_text=texts)
out['triplet_source']  # 'detector', 'retriever', or None
```

---

## Files Created

| File | Purpose |
|------|---------|
| `NL_BRIDGE_SUPERVISOR_SPEC.md` | Full specification document |
| `NL_BRIDGE_SUPERVISOR_REPORT.md` | This report |
| `scripts/nl_bridge/build_nl_dataset.py` | Agent A implementation |
| `scripts/nl_bridge/nl_retriever.py` | Agent B implementation |
| `scripts/nl_bridge/v4_integration_patch.py` | Agent C implementation |
| `data/nl_bridge/nl_train.jsonl` | Training data |
| `data/nl_bridge/nl_val.jsonl` | Validation data |
| `data/nl_bridge/nl_test.jsonl` | Test data |
| `data/nl_bridge/index_mapping.json` | Index verification |

---

## Next Steps

### Immediate (Agent B)
```bash
# Train the retriever
cd /mnt/c/Users/wakil/Downloads/Everthing-Tootra-TKS
python scripts/nl_bridge/nl_retriever.py train --epochs 50

# Output: models/nl_retriever.pt
```

### After Training (Agent C)
```bash
# Test integration
python scripts/nl_bridge/v4_integration_patch.py

# Run full eval
python scripts/nl_bridge/nl_retriever.py eval
```

### Validation Checklist
- [ ] Train retriever to Recall@5 > 60%
- [ ] Verify confidence threshold filters low-quality matches
- [ ] End-to-end test: NL -> retriever -> operator_core -> output
- [ ] Regression test: detector path still works
- [ ] Measure NL accuracy improvement (target: 20.8% -> >50%)

---

## Risk Analysis

| Risk | Mitigation |
|------|------------|
| Retriever underfits | More epochs, lower LR, data augmentation |
| Confidence too high | Lower threshold to 0.3-0.4 |
| Latency regression | Profile MiniLM encode (~10ms/sample) |
| Memory bloat | Corpus embeddings are 10MB, acceptable |

---

## Supervisor Handoff Notes

1. **All scaffolding is complete** - agents can execute in parallel
2. **Agent A is fully done** - dataset created and verified
3. **Agent B needs sentence-transformers** - `pip install sentence-transformers`
4. **Agent C depends on B** - integration test requires trained retriever
5. **Symmetry preservation** - retriever inherits corpus symmetry properties

---

*Report generated by tks-supervisor*
*Integration target: TKS-LLM v4 NL accuracy >50%*
