# TKS NL Bridge - Supervisor Specification

## Overview

This document coordinates three parallel agents building the NL (Natural Language) bridge for the TKS-LLM v4 model. The goal is to improve NL accuracy from 20.8% to >50% by adding a fallback retrieval path when the regex-based EquationDetector misses.

---

## Architecture Summary

```
User NL Input
     |
     v
+--------------------+
| EquationDetector   |  (Regex-based, existing)
| - Parses "A1 + B4" |
+--------------------+
     |
     | (If MISS)
     v
+--------------------+
| NL Retriever       |  <-- Agent B builds this
| - Linear 384D->40D |
| - Contrastive      |
+--------------------+
     |
     v
+--------------------+
| equation_triplet   |  (left_idx, right_idx, operator)
+--------------------+
     |
     v
+--------------------+
| TKSNoeticLM v4     |  <-- Agent C integrates here
| operator_core      |
+--------------------+
```

---

## AGENT A: NL Dataset Construction

### Source File
- **Path**: `/mnt/c/Users/wakil/Downloads/Everthing-Tootra-TKS/tks_6400_complete_merged.jsonl`
- **Count**: 6,400 equations
- **Format**: JSONL with interpretation field

### Input Record Structure
```json
{
  "equation": "A1 + B2",
  "left": "A1",
  "right": "B2",
  "operator": "+",
  "operator_name": "Association",
  "left_name": "Spiritual Mind",
  "right_name": "Mental Positive",
  "interpretation": "Spiritual Mind (A1) combined with Mental Positive (B2)...",
  "source": "original"  // or "generated"
}
```

### Required Output Format (for Agent B)
Agent A must produce a training dataset with this schema:

```json
{
  "nl_text": "Spiritual Mind combined with Mental Positive elevates consciousness...",
  "left_idx": 0,        // 0-39 (A1=0, A10=9, B1=10, ..., D10=39)
  "right_idx": 11,      // 0-39
  "operator_idx": 0,    // 0=+, 1=-, 2=x, 3=/
  "equation_idx": 42,   // Index into equation_corpus.pt (0-6399)
  "source_weight": 1.0  // 1.0 for original, 0.5 for generated
}
```

### Index Mapping (CRITICAL)
```python
# Element to index (must match equation_detector.py)
def element_to_index(element: str) -> int:
    world_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    world = element[0].upper()
    noetic = int(element[1:])  # 1-10
    return world_map[world] * 10 + (noetic - 1)

# Examples:
# A1 -> 0, A10 -> 9
# B1 -> 10, B10 -> 19
# C1 -> 20, C10 -> 29
# D1 -> 30, D10 -> 39

# Operator to index
OPERATOR_MAP = {'+': 0, '-': 1, 'x': 2, '*': 2, '/': 3}
```

### Output Files
Agent A should produce:
1. `data/nl_bridge/nl_train.jsonl` - Training set (80%)
2. `data/nl_bridge/nl_val.jsonl` - Validation set (10%)
3. `data/nl_bridge/nl_test.jsonl` - Test set (10%)
4. `data/nl_bridge/index_mapping.json` - Equation idx to corpus idx verification

### Data Augmentation Suggestions
To improve robustness, Agent A may:
- Extract multiple sentence fragments from each interpretation
- Create paraphrases of key phrases
- Include the full interpretation AND shorter excerpts
- Weight original corpus examples higher (source_weight=1.0 vs 0.5)

---

## AGENT B: Linear NL Retriever

### Constraints
- **PURE LINEAR** - No transformers, no attention, no MLPs
- Must map from 384D (MiniLM embedding) to equation triplet
- Contrastive learning against corpus embeddings

### Input Resources

1. **Equation Embeddings**
   - Path: `data/equation_embeddings/equation_corpus.pt`
   - Shape: `[6400, 384]`
   - Source: all-MiniLM-L6-v2 embeddings of interpretations
   - Contents:
     ```python
     {
       'embeddings': torch.Tensor([6400, 384]),
       'source_weights': torch.Tensor([6400]),  # 1.0 or 0.5
       'operators': torch.Tensor([6400]),       # 0-3
       'left_indices': torch.Tensor([6400]),    # 0-39
       'right_indices': torch.Tensor([6400])    # 0-39
     }
     ```

2. **Metadata**
   - Path: `data/equation_embeddings/equation_embeddings_meta.json`
   - Contains source_weights array

### Architecture Specification

```python
class LinearNLRetriever(nn.Module):
    """
    Pure linear NL->equation retriever.

    Pipeline:
      NL text -> MiniLM -> 384D -> Linear -> 384D (aligned)
                                     |
                                     v
                              cosine similarity with corpus
                                     |
                                     v
                              top-k retrieval -> triplet
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        corpus_path: str = "data/equation_embeddings/equation_corpus.pt",
        top_k: int = 5,
    ):
        super().__init__()
        # Single linear projection (pure linear, no bias for cosine)
        self.projection = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # Initialize close to identity
        nn.init.eye_(self.projection.weight)

        # Learnable temperature
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(0.07)))

        # Load corpus embeddings (frozen)
        corpus = torch.load(corpus_path, weights_only=False)
        self.register_buffer('corpus_embeddings', corpus['embeddings'])
        self.register_buffer('corpus_operators', corpus['operators'])
        self.register_buffer('corpus_left', corpus['left_indices'])
        self.register_buffer('corpus_right', corpus['right_indices'])
        self.register_buffer('source_weights', corpus['source_weights'])

        self.top_k = top_k

    @property
    def temperature(self):
        return torch.exp(self.log_temperature)

    def forward(
        self,
        nl_embeddings: torch.Tensor,  # [batch, 384] from MiniLM
    ) -> Dict[str, torch.Tensor]:
        """
        Retrieve equation triplets from NL embeddings.

        Returns:
            {
                'left_idx': [batch] int,
                'right_idx': [batch] int,
                'operator_idx': [batch] int,
                'confidence': [batch] float,
                'top_k_indices': [batch, k] int,
                'top_k_scores': [batch, k] float,
            }
        """
        # Project and normalize
        projected = self.projection(nl_embeddings)
        projected = F.normalize(projected, dim=-1)
        corpus_normed = F.normalize(self.corpus_embeddings, dim=-1)

        # Cosine similarity
        similarity = torch.mm(projected, corpus_normed.T) / self.temperature

        # Top-k retrieval
        top_k_scores, top_k_indices = similarity.topk(self.top_k, dim=-1)

        # Best match
        best_idx = top_k_indices[:, 0]

        return {
            'left_idx': self.corpus_left[best_idx],
            'right_idx': self.corpus_right[best_idx],
            'operator_idx': self.corpus_operators[best_idx],
            'confidence': torch.softmax(top_k_scores, dim=-1)[:, 0],
            'top_k_indices': top_k_indices,
            'top_k_scores': top_k_scores,
        }

    def contrastive_loss(
        self,
        nl_embeddings: torch.Tensor,   # [batch, 384]
        target_indices: torch.Tensor,  # [batch] corpus indices
        source_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """InfoNCE contrastive loss."""
        projected = self.projection(nl_embeddings)
        projected = F.normalize(projected, dim=-1)

        # Get target embeddings
        targets = F.normalize(self.corpus_embeddings[target_indices], dim=-1)

        # Positive similarity
        pos_sim = (projected * targets).sum(dim=-1) / self.temperature

        # Negative: all corpus (simplified in-batch negatives)
        corpus_normed = F.normalize(self.corpus_embeddings, dim=-1)
        all_sim = torch.mm(projected, corpus_normed.T) / self.temperature

        # InfoNCE
        log_softmax = pos_sim - torch.logsumexp(all_sim, dim=-1)

        if source_weights is not None:
            loss = -(log_softmax * source_weights).mean()
        else:
            loss = -log_softmax.mean()

        return loss
```

### Training Protocol
- Batch size: 64-128
- Optimizer: AdamW, lr=1e-3
- Epochs: 50-100
- Early stopping on validation recall@5
- Loss: InfoNCE contrastive with source weighting

### Output Files
Agent B should produce:
1. `models/nl_retriever.pt` - Trained model weights
2. `models/nl_retriever_config.json` - Model configuration

### Metrics to Report
- Recall@1, Recall@5, Recall@10 on test set
- Mean confidence score distribution
- Failure analysis: which equations are hardest to retrieve?

---

## AGENT C: v4 Integration

### Integration Point
File: `tks_llm_core_v4.py`, class `TKSNoeticLM`

### Current Flow
```python
# In forward():
if self.operator_core is not None and equation_triplet is not None:
    left_idx, right_idx, op_idx = equation_triplet
    # ... runs operator core
```

### Required Modification

Add NL retriever as fallback when EquationDetector returns None:

```python
class TKSNoeticLM(nn.Module):
    def __init__(self, config: TKSNoeticLMConfig) -> None:
        # ... existing init ...

        # NL Retriever (optional fallback)
        self.nl_retriever = None
        self.nl_encoder = None
        if config.use_nl_retriever and NL_RETRIEVER_AVAILABLE:
            self.nl_retriever = LinearNLRetriever(
                corpus_path=config.equation_corpus_path,
            )
            # Load pretrained weights
            if os.path.exists(config.nl_retriever_path):
                self.nl_retriever.load_state_dict(
                    torch.load(config.nl_retriever_path, weights_only=True)
                )
            # MiniLM encoder (frozen)
            from sentence_transformers import SentenceTransformer
            self.nl_encoder = SentenceTransformer('all-MiniLM-L6-v2')
            self.nl_encoder.eval()
            for p in self.nl_encoder.parameters():
                p.requires_grad = False

    def forward(
        self,
        tokens: torch.LongTensor,
        # ... existing params ...
        equation_triplet: Optional[Tuple[torch.Tensor, ...]] = None,
        raw_text: Optional[List[str]] = None,  # NEW: for NL retrieval
        use_nl_fallback: bool = True,          # NEW: enable fallback
    ) -> Dict[str, torch.Tensor]:
        # ... existing forward ...

        # Operator Core with NL Fallback
        operator_out = None
        triplet_source = None

        if equation_triplet is not None:
            triplet_source = 'detector'
        elif (use_nl_fallback and
              self.nl_retriever is not None and
              raw_text is not None):
            # NL Retrieval fallback
            with torch.no_grad():
                nl_embeddings = self.nl_encoder.encode(
                    raw_text, convert_to_tensor=True
                )
            retrieval = self.nl_retriever(nl_embeddings)

            # Only use if confidence > threshold
            if retrieval['confidence'].mean() > 0.5:
                equation_triplet = (
                    retrieval['left_idx'],
                    retrieval['right_idx'],
                    retrieval['operator_idx'],
                )
                triplet_source = 'retriever'

        if self.operator_core is not None and equation_triplet is not None:
            # ... existing operator core logic ...
            pass

        out['triplet_source'] = triplet_source  # Track which path was used
        return out
```

### Config Additions
```python
@dataclass
class TKSNoeticLMConfig:
    # ... existing fields ...

    # NL Retriever config
    use_nl_retriever: bool = False
    nl_retriever_path: str = "models/nl_retriever.pt"
    nl_confidence_threshold: float = 0.5
```

### Integration Test Cases
Agent C must verify:
1. Detector path still works unchanged
2. Retriever fallback activates when detector returns None
3. Confidence threshold filters low-quality retrievals
4. `triplet_source` correctly tracks which path was used
5. No regressions on existing benchmarks

---

## Interface Contracts (CRITICAL)

### Contract 1: NL Dataset <-> Retriever
Agent A's output indices MUST match Agent B's corpus indices:
- `equation_idx` in dataset corresponds to row in `equation_corpus.pt`
- `left_idx`, `right_idx` use same 0-39 mapping as `equation_detector.py`
- `operator_idx` uses {0:+, 1:-, 2:x, 3:/}

### Contract 2: Retriever <-> v4 Integration
Agent B's output format MUST match what Agent C expects:
```python
# Retriever returns:
{
    'left_idx': torch.LongTensor([batch]),     # 0-39
    'right_idx': torch.LongTensor([batch]),    # 0-39
    'operator_idx': torch.LongTensor([batch]), # 0-3
    'confidence': torch.FloatTensor([batch]),  # 0-1
}

# v4 expects equation_triplet as:
(left_idx, right_idx, operator_idx)  # All torch.LongTensor
```

### Contract 3: Symmetry Preservation
The retriever MUST preserve operator symmetry/anti-symmetry:
- For `+` and `x` (symmetric): order of left/right should not affect result
- For `-` and `/` (anti-symmetric): order matters and should be preserved

---

## Validation Checklist

### Agent A Deliverables
- [ ] `data/nl_bridge/nl_train.jsonl` exists and has correct schema
- [ ] `data/nl_bridge/nl_val.jsonl` exists
- [ ] `data/nl_bridge/nl_test.jsonl` exists
- [ ] Index mapping verified against equation_corpus.pt
- [ ] Source weights preserved (1.0 for original, 0.5 for generated)

### Agent B Deliverables
- [ ] `models/nl_retriever.pt` trained weights
- [ ] Architecture is PURE LINEAR (verified no hidden layers)
- [ ] Recall@5 > 60% on test set
- [ ] Contrastive loss uses source weighting
- [ ] Temperature is learnable

### Agent C Deliverables
- [ ] `tks_llm_core_v4.py` updated with NL fallback
- [ ] Config extended with `use_nl_retriever` option
- [ ] Fallback only triggers when detector misses
- [ ] Confidence threshold configurable
- [ ] `triplet_source` tracking in output

### Integration Tests
- [ ] End-to-end: NL text -> retriever -> operator core -> output
- [ ] Regression: existing detector path unchanged
- [ ] Performance: retriever adds < 50ms latency
- [ ] Accuracy: NL accuracy > 50% (up from 20.8%)

---

## File Locations Summary

| Purpose | Path |
|---------|------|
| Source equations | `tks_6400_complete_merged.jsonl` |
| Corpus embeddings | `data/equation_embeddings/equation_corpus.pt` |
| Corpus metadata | `data/equation_embeddings/equation_embeddings_meta.json` |
| NL train data | `data/nl_bridge/nl_train.jsonl` (Agent A creates) |
| Retriever model | `models/nl_retriever.pt` (Agent B creates) |
| v4 core | `tks_llm_core_v4.py` (Agent C modifies) |
| Equation detector | `equation_detector.py` (reference) |
| Compositional layer | `tks_compositional_layer.py` (reference) |

---

## Supervisor Notes

### Resolved Decisions
1. Retriever uses same MiniLM (all-MiniLM-L6-v2) as corpus embeddings
2. Pure linear constraint is non-negotiable for latency
3. Confidence threshold starts at 0.5, tunable
4. Source weighting (orig > generated) applies to both dataset and loss

### Open Questions for Human
1. Should retriever use top-1 or weighted average of top-k?
2. Should NL fallback be on by default in inference?
3. Target latency budget for retrieval path?

### Next Steps
1. Agent A: Begin dataset extraction
2. Agent B: Implement LinearNLRetriever class
3. Agent C: Draft integration PR for review

---

*Document version: 1.0*
*Supervisor: tks-supervisor*
*Date: 2025-12-23*
