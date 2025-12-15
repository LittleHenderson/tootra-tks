# TKS Training Integration Plan
## Scenario Inversion & Anti-Attractor Data Augmentation Strategy

**Version:** 1.0
**Date:** 2025-12-14
**Status:** Planning Document - No Implementation Required

---

## Executive Summary

This document outlines a comprehensive strategy for integrating TKS scenario inversion and anti-attractor synthesis into training pipelines for TKS-aware language models. The approach leverages two core augmentation APIs:

1. **`InvertStory`** (from `scenario_inversion.py`) - Multi-axis semantic inversion for controlled scenario transformation
2. **`AntiAttractorInvert`** (from `anti_attractor.py`) - Attractor signature analysis and counter-scenario synthesis

These APIs enable:
- **Automated data augmentation** through semantic transformations
- **Contrastive learning** via anti-attractor counter-examples
- **Synthetic scenario generation** from seed stories
- **Canonical validation** of model outputs against TKS formal semantics

---

## 1. Current API Overview

### 1.1 InvertStory API

**Location:** `C:\Users\wakil\downloads\everthing-tootra-tks\scenario_inversion.py`

**Core Function:**
```python
def InvertStory(
    story: str,
    axes: Set[str],
    mode: str,
    target: Optional[TargetProfile] = None,
    strict: bool = False
) -> Dict[str, Any]:
    """
    Full pipeline: story -> inverted story.

    Returns:
        Dict with expr_original, expr_inverted, story_inverted
    """
```

**Key Features:**
- **Multi-axis control:** `axes` parameter accepts {"N", "E", "W", "F", "S", "A", "P"} for selective inversion
  - N: Noetic (Mind principles: Positive↔Negative, Female↔Male, Cause↔Effect)
  - W: World (Spiritual↔Physical, Mental↔Emotional)
  - F: Foundation (Unity↔Lust, Wisdom↔Material, Life↔Power)
  - P: Polarity (Overall energetic signature)
- **Three inversion modes:**
  - `"soft"`: Standard semantic opposition
  - `"hard"`: Aggressive transformation with intensity scaling
  - `"targeted"`: Profile-guided remapping using `TargetProfile`
- **Canonical encoding/decoding:** Uses `narrative` module for TKS expression parsing
- **Strict validation:** Optional strict mode for rejecting invalid tokens

**Example Usage:**
```python
result = InvertStory(
    story="A spiritual teacher causes positive change",
    axes={"W", "N"},  # Invert World and Noetic
    mode="soft"
)
# result["story_inverted"] might be:
# "A physical student effects negative resistance"
```

### 1.2 AntiAttractorInvert API

**Location:** `C:\Users\wakil\downloads\everthing-tootra-tks\anti_attractor.py`

**Core Function:**
```python
def AntiAttractorInvert(
    expr: TKSExpression,
    return_signature: bool = False
) -> Dict[str, Any]:
    """
    Generate anti-attractor counter-scenario for a TKS expression.

    Returns:
        Dict with expr_inverted and optionally signature
    """
```

**Key Features:**
- **Attractor signature extraction:** Analyzes element frequency, polarity, foundation distribution
- **Signature inversion:** Applies canonical TKS oppositions (World: A↔D, B↔C; Noetic: 2↔3, 5↔6, 8↔9; Foundation: 1↔7, 2↔6, 3↔5)
- **Counter-scenario synthesis:** Generates new TKS expression occupying opposite semantic region
- **Pattern-based augmentation:** Creates maximally contrasting scenarios while maintaining structural coherence

**Example Usage:**
```python
from scenario_inversion import parse_equation

expr = parse_equation("B2 -> D5")  # Mental Positive causes Physical Female
result = AntiAttractorInvert(expr, return_signature=True)
# result["expr_inverted"].elements might be: ["C3", "A6"]
# (Emotional Negative -> Spiritual Male)
```

---

## 2. Data Augmentation Strategy

### 2.1 Scenario Inversion for Controlled Augmentation

**Objective:** Generate semantically meaningful training pairs with precise control over transformation axes.

**Use Cases:**
1. **World-axis augmentation** - Transform spiritual scenarios to physical equivalents
2. **Noetic polarity flips** - Generate positive/negative sentiment pairs
3. **Foundation remapping** - Create scenarios exploring different archetypal alignments
4. **Multi-axis combinations** - Complex transformations for robust model training

**Augmentation Pipeline:**
```
Input Story
    ↓
[EncodeStory] → TKS Expression (original)
    ↓
[ScenarioInvert with axes/mode] → TKS Expression (inverted)
    ↓
[DecodeStory] → Augmented Story
    ↓
Training Pair: (original, inverted, transformation_metadata)
```

**Metadata to Preserve:**
- `axes_applied`: Set of inverted axes
- `mode`: Inversion intensity/strategy
- `expr_original`: Original TKS expression (for canonical validation)
- `expr_inverted`: Inverted TKS expression (for canonical validation)
- `transformation_explanation`: Human-readable diff from `ExplainInversion()`

**Benefits:**
- **Semantic consistency:** Inversions preserve TKS formal semantics
- **Controlled diversity:** Axes parameter enables targeted augmentation
- **Bidirectional learning:** Model learns both story→inverted and inverted→story mappings
- **Canonical grounding:** All augmented pairs have verified TKS expressions

### 2.2 Anti-Attractor for Contrastive Learning

**Objective:** Generate maximally contrasting counter-examples to break attractor patterns and teach semantic boundaries.

**Use Cases:**
1. **Pattern disruption** - Prevent model from overfitting to dominant narrative patterns
2. **Semantic boundary learning** - Teach what constitutes opposite scenarios
3. **Diversity enforcement** - Ensure training data covers full TKS semantic space
4. **Bias mitigation** - Counter-balance scenarios with dominant world/noetic tendencies

**Contrastive Augmentation Pipeline:**
```
Input Story
    ↓
[EncodeStory] → TKS Expression
    ↓
[compute_attractor_signature] → AttractorSignature (original)
    ↓
[invert_signature] → AttractorSignature (anti-attractor)
    ↓
[synthesize_counter_scenario] → TKS Expression (counter)
    ↓
[DecodeStory] → Counter-Scenario Story
    ↓
Contrastive Pair: (original, counter, signature_metadata)
```

**Metadata to Preserve:**
- `original_signature`: Element counts, polarity, dominant world/noetic
- `inverted_signature`: Anti-attractor characteristics
- `attractor_distance`: Quantitative measure of semantic distance (future enhancement)
- `signature_explanation`: Human-readable signature comparison

**Benefits:**
- **Maximum contrast:** Anti-attractors occupy opposite semantic regions
- **Pattern awareness:** Model learns characteristic signatures of scenarios
- **Generalization:** Prevents mode collapse by exploring complementary patterns
- **Therapeutic alignment:** Matches TKS intervention logic (invert problematic attractors)

### 2.3 Synthetic Generation from Seed Stories

**Objective:** Expand limited training data by generating synthetic variations from high-quality seed scenarios.

**Generation Strategy:**

**Step 1: Extract Seed Scenarios**
- Curate collection of canonical TKS scenarios (from manuals, validated examples)
- Ensure seeds cover diverse world/noetic/foundation combinations
- Validate all seeds pass canonical expression parsing

**Step 2: Multi-Axis Exploration**
```python
seed_story = "A wise spiritual teacher causes enlightenment"
axes_combinations = [
    {"W"},           # World-only inversion
    {"N"},           # Noetic-only inversion
    {"F"},           # Foundation-only inversion
    {"W", "N"},      # World + Noetic
    {"W", "F"},      # World + Foundation
    {"N", "F"},      # Noetic + Foundation
    {"W", "N", "F"}, # Full semantic inversion
]

synthetic_variants = []
for axes in axes_combinations:
    result = InvertStory(seed_story, axes=axes, mode="soft")
    synthetic_variants.append(result["story_inverted"])
```

**Step 3: Anti-Attractor Diversification**
```python
for variant in synthetic_variants:
    expr = EncodeStory(variant)
    anti_result = AntiAttractorInvert(expr)
    counter_story = DecodeStory(anti_result["expr_inverted"])
    synthetic_variants.append(counter_story)
```

**Yield Calculation:**
- **Single seed** → 7 inversion variants → 7 anti-attractor counters = **15 synthetic scenarios**
- **100 seeds** → **1,500 synthetic scenarios** covering broad semantic space

**Quality Controls:**
1. Canonical validation (all expressions must parse correctly)
2. Diversity filtering (detect and filter duplicate/near-duplicate signatures)
3. Coherence scoring (future: semantic coherence metrics)

---

## 3. Integration Points

### 3.1 Preprocessing Pipeline Integration

**Scenario:** Offline batch augmentation before training begins.

**Architecture:**
```
Raw Story Corpus
    ↓
[Canonical Filter] (discard unparseable stories)
    ↓
[Inversion Augmenter]
    - Apply axis combinations to each story
    - Generate inverted pairs
    ↓
[Anti-Attractor Augmenter]
    - Compute attractor signatures
    - Generate counter-scenarios
    ↓
[Deduplication & Quality Filter]
    ↓
Augmented Training Corpus (stored as .jsonl)
```

**Pseudo-code:**
```python
# preprocessing/augment_corpus.py

from scenario_inversion import InvertStory, EncodeStory, DecodeStory
from anti_attractor import AntiAttractorInvert
import json

def augment_corpus(input_stories, axes_sets, use_anti_attractor=True):
    augmented_data = []

    for story in input_stories:
        # Original story
        try:
            expr_original = EncodeStory(story, strict=True)
        except ValueError:
            continue  # Skip unparseable stories

        augmented_data.append({
            "story": story,
            "expr": expr_original.raw,
            "augmentation_type": "original"
        })

        # Inversion augmentations
        for axes in axes_sets:
            result = InvertStory(story, axes=axes, mode="soft")
            augmented_data.append({
                "story": result["story_inverted"],
                "expr": result["expr_inverted"].raw,
                "augmentation_type": "inverted",
                "axes": list(axes),
                "parent_story": story
            })

        # Anti-attractor augmentation
        if use_anti_attractor:
            anti_result = AntiAttractorInvert(expr_original, return_signature=True)
            counter_story = DecodeStory(anti_result["expr_inverted"])
            augmented_data.append({
                "story": counter_story,
                "expr": anti_result["expr_inverted"].raw,
                "augmentation_type": "anti_attractor",
                "signature": anti_result["signature"],
                "parent_story": story
            })

    return augmented_data

# Usage
stories = load_corpus("raw_stories.txt")
axes_sets = [{"W"}, {"N"}, {"F"}, {"W", "N"}]
augmented = augment_corpus(stories, axes_sets)
save_jsonl(augmented, "augmented_corpus.jsonl")
```

**Benefits:**
- One-time preprocessing cost
- Easy to version control augmented datasets
- Can manually inspect/filter before training

**Drawbacks:**
- Storage overhead (multiple copies of augmented data)
- Static augmentation (no dynamic variation during training)

### 3.2 On-the-Fly Augmentation (DataLoader Integration)

**Scenario:** Dynamic augmentation during training for infinite variation.

**Architecture:**
```
Training Loop
    ↓
[DataLoader requests batch]
    ↓
[TKSAugmentationDataset]
    - Load original stories
    - Randomly select augmentation strategy
    - Apply inversion/anti-attractor on-the-fly
    - Return augmented batch
    ↓
[Model consumes batch]
```

**Pseudo-code:**
```python
# training/tks_dataset.py

import torch
from torch.utils.data import Dataset
from scenario_inversion import InvertStory, EncodeStory, DecodeStory
from anti_attractor import AntiAttractorInvert
import random

class TKSAugmentationDataset(Dataset):
    def __init__(self, stories, augment_prob=0.5, axes_pool=None):
        """
        Args:
            stories: List of original story strings
            augment_prob: Probability of applying augmentation to each story
            axes_pool: List of axis sets for inversion (e.g., [{"W"}, {"N"}, {"W", "N"}])
        """
        self.stories = stories
        self.augment_prob = augment_prob
        self.axes_pool = axes_pool or [{"W"}, {"N"}, {"F"}, {"W", "N"}]

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        story = self.stories[idx]

        # Random decision: augment or not
        if random.random() > self.augment_prob:
            return {"text": story, "augmented": False}

        # Random decision: inversion vs anti-attractor
        if random.random() < 0.7:  # 70% inversion, 30% anti-attractor
            # Inversion augmentation
            axes = random.choice(self.axes_pool)
            result = InvertStory(story, axes=axes, mode="soft")
            return {
                "text": result["story_inverted"],
                "augmented": True,
                "type": "inverted",
                "axes": list(axes)
            }
        else:
            # Anti-attractor augmentation
            expr = EncodeStory(story)
            anti_result = AntiAttractorInvert(expr)
            counter_story = DecodeStory(anti_result["expr_inverted"])
            return {
                "text": counter_story,
                "augmented": True,
                "type": "anti_attractor"
            }

# Usage in training loop
from torch.utils.data import DataLoader

dataset = TKSAugmentationDataset(stories, augment_prob=0.5)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    # batch["text"] contains mix of original and augmented stories
    train_step(model, batch["text"])
```

**Benefits:**
- Infinite variation (no two epochs see same augmentations)
- Memory efficient (no storage of augmented data)
- Easy to tune augmentation strength (adjust `augment_prob`)

**Drawbacks:**
- Computational overhead during training
- Harder to reproduce exact training data
- May slow down training if augmentation is expensive

### 3.3 Batch Processing Considerations

**Parallel Augmentation:**
```python
from multiprocessing import Pool
from functools import partial

def augment_story_worker(story, axes, mode="soft"):
    """Worker function for parallel augmentation."""
    try:
        result = InvertStory(story, axes=axes, mode=mode)
        return result["story_inverted"]
    except Exception as e:
        return None

def batch_augment(stories, axes, mode="soft", workers=4):
    """Parallel batch augmentation."""
    with Pool(workers) as pool:
        worker_fn = partial(augment_story_worker, axes=axes, mode=mode)
        augmented = pool.map(worker_fn, stories)
    return [s for s in augmented if s is not None]

# Usage
stories_batch = stories[0:1000]
augmented_batch = batch_augment(stories_batch, axes={"W", "N"}, workers=8)
```

**Batched Anti-Attractor Synthesis:**
```python
def batch_anti_attractor(stories, num_elements=3, workers=4):
    """Parallel batch anti-attractor synthesis."""
    def worker(story):
        try:
            expr = EncodeStory(story)
            anti_expr = anti_attractor(expr, num_elements=num_elements)
            return DecodeStory(anti_expr)
        except:
            return None

    with Pool(workers) as pool:
        counters = pool.map(worker, stories)
    return [c for c in counters if c is not None]
```

**Performance Optimization:**
- Pre-compile axis configurations
- Cache parsed expressions when possible
- Use process pools for CPU-bound augmentation
- Consider GPU-accelerated batch encoding/decoding (future)

---

## 4. Metrics to Track

### 4.1 Canonical Validator Pass-Rate

**Definition:** Percentage of model outputs that parse to valid TKS expressions.

**Measurement:**
```python
from scenario_inversion import EncodeStory

def canonical_pass_rate(model_outputs):
    """Compute % of outputs with valid TKS expressions."""
    total = len(model_outputs)
    valid = 0

    for output in model_outputs:
        try:
            expr = EncodeStory(output, strict=True)
            # Check for minimum structural validity
            if expr.elements and expr.ops:
                valid += 1
        except ValueError:
            pass  # Invalid expression

    return valid / total if total > 0 else 0.0

# Track during evaluation
eval_outputs = generate_batch(model, prompts)
pass_rate = canonical_pass_rate(eval_outputs)
wandb.log({"canonical_pass_rate": pass_rate})
```

**Target Metrics:**
- **Baseline model:** 40-60% pass-rate (untrained on TKS)
- **Augmented training:** 70-85% pass-rate
- **Fine-tuned specialist:** 90%+ pass-rate

**Component Breakdown:**
- **World validity:** % outputs with valid world letters {A, B, C, D}
- **Noetic validity:** % outputs with valid noetic indices {1-10}
- **Foundation validity:** % outputs with correctly attached foundations
- **Operator validity:** % outputs using only allowed operators

### 4.2 Loss Impact from Augmented Data

**Definition:** Measure training loss differential when including augmented scenarios.

**Experimental Setup:**
```python
# Baseline: Train on original data only
baseline_losses = train_model(original_stories, epochs=10)

# Augmented: Train on original + inverted + anti-attractor
augmented_stories = original_stories + inverted_stories + anti_attractor_stories
augmented_losses = train_model(augmented_stories, epochs=10)

# Compare convergence
import matplotlib.pyplot as plt
plt.plot(baseline_losses, label="Baseline")
plt.plot(augmented_losses, label="Augmented")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_comparison.png")
```

**Key Metrics:**
- **Final loss reduction:** `(baseline_final - augmented_final) / baseline_final`
- **Convergence speed:** Epochs to reach baseline final loss
- **Generalization gap:** Validation loss - Training loss (should decrease with augmentation)

**Expected Results:**
- Augmented training may show initially higher loss (more diverse data)
- Should converge to lower final loss due to better generalization
- Validation loss should improve significantly (reduced overfitting)

### 4.3 Diversity Metrics for Generated Scenarios

**Definition:** Quantify semantic diversity of augmented scenarios using TKS signature analysis.

**Signature Diversity Score:**
```python
from anti_attractor import compute_attractor_signature
from collections import Counter

def compute_diversity_score(scenarios):
    """Measure diversity via unique attractor signatures."""
    signatures = []

    for scenario in scenarios:
        expr = EncodeStory(scenario)
        sig = compute_attractor_signature(expr)

        # Create hashable signature representation
        sig_key = (
            sig.dominant_world,
            sig.dominant_noetic,
            sig.polarity,
            frozenset(sig.foundation_tags)
        )
        signatures.append(sig_key)

    # Diversity = ratio of unique signatures to total scenarios
    unique_sigs = len(set(signatures))
    total = len(signatures)

    return {
        "diversity_ratio": unique_sigs / total,
        "unique_signatures": unique_sigs,
        "total_scenarios": total,
        "signature_distribution": Counter(signatures)
    }

# Track diversity
baseline_diversity = compute_diversity_score(original_stories)
augmented_diversity = compute_diversity_score(original_stories + augmented_stories)

print(f"Baseline diversity: {baseline_diversity['diversity_ratio']:.2%}")
print(f"Augmented diversity: {augmented_diversity['diversity_ratio']:.2%}")
```

**Distributional Metrics:**
- **World coverage:** % of world letters {A, B, C, D} represented
- **Noetic coverage:** % of noetic indices {1-10} represented
- **Foundation coverage:** % of foundations {1-7} represented
- **Polarity balance:** Ratio of positive/negative/neutral scenarios

**Expected Improvements:**
- **Baseline:** 30-50% diversity ratio (natural data tends to cluster)
- **Inverted augmentation:** 60-75% diversity ratio (systematic axis exploration)
- **Anti-attractor augmentation:** 70-85% diversity ratio (maximal contrast synthesis)

### 4.4 Additional Evaluation Metrics

**Semantic Consistency:**
```python
def semantic_consistency_score(story1, story2, expected_axes):
    """Verify that inverted stories differ only on expected axes."""
    expr1 = EncodeStory(story1)
    expr2 = EncodeStory(story2)

    # Compare elements on non-inverted axes (should be unchanged)
    # This requires axis-aware comparison logic
    # Future implementation: track per-element diffs
    pass
```

**Round-Trip Fidelity:**
```python
def round_trip_fidelity(story, axes):
    """Test: invert(invert(story)) ≈ story for involutive axes."""
    result1 = InvertStory(story, axes=axes, mode="soft")
    result2 = InvertStory(result1["story_inverted"], axes=axes, mode="soft")

    # Compare signatures (should be similar, not necessarily identical due to decoding)
    sig_original = compute_attractor_signature(EncodeStory(story))
    sig_round_trip = compute_attractor_signature(EncodeStory(result2["story_inverted"]))

    # Compute similarity metric
    # Future: semantic distance metric
    pass
```

---

## 5. Future Considerations

### 5.1 Scaling Considerations

**Data Volume:**
- **Small scale (< 10K scenarios):** Use preprocessing pipeline with full axes exploration
- **Medium scale (10K - 100K scenarios):** Hybrid approach - preprocess with selective axes, on-the-fly for rare combinations
- **Large scale (> 100K scenarios):** On-the-fly only, with probabilistic augmentation sampling

**Computational Budget:**
- **Inversion cost:** ~5-20ms per story (encoding + inversion + decoding)
- **Anti-attractor cost:** ~10-30ms per story (signature extraction + synthesis)
- **Batching:** Parallel processing recommended for > 1K stories

**Storage Scaling:**
```
Original corpus: 100K stories @ 200 chars avg = 20MB raw text

Augmentation expansion:
- 4 axis combinations × 100K = 400K inverted stories
- 1 anti-attractor × 100K = 100K counter stories
Total augmented: 600K stories @ 200 chars = 120MB raw text

With metadata (expressions, signatures):
- ~500 bytes per story (JSON with TKS expression)
Total storage: ~300MB

Recommendation: Store as compressed .jsonl.gz for ~50-100MB
```

### 5.2 Caching Inverted Expressions

**Signature Caching Strategy:**
```python
import hashlib
import pickle
from pathlib import Path

class InversionCache:
    def __init__(self, cache_dir="./inversion_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _cache_key(self, story, axes, mode):
        """Generate cache key from story and parameters."""
        content = f"{story}|{sorted(axes)}|{mode}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, story, axes, mode):
        """Retrieve cached inversion result."""
        key = self._cache_key(story, axes, mode)
        cache_file = self.cache_dir / f"{key}.pkl"

        if cache_file.exists():
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        return None

    def set(self, story, axes, mode, result):
        """Cache inversion result."""
        key = self._cache_key(story, axes, mode)
        cache_file = self.cache_dir / f"{key}.pkl"

        with open(cache_file, "wb") as f:
            pickle.dump(result, f)

# Usage
cache = InversionCache()

def cached_invert_story(story, axes, mode):
    cached = cache.get(story, axes, mode)
    if cached:
        return cached

    result = InvertStory(story, axes=axes, mode=mode)
    cache.set(story, axes, mode, result)
    return result
```

**Cache Management:**
- Implement LRU eviction for large corpora
- Periodic cache pruning based on access frequency
- Version cache keys when inversion algorithm changes

**When to Use Caching:**
- Development/experimentation with same seed corpus
- Multi-epoch training with static augmentation
- Evaluation benchmarks (ensure reproducibility)

**When NOT to Use Caching:**
- On-the-fly augmentation with high randomness
- Memory-constrained environments
- When inversion logic is actively being modified

### 5.3 Quality Filtering

**Coherence Filtering:**
```python
def filter_coherent_scenarios(scenarios, min_elements=2, max_elements=10):
    """Filter scenarios with reasonable structural properties."""
    filtered = []

    for scenario in scenarios:
        try:
            expr = EncodeStory(scenario, strict=True)

            # Structural checks
            num_elements = len(expr.elements)
            num_ops = len(expr.ops)

            if not (min_elements <= num_elements <= max_elements):
                continue

            if num_ops != num_elements - 1:  # Ops should connect elements
                continue

            # Semantic checks (future)
            # - Verify causal consistency
            # - Check foundation-element alignment
            # - Detect degenerate patterns (all elements identical)

            filtered.append(scenario)
        except ValueError:
            continue

    return filtered
```

**Diversity Filtering:**
```python
def filter_duplicate_signatures(scenarios, max_duplicates=3):
    """Limit number of scenarios with identical signatures."""
    from collections import defaultdict

    sig_counts = defaultdict(list)

    for scenario in scenarios:
        expr = EncodeStory(scenario)
        sig = compute_attractor_signature(expr)

        sig_key = (sig.dominant_world, sig.dominant_noetic, sig.polarity)
        sig_counts[sig_key].append(scenario)

    # Take at most max_duplicates per signature
    filtered = []
    for sig_key, scenarios_with_sig in sig_counts.items():
        filtered.extend(scenarios_with_sig[:max_duplicates])

    return filtered
```

**Quality Metrics to Filter On:**
1. **Canonical validity:** Must parse to valid TKS expression
2. **Structural sanity:** Elements and operators consistent
3. **Semantic coherence:** No degenerate patterns (future: coherence model)
4. **Diversity:** Limit signature duplicates
5. **Length:** Filter extremely short/long scenarios

**Filtering Pipeline:**
```
Augmented Scenarios (raw)
    ↓
[Canonical Filter] → 90% pass
    ↓
[Structural Filter] → 85% pass
    ↓
[Diversity Filter] → 70% pass (deduplication)
    ↓
[Manual Review Sample] → Spot-check quality
    ↓
Final Training Corpus
```

### 5.4 Advanced Augmentation Strategies

**Targeted Profile Augmentation:**
```python
from inversion.engine import TargetProfile

# Example: Augment all scenarios toward "Physical + Male" pattern
target = TargetProfile(
    enable=True,
    to_world="D",      # Physical world
    to_noetic=6,       # Male principle
    from_world=None,   # Apply to all source worlds
    from_noetic=None
)

augmented = InvertStory(story, axes={"W", "N"}, mode="targeted", target=target)
```

**Cascaded Augmentation:**
```python
def cascaded_augmentation(story, axis_sequence):
    """Apply sequence of inversions for complex transformations."""
    current = story
    history = [{"step": 0, "story": story, "axes": set()}]

    for i, axes in enumerate(axis_sequence):
        result = InvertStory(current, axes=axes, mode="soft")
        current = result["story_inverted"]
        history.append({"step": i+1, "story": current, "axes": axes})

    return {
        "final_story": current,
        "transformation_history": history
    }

# Example: World → Noetic → Foundation cascade
cascaded = cascaded_augmentation(
    story="A wise teacher causes enlightenment",
    axis_sequence=[{"W"}, {"N"}, {"F"}]
)
```

**Foundation-Aware Augmentation:**
```python
def augment_with_foundation_constraint(story, target_foundation):
    """Generate augmentation that emphasizes specific foundation."""
    expr = EncodeStory(story)

    # Add target foundation if not present
    if target_foundation not in [fid for fid, _ in expr.foundations]:
        expr.foundations.append((target_foundation, "A"))  # Default subfoundation

    # Invert non-foundation axes
    inverted = ScenarioInvert(expr, axes={"W", "N"}, mode="soft")
    return DecodeStory(inverted)
```

### 5.5 Integration with Validation Framework

**Canonical Validator Integration:**
```python
# Assumes existence of canonical validator from TKS_LLM_Canonical_Validation_v1.0.md

def validate_augmented_batch(scenarios):
    """Run canonical validation on augmented scenarios."""
    from tks_llm_core import CanonicalValidator  # Hypothetical import

    validator = CanonicalValidator()
    results = []

    for scenario in scenarios:
        validation = validator.validate(scenario)
        results.append({
            "scenario": scenario,
            "valid": validation.is_valid,
            "world_valid": validation.world_check,
            "noetic_valid": validation.noetic_check,
            "foundation_valid": validation.foundation_check,
            "errors": validation.errors
        })

    return results

# Filter to only valid augmentations
validated = validate_augmented_batch(augmented_scenarios)
valid_scenarios = [r["scenario"] for r in validated if r["valid"]]
```

**Feedback Loop:**
```
Generate Augmented Batch
    ↓
[Canonical Validation]
    ↓
High Pass-Rate? → Use for training
    ↓
Low Pass-Rate? → Log errors, adjust augmentation parameters, regenerate
```

---

## 6. Implementation Roadmap

### Phase 1: Proof of Concept (Weeks 1-2)
- Implement basic preprocessing pipeline
- Generate augmented corpus from 100 seed scenarios
- Measure baseline canonical pass-rate
- Validate diversity improvements

**Deliverables:**
- `preprocessing/augment_corpus.py` script
- Sample augmented dataset (1000 scenarios)
- Initial metrics dashboard (pass-rate, diversity)

### Phase 2: Training Integration (Weeks 3-4)
- Build `TKSAugmentationDataset` for on-the-fly augmentation
- Integrate with existing training loop
- Run small-scale training experiment (10K scenarios)
- Compare baseline vs augmented loss curves

**Deliverables:**
- `training/tks_dataset.py` module
- Training experiment results
- Loss comparison plots
- Updated metrics tracking

### Phase 3: Scaling & Optimization (Weeks 5-6)
- Implement batch processing with multiprocessing
- Add inversion caching for repeated scenarios
- Scale to full corpus (100K+ scenarios)
- Optimize augmentation hyperparameters (augment_prob, axes combinations)

**Deliverables:**
- Optimized batch augmentation pipeline
- Cache management utilities
- Full-scale augmented dataset
- Performance benchmarks

### Phase 4: Quality & Validation (Weeks 7-8)
- Implement quality filtering pipeline
- Add canonical validator integration
- Conduct manual review of augmented samples
- Establish quality thresholds and monitoring

**Deliverables:**
- Quality filtering module
- Validation dashboard
- Quality assurance report
- Final training corpus (filtered & validated)

### Phase 5: Production Deployment (Week 9+)
- Integrate augmentation into production training pipeline
- Set up automated metrics tracking (W&B, TensorBoard)
- Monitor model performance on downstream tasks
- Iterate on augmentation strategies based on results

**Deliverables:**
- Production-ready augmentation pipeline
- Automated CI/CD for corpus generation
- Live metrics dashboard
- Documentation & runbooks

---

## 7. Example Integration Patterns

### Pattern 1: Simple Preprocessing Script

```python
#!/usr/bin/env python3
"""
Simple corpus augmentation script for TKS training data.
Usage: python augment_corpus.py --input stories.txt --output augmented.jsonl
"""

import argparse
import json
from pathlib import Path
from scenario_inversion import InvertStory, EncodeStory
from anti_attractor import AntiAttractorInvert, compute_attractor_signature

def main():
    parser = argparse.ArgumentParser(description="Augment TKS story corpus")
    parser.add_argument("--input", required=True, help="Input stories file (one per line)")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--axes", nargs="+", default=["W", "N", "F"], help="Axes to invert")
    parser.add_argument("--use-anti-attractor", action="store_true", help="Generate anti-attractor pairs")
    args = parser.parse_args()

    # Load stories
    stories = Path(args.input).read_text(encoding="utf-8").strip().split("\n")
    print(f"Loaded {len(stories)} stories from {args.input}")

    # Augment
    augmented_data = []
    axes_set = set(args.axes)

    for i, story in enumerate(stories):
        # Original
        try:
            expr_orig = EncodeStory(story, strict=True)
        except ValueError as e:
            print(f"Skipping invalid story {i}: {e}")
            continue

        augmented_data.append({
            "story": story,
            "type": "original",
            "idx": i
        })

        # Inverted
        result = InvertStory(story, axes=axes_set, mode="soft")
        augmented_data.append({
            "story": result["story_inverted"],
            "type": "inverted",
            "axes": list(axes_set),
            "parent_idx": i
        })

        # Anti-attractor
        if args.use_anti_attractor:
            from scenario_inversion import DecodeStory
            anti_result = AntiAttractorInvert(expr_orig, return_signature=True)
            counter_story = DecodeStory(anti_result["expr_inverted"])
            augmented_data.append({
                "story": counter_story,
                "type": "anti_attractor",
                "parent_idx": i
            })

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(stories)} stories...")

    # Save
    with open(args.output, "w", encoding="utf-8") as f:
        for item in augmented_data:
            f.write(json.dumps(item) + "\n")

    print(f"\nAugmentation complete!")
    print(f"Original stories: {len(stories)}")
    print(f"Total augmented: {len(augmented_data)}")
    print(f"Saved to: {args.output}")

if __name__ == "__main__":
    main()
```

### Pattern 2: PyTorch DataLoader with Augmentation

```python
# training/tks_augmented_dataloader.py

import torch
from torch.utils.data import Dataset, DataLoader
from scenario_inversion import InvertStory, EncodeStory
from anti_attractor import anti_attractor
import random

class TKSAugmentedDataset(Dataset):
    """PyTorch Dataset with on-the-fly TKS augmentation."""

    def __init__(self, stories, tokenizer, augment_config=None):
        """
        Args:
            stories: List of story strings
            tokenizer: Tokenizer for model input
            augment_config: Dict with keys:
                - prob: Augmentation probability (0-1)
                - axes_pool: List of axis sets for inversion
                - use_anti_attractor: Boolean
                - anti_attractor_prob: Probability of using anti-attractor vs inversion
        """
        self.stories = stories
        self.tokenizer = tokenizer

        # Default augmentation config
        default_config = {
            "prob": 0.5,
            "axes_pool": [{"W"}, {"N"}, {"W", "N"}, {"F"}],
            "use_anti_attractor": True,
            "anti_attractor_prob": 0.3
        }
        self.config = {**default_config, **(augment_config or {})}

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        story = self.stories[idx]

        # Apply augmentation with probability
        if random.random() < self.config["prob"]:
            story = self._augment(story)

        # Tokenize
        encoding = self.tokenizer(
            story,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0)  # For language modeling
        }

    def _augment(self, story):
        """Apply random augmentation to story."""
        # Anti-attractor augmentation
        if self.config["use_anti_attractor"] and random.random() < self.config["anti_attractor_prob"]:
            try:
                from scenario_inversion import DecodeStory
                expr = EncodeStory(story)
                anti_expr = anti_attractor(expr)
                return DecodeStory(anti_expr)
            except:
                pass  # Fall back to inversion

        # Inversion augmentation
        try:
            axes = random.choice(self.config["axes_pool"])
            result = InvertStory(story, axes=axes, mode="soft")
            return result["story_inverted"]
        except:
            return story  # Return original if augmentation fails

# Usage
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
dataset = TKSAugmentedDataset(
    stories=training_stories,
    tokenizer=tokenizer,
    augment_config={
        "prob": 0.6,
        "use_anti_attractor": True,
        "anti_attractor_prob": 0.25
    }
)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

# Training loop
for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
```

### Pattern 3: Metrics Tracking with Weights & Biases

```python
# training/metrics_tracker.py

import wandb
from scenario_inversion import EncodeStory
from anti_attractor import compute_attractor_signature

class TKSMetricsTracker:
    """Track TKS-specific training metrics."""

    def __init__(self, project_name="tks-training"):
        wandb.init(project=project_name)

    def log_canonical_metrics(self, model_outputs, step):
        """Log canonical validity metrics."""
        total = len(model_outputs)
        valid_world = 0
        valid_noetic = 0
        valid_full = 0

        for output in model_outputs:
            try:
                expr = EncodeStory(output, strict=True)

                # Check components
                if all(e[0] in "ABCD" for e in expr.elements):
                    valid_world += 1

                if all(int(e[1:]) in range(1, 11) for e in expr.elements):
                    valid_noetic += 1

                if expr.elements and expr.ops:
                    valid_full += 1
            except:
                pass

        wandb.log({
            "canonical/world_validity": valid_world / total,
            "canonical/noetic_validity": valid_noetic / total,
            "canonical/full_validity": valid_full / total,
        }, step=step)

    def log_diversity_metrics(self, scenarios, step, prefix="train"):
        """Log diversity metrics."""
        signatures = []
        worlds = set()
        noetics = set()
        foundations = set()

        for scenario in scenarios:
            try:
                expr = EncodeStory(scenario)
                sig = compute_attractor_signature(expr)

                signatures.append((sig.dominant_world, sig.dominant_noetic, sig.polarity))
                worlds.add(sig.dominant_world)
                noetics.add(sig.dominant_noetic)
                foundations.update(sig.foundation_tags)
            except:
                pass

        unique_sigs = len(set(signatures))
        total_sigs = len(signatures)

        wandb.log({
            f"{prefix}/diversity_ratio": unique_sigs / total_sigs if total_sigs > 0 else 0,
            f"{prefix}/world_coverage": len(worlds) / 4,  # 4 worlds total
            f"{prefix}/noetic_coverage": len(noetics) / 10,  # 10 noetics total
            f"{prefix}/foundation_coverage": len(foundations) / 7,  # 7 foundations total
        }, step=step)

    def log_augmentation_stats(self, batch_metadata, step):
        """Log augmentation statistics from batch."""
        total = len(batch_metadata)
        augmented = sum(1 for m in batch_metadata if m.get("augmented", False))
        inverted = sum(1 for m in batch_metadata if m.get("type") == "inverted")
        anti_attractor = sum(1 for m in batch_metadata if m.get("type") == "anti_attractor")

        wandb.log({
            "augmentation/total_rate": augmented / total,
            "augmentation/inversion_rate": inverted / total,
            "augmentation/anti_attractor_rate": anti_attractor / total,
        }, step=step)

# Usage in training loop
tracker = TKSMetricsTracker(project_name="tks-llm-training")

for step, batch in enumerate(dataloader):
    # Training step
    outputs = model(**batch)
    loss = outputs.loss
    wandb.log({"train/loss": loss.item()}, step=step)

    # Periodic evaluation
    if step % 100 == 0:
        eval_outputs = generate_samples(model, num_samples=50)
        tracker.log_canonical_metrics(eval_outputs, step)
        tracker.log_diversity_metrics(eval_outputs, step, prefix="eval")
```

---

## 8. Summary & Recommendations

### Key Takeaways

1. **Two Complementary Augmentation APIs:**
   - **InvertStory:** Controlled multi-axis transformations with precise semantic control
   - **AntiAttractorInvert:** Maximal contrast synthesis via signature inversion

2. **Flexible Integration Options:**
   - **Preprocessing:** Offline batch augmentation for reproducibility and inspection
   - **On-the-fly:** Dynamic augmentation for infinite variation and memory efficiency
   - **Hybrid:** Combine both for optimal balance

3. **Comprehensive Metrics:**
   - **Canonical pass-rate:** Core validity metric for TKS outputs
   - **Diversity scores:** Ensure broad semantic coverage
   - **Loss impact:** Validate augmentation effectiveness

4. **Scalability Path:**
   - Start small (100-1K scenarios) for proof of concept
   - Scale gradually with caching and parallel processing
   - Implement quality filtering before production deployment

### Recommended Starting Configuration

```python
# Initial augmentation setup for experimentation

AUGMENTATION_CONFIG = {
    # Preprocessing settings
    "preprocess": {
        "axes_combinations": [
            {"W"},      # World-only
            {"N"},      # Noetic-only
            {"W", "N"}, # World + Noetic
        ],
        "use_anti_attractor": True,
        "quality_filter": True,
        "cache_inversions": True,
    },

    # On-the-fly settings
    "online": {
        "augment_prob": 0.5,
        "axes_pool": [{"W"}, {"N"}, {"F"}, {"W", "N"}],
        "anti_attractor_prob": 0.3,
        "num_elements": 3,
    },

    # Metrics tracking
    "metrics": {
        "log_frequency": 100,  # steps
        "eval_samples": 50,
        "track_diversity": True,
        "track_canonical": True,
    }
}
```

### Next Steps

1. **Immediate (Week 1):**
   - Implement basic preprocessing script (`augment_corpus.py`)
   - Generate pilot augmented dataset (100 stories → 500 augmented)
   - Validate canonical pass-rates on augmented data

2. **Short-term (Weeks 2-4):**
   - Build PyTorch `TKSAugmentationDataset`
   - Run small training experiment comparing baseline vs augmented
   - Measure loss curves and diversity improvements

3. **Medium-term (Weeks 5-8):**
   - Scale to full corpus with optimization
   - Implement quality filtering pipeline
   - Integrate canonical validator for continuous monitoring

4. **Long-term (Week 9+):**
   - Deploy to production training pipeline
   - Monitor downstream task performance
   - Iterate on augmentation strategies based on empirical results

---

## Appendix: Quick Reference

### API Summary

| Function | Module | Purpose | Key Parameters |
|----------|--------|---------|----------------|
| `InvertStory` | `scenario_inversion.py` | Multi-axis scenario inversion | `axes`, `mode`, `target` |
| `AntiAttractorInvert` | `anti_attractor.py` | Anti-attractor synthesis | `return_signature` |
| `EncodeStory` | `scenario_inversion.py` | Story → TKS expression | `strict` |
| `DecodeStory` | `scenario_inversion.py` | TKS expression → Story | - |
| `compute_attractor_signature` | `anti_attractor.py` | Extract attractor signature | - |
| `ScenarioInvert` | `scenario_inversion.py` | Invert TKS expression directly | `axes`, `mode`, `target` |

### Axis Reference

| Code | Full Name | Inverts |
|------|-----------|---------|
| N | Noetic | Mind principles (2↔3, 5↔6, 8↔9) |
| W | World | Domains (A↔D, B↔C) |
| F | Foundation | Archetypal alignments (1↔7, 2↔6, 3↔5) |
| E | Element | Combined world + noetic |
| S | SubFoundation | Foundation world assignments |
| A | Acquisition | Knowledge/experience markers |
| P | Polarity | Overall energetic signature |

### File Paths

- **Inversion API:** `C:\Users\wakil\downloads\everthing-tootra-tks\scenario_inversion.py`
- **Anti-Attractor API:** `C:\Users\wakil\downloads\everthing-tootra-tks\anti_attractor.py`
- **Training Integration (to create):** `C:\Users\wakil\downloads\everthing-tootra-tks\training\tks_dataset.py`
- **Preprocessing Script (to create):** `C:\Users\wakil\downloads\everthing-tootra-tks\scripts\augment_corpus.py`

---

## 9. Phase 1 Implementation Steps

**Status:** In Progress
**Date:** 2025-12-14
**Goal:** Establish scaffolding for inversion/anti-attractor integration into data pipeline

### 9.1 Integration Hook Points Identified

Based on analysis of existing codebase, the following hook points have been identified for Phase 1 integration:

#### Hook Point 1: Synthetic Data Generation Pipeline
**Location:** `scripts/generate_pilot_data.py`
**Current Flow:**
```
Random Element Generation → TKS Equation Construction → JSON Export
```
**Integration Opportunity:**
- Add augmentation step after equation generation
- Apply InvertStory to generated scenarios
- Generate anti-attractor pairs for contrastive learning
- Preserve canonical validity through EncodeStory/DecodeStory

**Modification Strategy:**
```python
# BEFORE (current):
equation = generate_random_equation()
dataset.append(equation.to_dict())

# AFTER (Phase 1):
equation = generate_random_equation()
dataset.append(equation.to_dict())  # Original

# Inversion augmentation
inverted = generate_inverted_scenario(equation, axes={"W", "N"})
if validate_canonical(inverted):
    dataset.append(inverted)

# Anti-attractor augmentation
anti_attractor = generate_anti_attractor_pair(equation)
if validate_canonical(anti_attractor):
    dataset.append(anti_attractor)
```

#### Hook Point 2: Teacher Model Outputs
**Location:** `scripts/run_teacher.py` (teacher output augmentation)
**Current Flow:**
```
Model Inference → Output Generation → Storage
```
**Integration Opportunity:**
- Intercept teacher outputs before storage
- Apply semantic inversion to diversify training data
- Create contrastive pairs for student model training

**Modification Strategy:**
```python
# Teacher outputs → augmented training corpus
teacher_output = model.generate(prompt)

# Store original
training_corpus.append(teacher_output)

# Generate augmented variants
for axes in [{"W"}, {"N"}, {"W", "N"}]:
    augmented = InvertStory(teacher_output, axes=axes, mode="soft")
    if validate_canonical(augmented):
        training_corpus.append({
            "text": augmented["story_inverted"],
            "parent": teacher_output,
            "augmentation_type": "inverted",
            "axes": list(axes)
        })
```

#### Hook Point 3: Training Data Loader
**Location:** `training.PilotDataset` (referenced in `run_pilot_training.py`)
**Current Flow:**
```
Load JSONL → Parse → Tokenize → Batch
```
**Integration Opportunity:**
- Add on-the-fly augmentation in __getitem__
- Probabilistic augmentation (configurable via augment_prob)
- Cache inverted expressions for efficiency

**Modification Strategy:**
```python
class TKSAugmentedDataset(PilotDataset):
    def __init__(self, *args, augment_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.augment_prob = augment_config.get("prob", 0.0) if augment_config else 0.0
        self.axes_pool = augment_config.get("axes_pool", [{"W"}, {"N"}])

    def __getitem__(self, idx):
        item = super().__getitem__(idx)

        if random.random() < self.augment_prob:
            # Apply augmentation
            item = self._augment(item)

        return item
```

### 9.2 Code Path Outline (Pseudocode)

#### Path A: Offline Preprocessing (Batch Augmentation)
```python
# scripts/generate_augmented_data.py

def main(input_corpus: Path, output_corpus: Path, config: AugmentConfig):
    """
    Offline batch augmentation pipeline.

    Input:  Original training corpus (JSONL)
    Output: Augmented training corpus (JSONL)

    Flow:
        1. Load corpus
        2. For each story:
            a. Validate canonical (skip if invalid)
            b. Generate inversions (multiple axes)
            c. Generate anti-attractor pairs
            d. Validate all augmentations
        3. Deduplicate signatures
        4. Save augmented corpus
    """

    stories = load_corpus(input_corpus)
    augmented_data = []

    # Metrics tracking
    metrics = {
        "original_count": 0,
        "inverted_count": 0,
        "anti_attractor_count": 0,
        "validation_failures": 0,
        "augmentation_ratio": 0.0
    }

    for story in stories:
        # Validate original
        try:
            expr_original = validate_canonical(story)
            metrics["original_count"] += 1
        except ValueError:
            metrics["validation_failures"] += 1
            continue

        # Store original
        augmented_data.append({
            "story": story,
            "expr": expr_original.raw,
            "type": "original"
        })

        # Generate inversions
        for axes in config.axes_combinations:
            inverted = generate_inverted_scenarios(
                story=story,
                axes=axes,
                mode=config.inversion_mode
            )

            if validate_canonical(inverted["story"]):
                augmented_data.append({
                    "story": inverted["story"],
                    "expr": inverted["expr"].raw,
                    "type": "inverted",
                    "axes": list(axes),
                    "parent": story
                })
                metrics["inverted_count"] += 1

        # Generate anti-attractor pairs
        if config.use_anti_attractor:
            anti_pair = generate_anti_attractor_pairs(
                expr=expr_original,
                num_elements=config.anti_attractor_elements
            )

            if validate_canonical(anti_pair["story"]):
                augmented_data.append({
                    "story": anti_pair["story"],
                    "expr": anti_pair["expr"].raw,
                    "type": "anti_attractor",
                    "signature": anti_pair["signature"],
                    "parent": story
                })
                metrics["anti_attractor_count"] += 1

    # Calculate metrics
    metrics["augmentation_ratio"] = (
        (metrics["inverted_count"] + metrics["anti_attractor_count"])
        / metrics["original_count"]
    )

    # Save
    save_augmented_corpus(augmented_data, output_corpus)
    save_metrics(metrics, output_corpus.with_suffix(".metrics.json"))

    return metrics
```

#### Path B: On-the-Fly Augmentation (DataLoader)
```python
# training/tks_augmented_dataset.py

class TKSAugmentedDataset(Dataset):
    """
    PyTorch Dataset with on-the-fly augmentation.

    Applies probabilistic augmentation during training:
    - Inversion: 70% of augmentations
    - Anti-attractor: 30% of augmentations
    """

    def __getitem__(self, idx):
        # Load original story
        story = self.stories[idx]

        # Probabilistic augmentation
        if random.random() < self.augment_prob:
            if random.random() < 0.7:  # Inversion
                axes = random.choice(self.axes_pool)
                result = generate_inverted_scenarios(story, axes, mode="soft")
                return {
                    "text": result["story"],
                    "augmented": True,
                    "type": "inverted"
                }
            else:  # Anti-attractor
                expr = EncodeStory(story)
                result = generate_anti_attractor_pairs(expr)
                return {
                    "text": result["story"],
                    "augmented": True,
                    "type": "anti_attractor"
                }

        # Return original
        return {
            "text": story,
            "augmented": False
        }
```

### 9.3 Validation Checkpoint

**Canonical Validation Function:**
```python
def validate_canonical(story_or_expr) -> TKSExpression:
    """
    Validate that story/expression conforms to TKS canonical semantics.

    Checks:
        1. Valid world letters (A, B, C, D only)
        2. Valid noetic indices (1-10)
        3. Valid operators from ALLOWED_OPS
        4. Structural consistency (elements/ops alignment)
        5. Foundation validity (if present)

    Returns:
        TKSExpression if valid

    Raises:
        ValueError with detailed error message if invalid
    """
    if isinstance(story_or_expr, str):
        # Encode story to expression
        try:
            expr = EncodeStory(story_or_expr, strict=True)
        except ValueError as e:
            raise ValueError(f"Canonical validation failed (encoding): {e}")
    else:
        expr = story_or_expr

    # Structural checks
    if not expr.elements:
        raise ValueError("Empty elements list")

    if len(expr.ops) != len(expr.elements) - 1:
        raise ValueError(
            f"Operator count mismatch: {len(expr.ops)} ops for {len(expr.elements)} elements"
        )

    # World validation
    valid_worlds = {"A", "B", "C", "D"}
    for elem in expr.elements:
        world = elem[0]
        if world not in valid_worlds:
            raise ValueError(f"Invalid world letter: {world} (must be A, B, C, or D)")

    # Noetic validation
    for elem in expr.elements:
        try:
            noetic = int(elem[1:])
            if not (1 <= noetic <= 10):
                raise ValueError(f"Invalid noetic index: {noetic} (must be 1-10)")
        except (ValueError, IndexError) as e:
            raise ValueError(f"Malformed element: {elem}")

    # Operator validation
    from narrative.constants import ALLOWED_OPS
    for op in expr.ops:
        if op not in ALLOWED_OPS:
            raise ValueError(f"Invalid operator: {op}")

    return expr
```

**Checkpoint Integration:**
```python
# Before adding to training corpus
augmented_scenarios = generate_all_augmentations(corpus)

validated_scenarios = []
validation_metrics = {
    "total": len(augmented_scenarios),
    "passed": 0,
    "failed": 0,
    "pass_rate": 0.0
}

for scenario in augmented_scenarios:
    try:
        expr = validate_canonical(scenario)
        validated_scenarios.append(scenario)
        validation_metrics["passed"] += 1
    except ValueError as e:
        validation_metrics["failed"] += 1
        log_validation_failure(scenario, e)

validation_metrics["pass_rate"] = (
    validation_metrics["passed"] / validation_metrics["total"]
)

# Only proceed if pass rate meets threshold
if validation_metrics["pass_rate"] < 0.90:  # 90% threshold
    raise RuntimeError(
        f"Validation pass rate too low: {validation_metrics['pass_rate']:.2%}"
    )
```

### 9.4 Metrics to Track

#### Metric 1: Validator Pass-Rate
```python
def compute_validator_pass_rate(scenarios: List[str]) -> Dict[str, Any]:
    """
    Compute canonical validation pass rate for scenarios.

    Returns:
        {
            "total": int,
            "valid": int,
            "pass_rate": float,
            "world_validity": float,
            "noetic_validity": float,
            "operator_validity": float,
            "structural_validity": float
        }
    """
    total = len(scenarios)
    metrics = {
        "total": total,
        "valid": 0,
        "world_valid": 0,
        "noetic_valid": 0,
        "operator_valid": 0,
        "structural_valid": 0
    }

    for scenario in scenarios:
        try:
            expr = EncodeStory(scenario, strict=True)

            # Component checks
            worlds_ok = all(e[0] in "ABCD" for e in expr.elements)
            noetics_ok = all(1 <= int(e[1:]) <= 10 for e in expr.elements)
            ops_ok = all(op in ALLOWED_OPS for op in expr.ops)
            structure_ok = len(expr.ops) == len(expr.elements) - 1

            if worlds_ok:
                metrics["world_valid"] += 1
            if noetics_ok:
                metrics["noetic_valid"] += 1
            if ops_ok:
                metrics["operator_valid"] += 1
            if structure_ok:
                metrics["structural_valid"] += 1

            if all([worlds_ok, noetics_ok, ops_ok, structure_ok]):
                metrics["valid"] += 1
        except:
            pass

    return {
        "total": total,
        "valid": metrics["valid"],
        "pass_rate": metrics["valid"] / total if total > 0 else 0.0,
        "world_validity": metrics["world_valid"] / total if total > 0 else 0.0,
        "noetic_validity": metrics["noetic_valid"] / total if total > 0 else 0.0,
        "operator_validity": metrics["operator_valid"] / total if total > 0 else 0.0,
        "structural_validity": metrics["structural_valid"] / total if total > 0 else 0.0
    }
```

**Tracking Integration:**
```python
# During augmentation
original_metrics = compute_validator_pass_rate(original_corpus)
augmented_metrics = compute_validator_pass_rate(augmented_corpus)

print(f"Original pass rate: {original_metrics['pass_rate']:.2%}")
print(f"Augmented pass rate: {augmented_metrics['pass_rate']:.2%}")

# Log to file
with open("augmentation_metrics.json", "w") as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "original": original_metrics,
        "augmented": augmented_metrics
    }, f, indent=2)
```

#### Metric 2: Augmentation Ratio
```python
def compute_augmentation_ratio(corpus_metadata: List[Dict]) -> Dict[str, float]:
    """
    Compute ratio of augmented scenarios to originals.

    Returns:
        {
            "total_ratio": float,      # (inverted + anti) / original
            "inversion_ratio": float,  # inverted / original
            "anti_attractor_ratio": float  # anti / original
        }
    """
    original_count = sum(1 for item in corpus_metadata if item["type"] == "original")
    inverted_count = sum(1 for item in corpus_metadata if item["type"] == "inverted")
    anti_count = sum(1 for item in corpus_metadata if item["type"] == "anti_attractor")

    if original_count == 0:
        return {"total_ratio": 0.0, "inversion_ratio": 0.0, "anti_attractor_ratio": 0.0}

    return {
        "total_ratio": (inverted_count + anti_count) / original_count,
        "inversion_ratio": inverted_count / original_count,
        "anti_attractor_ratio": anti_count / original_count
    }
```

**Target Ratios:**
- Phase 1 (Proof of Concept): 2.0x (2 augmented per original)
- Phase 2 (Scaled): 5.0x (multiple axes + anti-attractor)
- Phase 3 (Production): 3.0x (optimized balance)

#### Metric 3: Loss Impact
```python
def track_loss_impact(
    baseline_losses: List[float],
    augmented_losses: List[float]
) -> Dict[str, float]:
    """
    Compare training loss between baseline and augmented training.

    Returns:
        {
            "baseline_final": float,
            "augmented_final": float,
            "improvement": float,       # (baseline - augmented) / baseline
            "convergence_speedup": float  # epochs saved to reach baseline final
        }
    """
    baseline_final = baseline_losses[-1]
    augmented_final = augmented_losses[-1]
    improvement = (baseline_final - augmented_final) / baseline_final

    # Find convergence point
    convergence_epoch = 0
    for i, loss in enumerate(augmented_losses):
        if loss <= baseline_final:
            convergence_epoch = i
            break

    speedup = (len(baseline_losses) - convergence_epoch) / len(baseline_losses)

    return {
        "baseline_final": baseline_final,
        "augmented_final": augmented_final,
        "improvement": improvement,
        "convergence_speedup": speedup
    }
```

**Experimental Setup:**
```python
# Baseline experiment (no augmentation)
baseline_config = TrainingConfig(
    data_file="original_corpus.jsonl",
    epochs=20,
    augmentation=None
)
baseline_losses = run_training(baseline_config)

# Augmented experiment
augmented_config = TrainingConfig(
    data_file="augmented_corpus.jsonl",
    epochs=20,
    augmentation=None  # Pre-augmented offline
)
augmented_losses = run_training(augmented_config)

# Compare
impact = track_loss_impact(baseline_losses, augmented_losses)
print(f"Loss improvement: {impact['improvement']:.2%}")
print(f"Convergence speedup: {impact['convergence_speedup']:.2%}")
```

### 9.5 Phase 1 Success Criteria

**Deliverables:**
1. ✓ Updated `docs/TRAINING_INTEGRATION_PLAN.md` with Phase 1 section
2. ✓ Stub script `scripts/generate_augmented_data.py` with function signatures
3. ✓ Hook points documented and validated
4. ✓ Validation checkpoint implemented
5. ✓ Metrics tracking functions defined

**Validation Checkpoints:**
1. Canonical validator pass-rate >= 90% on augmented data
2. Augmentation ratio >= 2.0x (2 augmented per original)
3. No loss of canonical semantics (all augmented scenarios parseable)
4. Metrics pipeline functional (validation, ratios, loss tracking)

**Next Phase Trigger:**
- Phase 1 deliverables complete
- Validation checkpoints passed
- Initial metrics collected from pilot corpus (100 scenarios)
- Team approval for Phase 2 (full implementation)

### 9.6 Phase 1 Timeline

**Week 1 (Current):**
- [x] Identify hook points
- [x] Document integration strategy
- [x] Create stub script with signatures
- [ ] Test validation checkpoint on pilot data

**Week 2:**
- [ ] Implement generate_inverted_scenarios()
- [ ] Implement generate_anti_attractor_pairs()
- [ ] Run validation on 100 seed scenarios
- [ ] Collect baseline metrics

**Phase 1 Completion Target:** 2 weeks from 2025-12-14

---

## Phase 1.5: Augmentation Pipeline

**Status:** Scaffolding Complete
**Date:** 2025-12-14

### Overview

The augmentation pipeline script (`scripts/generate_augmented_data.py`) provides a unified interface for generating augmented training data using scenario inversion and anti-attractor synthesis. This Phase 1.5 implementation establishes the scaffolding and function signatures for batch augmentation workflows.

### Augmentation Workflow

The pipeline follows a multi-stage augmentation process:

```
Input Corpus (JSONL)
    |
    v
[Load & Validate] - Canonical validation of original scenarios
    |
    v
[Scenario Inversion] - Apply multi-axis inversions
    |                   (configurable axes: N, E, W, F, S, A, P)
    v
[Anti-Attractor Synthesis] - Generate counter-scenarios
    |                          (optional, enabled via --use-anti-attractor)
    v
[Validation Filter] - Drop invalid augmentations (optional)
    |
    v
[Metadata Enrichment] - Add aug_type, source_id, validator_pass
    |
    v
Output Corpus (JSONL) + Metrics (JSON)
```

### Output Fields

Each augmented scenario in the output JSONL contains the following fields:

- **story**: Augmented scenario text
- **expr**: TKS expression (canonical form)
- **aug_type**: Augmentation type (`"original"`, `"inverted"`, or `"anti_attractor"`)
- **source_id**: Reference to parent scenario (for inverted/anti-attractor entries)
- **validator_pass**: Boolean indicating canonical validation result
- **expr_elements**: List of TKS elements (e.g., `["B2", "D5"]`)
- **expr_ops**: List of operators connecting elements (e.g., `["->", "+T"]`)

Additional fields for inverted scenarios:
- **axes**: List of inversion axes applied (e.g., `["W", "N"]`)
- **mode**: Inversion mode (`"soft"`, `"hard"`, or `"targeted"`)

Additional fields for anti-attractor scenarios:
- **signature**: Attractor signature metadata (element counts, polarity, dominant world/noetic)

### Usage Example

```bash
# Basic augmentation with world and noetic axes
python scripts/generate_augmented_data.py \
  --input data/pilot/stories.jsonl \
  --output data/pilot/augmented.jsonl \
  --axes W N F \
  --use-anti-attractor \
  --validate

# Production mode with strict validation
python scripts/generate_augmented_data.py \
  --input data/training/corpus.jsonl \
  --output data/training/augmented.jsonl \
  --axes W N \
  --mode soft \
  --min-pass-rate 0.95 \
  --save-metrics

# Lenient mode for experimental data
python scripts/generate_augmented_data.py \
  --input data/experimental/raw.jsonl \
  --output data/experimental/augmented.jsonl \
  --axes W N F \
  --lenient \
  --validate
```

### Integration Points

The augmentation script integrates with:
- **Preprocessing Pipeline**: Offline batch augmentation before training
- **Training DataLoader**: Can be used to pre-generate augmented datasets
- **Validation Framework**: Canonical validation ensures TKS semantic consistency
- **Metrics Tracking**: Outputs augmentation metrics for monitoring data quality

### Next Steps

Phase 2 implementation will add:
- Full implementation of `generate_inverted_scenarios()` and `generate_anti_attractor_pairs()`
- Integration with `scenario_inversion.py` and `anti_attractor.py` APIs
- Parallel batch processing for performance optimization
- Quality filtering and deduplication

---

## Phase 2 Implementation: Training Integration

**Status:** Complete
**Date:** 2025-12-14
**Goal:** Implement minimal training path that consumes augmented data

### Overview

Phase 2 establishes the training infrastructure to consume augmented JSONL data generated by `scripts/generate_augmented_data.py`. The implementation provides a functional data pipeline with minimal training loop scaffolding, ready for integration with actual model architectures.

### Implementation Details

#### 2.1 Core Functions Implemented

**Function: `load_augmented_corpus(path, filter_validated=False)`**

Location: `scripts/train_with_augmented.py:14-74`

Loads augmented JSONL corpus with optional validation filtering.

Features:
- JSONL parsing with error handling for malformed entries
- Optional filtering by `validator_pass` field
- Preserves all augmentation metadata (aug_type, axes, mode, etc.)
- Comprehensive error reporting for debugging

Usage:
```python
# Load all entries
corpus = load_augmented_corpus("output/augmented.jsonl")

# Load only validated entries
validated_corpus = load_augmented_corpus(
    "output/augmented.jsonl",
    filter_validated=True
)
```

**Function: `prepare_training_batch(entries, config)`**

Location: `scripts/train_with_augmented.py:77-160`

Converts augmented entries to model-ready inputs and targets.

Configuration Options:
- `max_length`: Maximum sequence length (default: 512)
- `use_expr`: If True, train on TKS expressions; if False, use stories
- `include_metadata`: If True, prefix inputs with aug_type tag

Handles three augmentation types:
1. **original**: Baseline training examples
2. **inversion**: Semantic transformations via axis inversions
3. **anti_attractor**: Counter-scenarios for robustness

Current implementation uses simple string encoding (language modeling setup where input = target). Future enhancements will add:
- Tokenization via transformer tokenizers
- Contrastive pairs for inversion learning
- Embedding-based representations

Usage:
```python
config = {
    'max_length': 512,
    'use_expr': True,  # Train on expressions
    'include_metadata': False
}
inputs, targets = prepare_training_batch(batch_entries, config)
```

**Function: `train_step(batch, config)`**

Location: `scripts/train_with_augmented.py:163-203`

Executes single training step with dummy loss computation.

Current implementation:
- Computes dummy loss based on batch statistics
- MSE between actual and expected batch size
- Adds variance from text length distribution
- No actual model forward/backward pass (stub)

Dummy loss formula:
```
loss = |batch_size - expected_size| / expected_size
     + avg_text_length / 100 * 0.1
     (minimum: 0.01)
```

Future enhancements:
- Add actual model (transformer, LSTM, etc.)
- Implement real loss function (cross-entropy, contrastive)
- Add optimizer and backpropagation
- Track gradients and learning curves

**Function: `run_smoke_test(data_path)`**

Location: `scripts/train_with_augmented.py:206-287`

Comprehensive smoke test suite for data pipeline verification.

Tests performed:
1. Load augmented corpus from JSONL
2. Verify entry structure (required fields)
3. Test batch preparation with stories
4. Test batch preparation with expressions
5. Verify aug_type filtering
6. Test validation filtering
7. End-to-end batch processing with dummy loss

Usage:
```bash
python scripts/train_with_augmented.py \
    --data output/augmented.jsonl \
    --test
```

#### 2.2 CLI Configuration Options

The training script supports extensive CLI configuration:

**Data Paths:**
- `--data`: Path to augmented JSONL (required)
- `--original-data`: Optional path to original corpus for comparison

**Augmentation Toggles:**
- `--use-augmented`: Use augmented samples (default: True)
- `--filter-validated`: Only use entries with validator_pass=True

**Training Configuration:**
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size for training (default: 32)
- `--learning-rate`: Learning rate for optimizer (default: 1e-4)
- `--max-length`: Maximum sequence length (default: 512)

**Input Format Options:**
- `--use-expr`: Train on TKS expressions instead of stories
- `--include-metadata`: Include aug_type metadata in inputs

**Output Options:**
- `--output-dir`: Directory to save models (default: output/models)

**Testing:**
- `--test`: Run smoke test instead of training

#### 2.3 Training Loop Structure

The main training loop implements a minimal but functional structure:

```
1. Load augmented corpus
   ↓
2. Filter by validation status (optional)
   ↓
3. Analyze augmentation type distribution
   ↓
4. Shuffle corpus
   ↓
5. Create batches
   ↓
6. For each epoch:
   a. Iterate over batches
   b. Prepare batch (entries → inputs/targets)
   c. Execute training step (dummy loss)
   d. Log batch metrics every 10 batches
   e. Compute epoch summary (avg loss, std dev)
   ↓
7. Print final summary and next steps
```

Key features:
- Handles variable batch sizes (last batch may be smaller)
- Skips empty batches
- Computes epoch-level statistics (avg loss, std deviation)
- Provides clear progress logging

#### 2.4 Example Usage

**Basic Training:**
```bash
python scripts/train_with_augmented.py \
    --data output/augmented_corpus.jsonl \
    --epochs 5 \
    --batch-size 16
```

**Training with Validation Filtering:**
```bash
python scripts/train_with_augmented.py \
    --data output/augmented_corpus.jsonl \
    --filter-validated \
    --epochs 10 \
    --batch-size 32
```

**Training on TKS Expressions:**
```bash
python scripts/train_with_augmented.py \
    --data output/augmented_corpus.jsonl \
    --use-expr \
    --include-metadata \
    --epochs 20
```

**Running Smoke Test:**
```bash
python scripts/train_with_augmented.py \
    --data output/augmented_corpus.jsonl \
    --test
```

#### 2.5 Output Format

**Training Output:**
```
======================================================================
TKS TRAINING WITH AUGMENTED DATA - Phase 2 Implementation
======================================================================

Loading augmented corpus from: output/augmented_corpus.jsonl
Loaded 150 entries from augmented corpus

Augmentation type distribution:
  - anti_attractor: 50
  - inversion: 50
  - original: 50

Training configuration:
  Epochs: 5
  Batch size: 16
  Learning rate: 0.0001
  Max length: 512
  Use expressions: False
  Include metadata: False
  Filter validated: False

Preparing batches...
Total batches: 10

======================================================================
TRAINING LOOP (Minimal Stub)
======================================================================

Epoch 1/5
----------------------------------------------------------------------
  Batch 0/10: size=16, loss=0.0234

  Epoch 1 Summary:
    Average loss: 0.0245
    Total batches: 10
    Loss std dev: 0.0012

[... epochs 2-5 ...]

======================================================================
TRAINING COMPLETE
======================================================================
Total entries processed: 150
Total epochs: 5
Total batches per epoch: 10

Note: This is a minimal stub implementation.
No actual model training occurred - only data pipeline verification.

Next steps:
  1. Add actual model (e.g., transformer, LSTM)
  2. Implement real loss function (e.g., cross-entropy)
  3. Add optimizer and backpropagation
  4. Add validation loop
  5. Add model checkpointing
======================================================================
```

**Smoke Test Output:**
```
======================================================================
SMOKE TEST - Data Pipeline Verification
======================================================================

[Test 1] Loading augmented corpus...
  ✓ Loaded 150 entries

[Test 2] Checking entry structure...
  ✓ All required fields present

[Test 3] Testing batch preparation (stories)...
  ✓ Generated 10 training pairs

[Test 4] Testing batch preparation (expressions)...
  ✓ Generated 10 expression pairs

[Test 5] Testing aug_type filtering...
  Found aug_types: {'original', 'inversion', 'anti_attractor'}
  ✓ Aug types present: anti_attractor, inversion, original

[Test 6] Testing validation filtering...
  ✓ Validation rate: 95.33% (143/150)

[Test 7] Testing batch processing...
  ✓ Batch processed, dummy loss: 0.0198

======================================================================
✓ ALL SMOKE TESTS PASSED
======================================================================
```

#### 2.6 Integration with Augmentation Pipeline

The training script integrates seamlessly with `scripts/generate_augmented_data.py`:

**Workflow:**
```
1. Generate augmented data:
   python scripts/generate_augmented_data.py \
       --input data/original.jsonl \
       --output output/augmented.jsonl \
       --axes W N F \
       --use-anti-attractor

2. Verify data pipeline:
   python scripts/train_with_augmented.py \
       --data output/augmented.jsonl \
       --test

3. Train model (minimal stub):
   python scripts/train_with_augmented.py \
       --data output/augmented.jsonl \
       --epochs 10 \
       --batch-size 32
```

**Expected Data Format:**

Input JSONL from augmentation script:
```json
{
  "id": "entry_001",
  "story": "A spiritual teacher causes enlightenment",
  "expr": "A5 -> D2",
  "expr_elements": ["A5", "D2"],
  "expr_ops": ["->"],
  "aug_type": "original",
  "source_id": "entry_001",
  "validator_pass": true
}
```

Consumed by training script:
- `story` or `expr` → model inputs
- `aug_type` → training strategy (original/inversion/anti_attractor)
- `validator_pass` → optional filtering
- `axes`, `mode` → metadata for analysis

#### 2.7 Metrics Tracked

Current implementation tracks:

**Corpus-level metrics:**
- Total entries loaded
- Augmentation type distribution
- Validation pass rate

**Training-level metrics:**
- Batch size per step
- Dummy loss per batch
- Average loss per epoch
- Loss standard deviation per epoch

**Future metrics:**
- Canonical validation pass rate on model outputs
- Diversity scores for generated scenarios
- Loss impact from augmented data vs baseline
- Contrastive loss for inversion pairs

#### 2.8 Future Enhancements

**Short-term (Phase 3):**
1. Add actual model architecture (e.g., GPT-2, LSTM)
2. Implement real loss function (cross-entropy)
3. Add optimizer (Adam, SGD)
4. Implement backpropagation
5. Add validation loop with separate dataset
6. Add model checkpointing

**Medium-term (Phase 4):**
1. Implement contrastive learning for inversions
2. Add TKS-aware embeddings
3. Implement canonical validator integration
4. Add metrics tracking (W&B, TensorBoard)
5. Add early stopping
6. Add learning rate scheduling

**Long-term (Phase 5):**
1. Multi-task learning (stories + expressions)
2. Foundation-aware attention mechanisms
3. Hierarchical models for world/noetic/foundation
4. Interpretability tools for TKS reasoning
5. Fine-tuning pipelines for downstream tasks

### Phase 2 Success Criteria

**Deliverables:**
- ✓ `load_augmented_corpus()` implemented and tested
- ✓ `prepare_training_batch()` implemented with multiple config options
- ✓ CLI configuration options added
- ✓ Minimal training loop implemented
- ✓ Smoke test function implemented
- ✓ Documentation updated with Phase 2 section

**Validation Checkpoints:**
- ✓ Can load augmented JSONL files
- ✓ Can filter by validator_pass
- ✓ Can prepare batches with stories and expressions
- ✓ Can iterate over batches with dummy loss
- ✓ Smoke tests pass for sample data

**Integration Verified:**
- ✓ Consumes output from `generate_augmented_data.py`
- ✓ Handles all aug_types (original, inversion, anti_attractor)
- ✓ Respects validation metadata
- ✓ Supports both story and expression inputs

### Next Steps

**Immediate (Week 1):**
1. Create sample augmented dataset (10-20 entries)
2. Run smoke test on sample data
3. Verify end-to-end pipeline (augmentation → training)
4. Document any issues or edge cases

**Short-term (Weeks 2-3):**
1. Integrate actual model architecture
2. Add real loss computation
3. Implement training/validation split
4. Add metrics logging

**Medium-term (Month 2):**
1. Scale to full corpus (1000+ entries)
2. Implement contrastive learning
3. Add canonical validation on outputs
4. Performance optimization

---

**Document Status:** Phase 2 Implementation Complete
**Next Action:** Create sample augmented dataset and run smoke test

---

## 10. Monitoring: Data/Model Metrics Tracking

**Status:** Implemented
**Date:** 2025-12-14

### 10.1 Overview

The TKS augmentation and training pipeline includes comprehensive metrics tracking to monitor data quality, validation rates, and distribution characteristics. The monitoring system is lightweight with no heavy dependencies (no pandas/wandb required in the core implementation).

**Key Features:**
- In-memory metric accumulation
- JSON export for persistence
- Formatted console output
- Track augmentation, validation, and distribution statistics
- Per-epoch and per-batch tracking
- Baseline vs augmented comparisons

### 10.2 Augmentation Metrics Module

**Location:** `C:\Users\wakil\downloads\everthing-tootra-tks\scripts\augmentation_metrics.py`

#### Core Components

**1. AugmentationLogger Class**

Primary interface for metrics tracking:

```python
from augmentation_metrics import AugmentationLogger

# Initialize logger
logger = AugmentationLogger()

# Log individual entry
logger.log_entry(entry_dict)

# Log batch of entries
logger.log_batch(entries_list)

# Get summary statistics
summary = logger.get_summary()

# Print formatted summary
logger.print_summary(detailed=True)

# Save to JSON
logger.save("metrics.json")

# Reset for next batch/epoch
logger.reset()
```

**2. Helper Functions**

```python
from augmentation_metrics import (
    compute_batch_stats,
    track_epoch_stats,
    compare_metrics
)

# Compute statistics for a batch
stats = compute_batch_stats(entries)

# Track and save epoch metrics
epoch_stats = track_epoch_stats(
    epoch=5,
    entries=epoch_entries,
    output_dir="metrics/"
)

# Compare baseline vs augmented
comparison = compare_metrics(baseline_summary, augmented_summary)
```

### 10.3 Metrics Tracked

#### Augmentation Metrics

| Metric | Description | Target Value |
|--------|-------------|--------------|
| `original_count` | Number of original entries | N/A |
| `inversion_count` | Number of inverted entries | 2-3x original |
| `anti_attractor_count` | Number of anti-attractor entries | 0.5-1x original |
| `augmentation_ratio` | Total augmented / original | 2.5-5x |
| `inversion_ratio` | Inverted / original | 2-3x |
| `anti_attractor_ratio` | Anti-attractor / original | 0.5-1x |
| `axes_usage` | Count per axis (W, N, F, etc.) | Balanced |
| `mode_counts` | Count per mode (soft, hard, targeted) | Mostly soft |

#### Validation Metrics

| Metric | Description | Target Value |
|--------|-------------|--------------|
| `pass_rate` | Overall validation pass rate | > 90% |
| `world_validity_rate` | % with valid world letters (A,B,C,D) | > 95% |
| `noetic_validity_rate` | % with valid noetic indices (1-10) | > 95% |
| `operator_validity_rate` | % with valid operators | > 98% |
| `structural_validity_rate` | % with valid structure | > 95% |
| `foundation_validity_rate` | % with valid foundations (1-7) | > 90% |
| `error_counts` | Errors by type | Monitor trends |

#### Distribution Metrics

| Metric | Description | Ideal Range |
|--------|-------------|-------------|
| `world_distribution` | % for each world (A, B, C, D) | 20-30% each |
| `noetic_distribution` | % for each noetic (1-10) | 8-12% each |
| `operator_distribution` | % for each operator | Varies by corpus |
| `foundation_distribution` | % for each foundation (1-7) | 10-20% each |
| `polarity_distribution` | Positive/negative/neutral % | Balanced |

### 10.4 Integration with Augmentation Pipeline

**Location:** `C:\Users\wakil\downloads\everthing-tootra-tks\scripts\generate_augmented_data.py`

The augmentation script automatically logs all generated entries:

```python
# Logger is initialized in augment_corpus()
logger = AugmentationLogger()

# Each entry is logged as it's created
logger.log_entry(original_entry)
logger.log_entry(inversion_entry)
logger.log_entry(anti_entry)

# At end of augmentation
logger.save(output_path.with_suffix(".detailed_metrics.json"))
logger.print_summary(detailed=True)
```

**Output Files Generated:**

After running `generate_augmented_data.py`:

```
data/pilot/
├── augmented.jsonl                         # Augmented corpus
├── augmented.jsonl.metrics.json            # Basic metrics
└── augmented.jsonl.detailed_metrics.json   # Detailed logger metrics
```

### 10.5 Integration with Training Pipeline

**Location:** `C:\Users\wakil\downloads\everthing-tootra-tks\scripts\train_with_augmented.py`

The training script tracks per-epoch metrics:

```python
from augmentation_metrics import AugmentationLogger, track_epoch_stats

# Initialize logger for epoch
epoch_logger = AugmentationLogger()

# Log each batch
for batch in dataloader:
    epoch_logger.log_batch(batch_entries)
    # ... training step ...

# Save epoch metrics
output_dir = Path(args.output_dir) / "metrics"
epoch_stats = track_epoch_stats(
    epoch=epoch_num,
    entries=epoch_entries,
    output_dir=output_dir
)

# Reset for next epoch
epoch_logger.reset()
```

**Output Files Generated:**

After each epoch:

```
output/models/metrics/
├── epoch_001_metrics.json
├── epoch_002_metrics.json
└── epoch_003_metrics.json
```

### 10.6 Usage Examples

#### Example 1: Basic Metrics Logging

```python
from augmentation_metrics import AugmentationLogger

logger = AugmentationLogger()

# Log entries as they're processed
for entry in augmented_corpus:
    logger.log_entry(entry)

# Print summary
logger.print_summary()

# Save to file
logger.save("augmentation_metrics.json")
```

#### Example 2: Per-Epoch Tracking

```python
from augmentation_metrics import track_epoch_stats

# Track epoch metrics
stats = track_epoch_stats(
    epoch=1,
    entries=epoch_entries,
    output_dir="output/metrics"
)

# Access specific metrics
print(f"Pass rate: {stats['validation']['pass_rate']:.2%}")
print(f"Augmentation ratio: {stats['augmentation']['augmentation_ratio']:.2f}x")
```

#### Example 3: Baseline Comparison

```python
from augmentation_metrics import AugmentationLogger, compare_metrics

# Log baseline
baseline_logger = AugmentationLogger()
baseline_logger.log_batch(original_entries)
baseline_summary = baseline_logger.get_summary()

# Log augmented
augmented_logger = AugmentationLogger()
augmented_logger.log_batch(augmented_entries)
augmented_summary = augmented_logger.get_summary()

# Compare
comparison = compare_metrics(baseline_summary, augmented_summary)
print(f"Pass rate improvement: {comparison['pass_rate_improvement']:.2%}")
print(f"Augmentation ratio: {comparison['augmentation_ratio']:.2f}x")
```

### 10.7 Metrics Output Format

#### Console Output Example

```
======================================================================
TKS AUGMENTATION METRICS SUMMARY
======================================================================
Timestamp: 2025-12-14T10:30:00.000000
Duration: 45.23 seconds

AUGMENTATION STATISTICS
----------------------------------------------------------------------
  Original entries:              100
  Inverted entries:              300
  Anti-attractor entries:        100
  Total entries:                 500

  Augmentation ratio:           4.00x
  Inversion ratio:              3.00x
  Anti-attractor ratio:         1.00x

  Axes usage:
    W:    200
    N:    200
    F:    100

  Mode usage:
    soft:    300

VALIDATION STATISTICS
----------------------------------------------------------------------
  Total validated:               500
  Passed:                        475
  Failed:                         25
  Pass rate:                   95.00%

  World validity:              98.00%
  Noetic validity:             97.00%
  Operator validity:           99.00%
  Structural validity:         96.00%
  Foundation validity:         92.00%

DISTRIBUTION STATISTICS
----------------------------------------------------------------------
  World distribution:
    A:    125 (25.00%)
    B:    130 (26.00%)
    C:    120 (24.00%)
    D:    125 (25.00%)

  Noetic distribution:
    1:     45 ( 9.00%)
    2:     55 (11.00%)
    3:     50 (10.00%)
    4:     45 ( 9.00%)
    5:     55 (11.00%)
    6:     50 (10.00%)
    7:     45 ( 9.00%)
    8:     55 (11.00%)
    9:     50 (10.00%)
   10:     50 (10.00%)

  Operator distribution:
    ->:    200 (40.00%)
    +:     150 (30.00%)
    -:     100 (20.00%)
    *T:     50 (10.00%)

======================================================================
```

#### JSON Output Example

```json
{
  "timestamp": "2025-12-14T10:30:00.000000",
  "duration_seconds": 45.23,
  "augmentation": {
    "original_count": 100,
    "inversion_count": 300,
    "anti_attractor_count": 100,
    "total_count": 500,
    "augmentation_ratio": 4.0,
    "inversion_ratio": 3.0,
    "anti_attractor_ratio": 1.0,
    "axes_usage": {"W": 200, "N": 200, "F": 100},
    "mode_counts": {"soft": 300}
  },
  "validation": {
    "total": 500,
    "passed": 475,
    "failed": 25,
    "pass_rate": 0.95,
    "world_validity_rate": 0.98,
    "noetic_validity_rate": 0.97,
    "operator_validity_rate": 0.99,
    "structural_validity_rate": 0.96,
    "foundation_validity_rate": 0.92,
    "error_counts": {}
  },
  "distribution": {
    "world_counts": {"A": 125, "B": 130, "C": 120, "D": 125},
    "world_distribution": {"A": 0.25, "B": 0.26, "C": 0.24, "D": 0.25},
    "noetic_counts": {"1": 45, "2": 55, "3": 50, ...},
    "noetic_distribution": {"1": 0.09, "2": 0.11, "3": 0.10, ...},
    "operator_counts": {"->": 200, "+": 150, "-": 100, "*T": 50},
    "operator_distribution": {"->": 0.40, "+": 0.30, "-": 0.20, "*T": 0.10}
  }
}
```

### 10.8 Monitoring Best Practices

#### 1. Track at Multiple Stages

- **Pre-augmentation**: Baseline corpus metrics
- **Post-augmentation**: Validate augmentation quality
- **Per-epoch**: Monitor training data distribution
- **Post-training**: Evaluate final model performance

#### 2. Set Quality Thresholds

```python
# Check validation pass rate
if metrics['validation']['pass_rate'] < 0.90:
    print("WARNING: Pass rate below 90% threshold")

# Check distribution balance
world_dist = metrics['distribution']['world_distribution']
if max(world_dist.values()) > 0.50:
    print("WARNING: Skewed world distribution")
```

#### 3. Monitor Trends Over Time

```python
# Track metrics across epochs
epoch_metrics = []
for epoch in range(num_epochs):
    stats = track_epoch_stats(epoch, epoch_entries, output_dir)
    epoch_metrics.append(stats)

# Analyze trends
pass_rates = [m['validation']['pass_rate'] for m in epoch_metrics]
print(f"Pass rate trend: {pass_rates}")
```

#### 4. Compare Baselines

Always compare augmented results against original corpus:

```python
baseline_metrics = compute_batch_stats(original_entries)
augmented_metrics = compute_batch_stats(augmented_entries)
comparison = compare_metrics(baseline_metrics, augmented_metrics)
```

### 10.9 Troubleshooting

#### Low Pass Rates (< 80%)

Potential causes:
- Invalid TKS expression generation
- Incorrect world/noetic ranges
- Operator misuse

Solutions:
1. Check canonical validator configuration
2. Verify inversion logic
3. Review error_counts in detailed metrics
4. Test with strict validation mode

#### Skewed Distributions

Symptoms:
- One world/noetic dominates (> 50%)
- Missing worlds or noetics

Solutions:
1. Use targeted augmentation with specific axes
2. Apply balanced sampling in data loader
3. Generate more inversions on underrepresented axes
4. Filter or downsample overrepresented entries

#### High Error Counts

Symptoms:
- error_counts growing over epochs
- Specific error types recurring

Solutions:
1. Review error messages in detailed metrics
2. Check for systematic encoding failures
3. Verify foundation assignments
4. Ensure operator semantics match TKS

### 10.10 Expected Output Locations

#### Augmentation Metrics

When running `generate_augmented_data.py`:

```
<output_path>.metrics.json          # Basic metrics
<output_path>.detailed_metrics.json # Detailed logger metrics
```

Example:
```
data/pilot/augmented.jsonl.metrics.json
data/pilot/augmented.jsonl.detailed_metrics.json
```

#### Training Metrics

When running `train_with_augmented.py`:

```
<output_dir>/metrics/epoch_<NNN>_metrics.json
```

Example:
```
output/models/metrics/epoch_001_metrics.json
output/models/metrics/epoch_002_metrics.json
```

### 10.11 API Reference

#### AugmentationLogger

```python
class AugmentationLogger:
    def __init__(self):
        """Initialize logger with empty metrics."""

    def log_entry(self, entry: Dict[str, Any]) -> None:
        """Log a single entry."""

    def log_batch(self, entries: List[Dict[str, Any]]) -> None:
        """Log a batch of entries."""

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""

    def print_summary(self, detailed: bool = False) -> None:
        """Print formatted summary to console."""

    def save(self, filepath: str) -> None:
        """Save metrics to JSON file."""

    def reset(self) -> None:
        """Reset all metrics."""
```

#### Helper Functions

```python
def compute_batch_stats(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute statistics for a batch of entries."""

def track_epoch_stats(
    epoch: int,
    entries: List[Dict[str, Any]],
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Track and optionally save epoch statistics."""

def compare_metrics(
    baseline_summary: Dict[str, Any],
    augmented_summary: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare baseline and augmented metrics."""
```

### 10.12 Future Enhancements

**Phase 2+ Improvements:**

1. **Real-time Dashboards**
   - TensorBoard integration
   - Weights & Biases (wandb) support
   - Live plotting during training

2. **Advanced Analytics**
   - Semantic similarity analysis
   - Attractor signature clustering
   - Diversity metrics using embeddings

3. **Automated Quality Control**
   - Auto-filtering of low-quality augmentations
   - Dynamic parameter adjustment
   - Adaptive sampling based on distribution gaps

4. **Cross-Run Comparisons**
   - A/B testing framework
   - Historical trending
   - Regression detection

---

**Document Status:** Phase 1-3 Implementation Complete with Real Model Integration
**Next Action:** Run full training experiments with TKSLLMCorePipeline

---

## 11. Phase 3 Implementation: Real Model Training

**Status:** COMPLETE
**Date:** 2025-12-14
**Goal:** Replace DummyTKSModel with TKSLLMCorePipeline, implement full training loop with metrics

### 11.1 Implementation Summary

Phase 3 completed the training integration by:

1. **Real Model Integration:** Wired `train_with_augmented.py` to use `TKSLLMCorePipeline` from `tks_llm_core_v2.py`
2. **Full Loss Function:** Integrated `TKSLoss` multi-component loss (task + RPM + attractor + involution + spectral + cascade)
3. **PyTorch Training Loop:** Proper DataLoader, optimizer (AdamW), scheduler (CosineAnnealingLR), gradient clipping
4. **Evaluation Function:** Added `evaluate_model()` with loss, accuracy, and perplexity metrics
5. **Smoke Test:** Added `--test` flag for end-to-end pipeline verification
6. **Comprehensive CLI:** Config via command line args for all hyperparameters

### 11.2 File Changes

#### Updated: `scripts/train_with_augmented.py`

**Key Changes:**
- Replaced `DummyTKSModel` with `TKSLLMCorePipeline` (when available)
- Added `TKSTokenizer` class with TKS element and operator support
- Added `TKSAugmentedDataset` PyTorch Dataset class
- Added `TrainingMetricsLogger` for tracking loss/validation/augmentation stats
- Added `evaluate_model()` function for held-out evaluation
- Added `run_smoke_test()` for pipeline verification
- Full CLI with `argparse` for all hyperparameters

**New CLI Arguments:**
```bash
# Data arguments
--data PATH            # Required: Path to augmented JSONL
--original-data PATH   # Optional: Path to original corpus
--filter-validated     # Only use validated entries
--use-expr            # Train on TKS expressions vs stories

# Model arguments
--vocab-size INT      # Default: 1000
--hidden-dim INT      # Default: 128
--max-length INT      # Default: 256
--use-dummy           # Use fallback LSTM instead of real model

# Training arguments
--epochs INT          # Default: 10
--batch-size INT      # Default: 16
--learning-rate FLOAT # Default: 1e-4
--weight-decay FLOAT  # Default: 0.01
--max-steps INT       # Optional: limit training steps

# Output arguments
--output-dir PATH     # Default: output/models
--log-interval INT    # Default: 10 batches

# Misc arguments
--seed INT            # Default: 42
--test                # Run smoke test
--dry-run             # Single batch validation
```

### 11.3 Usage Examples

#### Smoke Test (Verify Pipeline)
```bash
cd C:\Users\wakil\downloads\everthing-tootra-tks
python scripts/train_with_augmented.py --data output/sample_augmented.jsonl --test
```

Expected output:
```
SMOKE TEST - End-to-End Training Pipeline
======================================================================
Device: cpu

[Test 1] Loading data...
  Loaded 16 entries
  [PASS] Data loading

[Test 2] Creating DataLoader...
  [PASS] DataLoader creation

[Test 3] Initializing model...
  Using TKSLLMCorePipeline (real model)
  [PASS] Model initialization

[Test 4] Forward pass...
  [PASS] Forward pass

[Test 5] Loss computation...
  Total loss: X.XXXX
  Task loss: X.XXXX
  [PASS] Loss computation

[Test 6] Backward pass...
  [PASS] Backward pass

[Test 7] Evaluation...
  [PASS] Evaluation

======================================================================
[PASS] ALL SMOKE TESTS PASSED
======================================================================
```

#### Dry Run (Pipeline Validation)
```bash
python scripts/train_with_augmented.py \
    --data output/sample_augmented.jsonl \
    --dry-run
```

#### Full Training Run
```bash
python scripts/train_with_augmented.py \
    --data output/augmented_corpus.jsonl \
    --epochs 10 \
    --batch-size 16 \
    --learning-rate 1e-4 \
    --output-dir output/models
```

#### Train with Expressions (Not Stories)
```bash
python scripts/train_with_augmented.py \
    --data output/augmented_corpus.jsonl \
    --use-expr \
    --filter-validated \
    --epochs 5
```

### 11.4 Model Architecture

The training script uses `TKSLLMCorePipeline` which implements:

```
tokens --> NoeticEmbedding --> NoeticProcessor --> FractalAttention
       --> AttractorComputation --> RPMGating --> output logits
```

**Components:**
1. **NoeticEmbeddingLayer:** Maps tokens to 40-dim noetic space (10 noetics x 4 worlds)
2. **NoeticProcessor:** Applies noetic transforms (involution pairs: 2-3, 5-6, 8-9)
3. **FractalAttentionMechanism:** Multi-scale attention across noetic dimensions
4. **AttractorComputationLayer:** Fixed-point iteration for thought convergence
5. **RPMGatingMechanism:** D/W/P (Desire/Wisdom/Power) goal-oriented filtering

### 11.5 Loss Function Components

`TKSLoss` computes weighted combination:

```
L_total = lambda_task * L_task
        + lambda_rpm * L_rpm
        + lambda_attractor * L_attractor
        + lambda_involution * L_involution
        + lambda_spectral * L_spectral
        + lambda_cascade * L_cascade
```

**Default Weights:**
```python
TKSLossConfig(
    lambda_task=1.0,          # Primary task loss (CrossEntropy)
    lambda_rpm=0.3,           # RPM D/W/P alignment
    lambda_attractor=0.2,     # Attractor convergence
    lambda_involution=0.2,    # Noetic involution constraints
    lambda_spectral=0.1,      # Spectral radius bounds
    lambda_cascade=0.1,       # World cascade flow (A->B->C->D)
)
```

### 11.6 Output Files

After training, the following files are generated:

```
output/models/
    best_model.pt              # Best model checkpoint (lowest eval loss)
    final_model.pt             # Final model checkpoint
    metrics/
        training_metrics.json  # Full training metrics
        epoch_001_metrics.json # Per-epoch detailed metrics
        epoch_002_metrics.json
        ...
```

**Metrics JSON Structure:**
```json
{
    "timestamp": "2025-12-14T...",
    "duration_seconds": 123.45,
    "total_epochs": 10,
    "total_steps": 500,
    "total_samples": 8000,
    "loss": {
        "epoch_losses": [...],
        "final_loss": 0.1234,
        "initial_loss": 0.9876
    },
    "validation": {
        "total": 1000,
        "passed": 875,
        "pass_rate": 0.875
    },
    "augmentation": {
        "distribution": {"original": 400, "inversion": 400, "anti_attractor": 200},
        "original_count": 400,
        "inversion_count": 400,
        "anti_attractor_count": 200
    },
    "eval_results": [
        {"epoch": 1, "loss": 0.8, "accuracy": 0.15, "perplexity": 2.23},
        ...
    ]
}
```

### 11.7 TKS Tokenizer

The `TKSTokenizer` class handles:

1. **Special Tokens:** `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`
2. **TKS Elements:** A1-D10 (40 element codes)
3. **TKS Operators:** +, -, +T, -T, ->, <-, *T, /T, o
4. **Character Fallback:** Lowercase/uppercase letters, digits, punctuation

**Tokenization Example:**
```python
tokenizer = TKSTokenizer(vocab_size=1000, max_length=256)

# Tokenize TKS expression
tokens = tokenizer.tokenize("B2 -> D5")
# Returns: [BOS, B2_id, space_id, ->, space_id, D5_id, EOS, PAD, ...]

# Tokenize story
tokens = tokenizer.tokenize("A spiritual teacher causes enlightenment")
# Returns: [BOS, A_id, space_id, s_id, p_id, ..., EOS, PAD, ...]
```

### 11.8 Evaluation Metrics

The `evaluate_model()` function computes:

| Metric | Description |
|--------|-------------|
| `loss` | Average batch loss |
| `accuracy` | Token-level accuracy (excluding padding) |
| `perplexity` | exp(loss), capped at 1e6 |
| `num_batches` | Number of eval batches |
| `total_tokens` | Total non-padding tokens evaluated |

### 11.9 Fallback Mode

If `TKSLLMCorePipeline` is not available (import fails), the script falls back to a simple LSTM model:

```python
model = nn.Sequential(
    nn.Embedding(vocab_size, hidden_dim),
    nn.LSTM(hidden_dim, hidden_dim, batch_first=True),
    nn.Linear(hidden_dim, vocab_size),
)
```

Use `--use-dummy` flag to force fallback mode for testing.

### 11.10 Known Limitations

1. **No Multi-GPU Support:** Single device training only
2. **No Mixed Precision:** fp32 training (can add --mixed-precision flag later)
3. **No Checkpoint Resume:** Training must start from scratch (can add --resume flag)
4. **Simple Tokenizer:** Character-level with TKS special tokens (not BPE/SentencePiece)

### 11.11 Next Steps (Phase 4)

1. **Run Full Training Experiment:**
   - Generate larger augmented corpus (1000+ entries)
   - Train for 50+ epochs
   - Compare baseline vs augmented loss curves

2. **Add Curriculum Learning:**
   - Use `CurriculumLossScheduler` to phase in loss components
   - Stage 1: Task only
   - Stage 2: + Involution
   - Stage 3: + RPM
   - Stage 4: + Attractor
   - Stage 5: Full pipeline

3. **Checkpoint Resume:**
   - Add `--resume PATH` flag to continue training

4. **Distributed Training:**
   - Add DataParallel / DistributedDataParallel support

5. **Better Tokenization:**
   - Integrate BPE or SentencePiece for stories
   - Keep special handling for TKS elements/operators

---

## 12. Metrics Visualization Dashboard

**Status:** Complete
**Date:** 2025-12-14
**Location:** `scripts/plot_metrics.py`, `scripts/augmentation_metrics.py`

### 12.1 Overview

The metrics visualization system provides comprehensive plotting capabilities for training and augmentation metrics. It generates publication-quality PNG plots from JSON or CSV metrics files, supporting both individual plots and combined dashboards.

### 12.2 Available Plot Types

| Plot Type | Description | Output File |
|-----------|-------------|-------------|
| `loss` | Training loss curve over epochs | `loss_curve.png` |
| `validation` | Validator pass rates over time (overall, world, noetic, operator) | `validation_rates.png` |
| `distribution` | Augmentation type distribution (pie chart) | `augmentation_distribution.png` |
| `counts-bar` | Augmentation counts by type over epochs (grouped bar chart) | `augmentation_counts_bar.png` |
| `world-noetic` | World (A/B/C/D) and noetic (1-10) distribution | `world_noetic_distribution.png` |
| `ratios` | Augmentation ratios over time | `augmentation_ratios.png` |
| `dashboard` | Combined 2x2 dashboard with loss, validation, augmentation, summary | `combined_dashboard.png` |
| `all` | Generate all available plots | (all files above) |

### 12.3 Running the Plotting Script

**Basic Usage:**
```bash
# Generate all plots from JSON metrics
python scripts/plot_metrics.py \
  --input output/metrics.json \
  --output-dir output/plots \
  --plot-type all

# Generate specific plot from CSV metrics
python scripts/plot_metrics.py \
  --input output/metrics.csv \
  --output-dir output/plots \
  --plot-type loss

# Generate combined dashboard
python scripts/plot_metrics.py \
  --input output/training_metrics.json \
  --output-dir output/plots \
  --plot-type dashboard

# Generate plots with custom prefix
python scripts/plot_metrics.py \
  --input output/metrics.json \
  --output-dir output/plots \
  --plot-type all \
  --prefix experiment_01
```

**Command-Line Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `--input` | Yes | Path to metrics file (JSON or CSV) |
| `--output-dir` | Yes | Directory to save PNG outputs |
| `--plot-type` | No | Plot type to generate (default: `all`) |
| `--prefix` | No | Prefix for output filenames |

### 12.4 Input File Formats

**JSON Format (recommended):**
```json
[
  {
    "epoch": 1,
    "loss": 0.85,
    "validation": {
      "pass_rate": 0.75,
      "world_validity_rate": 0.92,
      "noetic_validity_rate": 0.88,
      "operator_validity_rate": 0.95
    },
    "augmentation": {
      "original_count": 100,
      "inversion_count": 200,
      "anti_attractor_count": 100,
      "augmentation_ratio": 3.0
    }
  },
  {
    "epoch": 2,
    "loss": 0.72,
    ...
  }
]
```

**CSV Format:**
```csv
timestamp,epoch,loss,pass_rate,world_validity_rate,noetic_validity_rate,original_count,inversion_count,anti_attractor_count
2025-12-14T10:00:00,1,0.85,0.75,0.92,0.88,100,200,100
2025-12-14T11:00:00,2,0.72,0.82,0.94,0.91,100,200,100
```

### 12.5 Interpreting the Plots

#### Loss Curve (`loss_curve.png`)
- **X-axis:** Epoch number
- **Y-axis:** Loss value
- **Interpretation:** Monitor for:
  - Decreasing trend indicates model is learning
  - Plateaus suggest learning rate adjustment needed
  - Spikes may indicate batch anomalies or overfitting

#### Validation Rates (`validation_rates.png`)
- **Lines:**
  - Overall Pass Rate (solid blue): Percentage of outputs that pass full canonical validation
  - World Validity (dashed magenta): Outputs with valid world letters (A/B/C/D)
  - Noetic Validity (dashed orange): Outputs with valid noetic indices (1-10)
  - Operator Validity (dashed red): Outputs using only allowed operators
- **Target:** Overall pass rate should reach 90%+ for production-ready model
- **Interpretation:**
  - Component rates exceeding overall rate indicate structural issues
  - Diverging lines suggest specific validation bottlenecks

#### Augmentation Counts (`augmentation_counts_bar.png`)
- **Bars:**
  - Original (magenta): Unmodified training scenarios
  - Inversion (orange): Scenarios transformed via axis inversion
  - Anti-Attractor (red): Counter-scenarios from attractor synthesis
- **Interpretation:**
  - Balanced bars indicate diverse augmentation
  - Target ratio: 2-5x augmented per original

#### Combined Dashboard (`combined_dashboard.png`)
- **Top-left:** Loss vs. Epoch
- **Top-right:** Validator Pass Rate Over Time
- **Bottom-left:** Augmentation Counts (stacked area)
- **Bottom-right:** Summary Statistics Table

The dashboard provides a single-view overview for training monitoring and reporting.

### 12.6 Integration with Training Pipeline

**During Training:**
```python
from scripts.augmentation_metrics import AugmentationLogger

# Initialize logger
logger = AugmentationLogger()

# Log entries during training
for batch in dataloader:
    # ... training step ...
    logger.log_batch(batch_entries)

# Save metrics after each epoch
logger.save_to_json("output/metrics.json", append=True)
logger.reset()  # Reset for next epoch
```

**After Training:**
```bash
# Generate dashboard from accumulated metrics
python scripts/plot_metrics.py \
  --input output/metrics.json \
  --output-dir output/plots \
  --plot-type dashboard
```

### 12.7 Augmentation Metrics Logger

**Location:** `scripts/augmentation_metrics.py`

The `AugmentationLogger` class provides lightweight in-memory metrics accumulation with export capabilities.

**Usage:**
```python
from scripts.augmentation_metrics import AugmentationLogger

logger = AugmentationLogger()

# Log individual entry
logger.log_entry({
    "aug_type": "inversion",
    "validator_pass": True,
    "expr_elements": ["B2", "D5"],
    "expr_ops": ["->"],
    "axes": ["W", "N"],
    "mode": "soft"
})

# Log batch
logger.log_batch(entries)

# Get summary statistics
summary = logger.get_summary()

# Print formatted summary
logger.print_summary(detailed=True)

# Save to JSON
logger.save("output/metrics.json")

# Save to CSV (for spreadsheet analysis)
logger.save_to_csv("output/metrics.csv", append=True)
```

**Tracked Metrics:**
- **Augmentation:** original/inversion/anti-attractor counts, ratios, axes usage, mode counts
- **Validation:** total/passed/failed, pass rate, component validity rates, error counts
- **Distribution:** world/noetic/operator/foundation counts and percentages, polarity balance

### 12.8 Troubleshooting

**No loss data found:**
- Ensure metrics file contains `loss`, `avg_loss`, or `epoch_loss` field
- Alternatively, provide `pass_rate` which will be converted to pseudo-loss (1 - pass_rate)

**No augmentation data found:**
- Ensure metrics contain `augmentation` object or flat fields:
  - `original_count`, `inversion_count`, `anti_attractor_count`

**matplotlib not installed:**
```bash
pip install matplotlib
```

### 12.9 Dependencies

- **Required:** `matplotlib` (for plotting)
- **Optional:** `numpy` (auto-imported by matplotlib)

Install with:
```bash
pip install matplotlib
```

---

## 13. Phase 3 Implementation: Real Model Training Integration

**Status:** Complete
**Date:** 2025-12-14
**Goal:** Complete training integration using the real TKSLLMCorePipeline model with proper training, evaluation, and smoke testing.

### 13.1 Overview

Phase 3 establishes the complete end-to-end training pipeline that integrates the real TKSLLMCorePipeline model (from `tks_llm_core_v2.py`) with the augmented data pipeline. This phase transitions from the dummy/stub implementations of Phase 2 to actual model training with proper loss computation, evaluation, and checkpoint management.

### 13.2 Key Components Implemented

#### 13.2.1 Real Model Integration

**TKSLLMCorePipeline** is now fully wired into the training loop:

```python
from tks_llm_core_v2 import TKSLLMCorePipeline, TOTAL_DIM
from training.losses import TKSLoss, TKSLossConfig

# Initialize real model
model = TKSLLMCorePipeline(
    vocab_size=tokenizer.actual_vocab_size,
    hidden_dim=128,
    noetic_dim=TOTAL_DIM,  # 40 = 10 noetics x 4 worlds
    num_scales=3,
    max_attractor_iter=10,
    contraction_factor=0.5,
).to(device)

# Initialize TKS-specific loss
loss_config = TKSLossConfig(
    lambda_task=1.0,
    lambda_rpm=0.3,
    lambda_attractor=0.2,
    lambda_involution=0.2,
    lambda_spectral=0.1,
    lambda_cascade=0.1,
)
loss_fn = TKSLoss(loss_config)
```

**Pipeline Architecture:**
1. NoeticEmbeddingLayer: tokens -> 40-dim noetic space
2. NoeticProcessor: apply noetic transforms
3. FractalAttentionMechanism: multi-scale attention
4. AttractorComputationLayer: converge to thought attractor
5. RPMGatingMechanism: D/W/P goal-oriented filtering

#### 13.2.2 Enhanced Training Script

**Location:** `scripts/train_with_augmented.py`

**CLI Configuration:**
```bash
python scripts/train_with_augmented.py \
    --data output/sample_augmented.jsonl \
    --epochs 10 \
    --batch-size 16 \
    --learning-rate 1e-4 \
    --hidden-dim 128 \
    --max-length 256 \
    --output-dir output/models
```

**Key Features:**
- Real model training with TKSLLMCorePipeline
- Multi-component TKSLoss (task + RPM + attractor + involution + spectral + cascade)
- Cosine annealing learning rate scheduler
- Gradient clipping for stable training
- Train/eval split with held-out evaluation
- Automatic checkpoint saving (best + final models)
- Comprehensive metrics logging to JSON

**Training Configuration Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--data` | Required | Path to augmented JSONL |
| `--original-data` | None | Optional original corpus |
| `--epochs` | 10 | Number of epochs |
| `--batch-size` | 16 | Training batch size |
| `--learning-rate` | 1e-4 | Initial learning rate |
| `--weight-decay` | 0.01 | AdamW weight decay |
| `--hidden-dim` | 128 | Model hidden dimension |
| `--max-length` | 256 | Maximum sequence length |
| `--filter-validated` | False | Only use validated entries |
| `--use-expr` | False | Train on expressions vs stories |
| `--use-dummy` | False | Use fallback model |
| `--dry-run` | False | Single batch validation |
| `--test` | False | Run smoke test |

#### 13.2.3 Evaluation Script

**Location:** `scripts/evaluate_model.py`

Provides comprehensive model evaluation on held-out test data:

```bash
python scripts/evaluate_model.py \
    --checkpoint output/models/best_model.pt \
    --data output/sample_augmented.jsonl \
    --test-ratio 0.2 \
    --output eval_report.json
```

**Evaluation Metrics:**
- **Model Performance:** Loss, accuracy, perplexity
- **Per-Component Losses:** task, RPM, attractor, involution, spectral, cascade
- **Per-Augmentation-Type Accuracy:** original, inversion, anti_attractor breakdown
- **Canonical Validity:** World, noetic, operator, full validity rates

**CanonicalValidator Class:**
```python
from scripts.evaluate_model import CanonicalValidator

validator = CanonicalValidator()

# Validate single element
result = validator.validate_element("B5")
# -> {'valid': True, 'world_valid': True, 'noetic_valid': True, 'errors': []}

# Validate full expression
result = validator.validate_expression(
    expr_elements=["A1", "B5", "D2"],
    expr_ops=["->", "+T"]
)
# -> {'valid': True, 'world_validity_rate': 1.0, ...}
```

#### 13.2.4 Smoke Test Script

**Location:** `scripts/smoke_test_training.py`

End-to-end pipeline verification with 11 comprehensive tests:

```bash
python scripts/smoke_test_training.py --verbose
```

**Tests Performed:**
1. Data file exists and contains entries
2. Tokenizer initialization and round-trip
3. Dataset loading with proper structure
4. DataLoader creation with batching
5. Model initialization (real or fallback)
6. Forward pass through all layers
7. Loss computation (all components)
8. Backward pass and optimizer step
9. Evaluation function execution
10. Checkpoint save/load cycle
11. Metrics logger functionality

**Exit Codes:**
- `0`: All tests passed
- `1`: One or more tests failed

### 13.3 Metrics Tracked

#### Training Metrics

The `TrainingMetricsLogger` class tracks:

```python
from scripts.train_with_augmented import TrainingMetricsLogger

metrics = TrainingMetricsLogger(output_dir=Path("output/metrics"))

# Log training step
metrics.log_step(epoch=1, step=100, loss=0.5, batch_size=16)

# Log epoch summary
metrics.log_epoch(epoch=1, avg_loss=0.45, entries=epoch_entries)

# Log evaluation results
metrics.log_eval(epoch=1, result={'loss': 0.4, 'accuracy': 0.75})

# Save to JSON
metrics.save("training_metrics.json")

# Print summary
metrics.print_summary()
```

**Output JSON Structure:**
```json
{
  "timestamp": "2025-12-14T...",
  "duration_seconds": 120.5,
  "total_epochs": 10,
  "total_steps": 500,
  "total_samples": 8000,
  "loss": {
    "epoch_losses": [...],
    "final_loss": 0.25,
    "initial_loss": 2.5
  },
  "validation": {
    "total": 100,
    "passed": 95,
    "pass_rate": 0.95
  },
  "augmentation": {
    "distribution": {"original": 50, "inversion": 30, "anti_attractor": 20},
    "original_count": 50,
    "inversion_count": 30,
    "anti_attractor_count": 20
  },
  "eval_results": [...]
}
```

#### Evaluation Metrics

The evaluation script computes:

| Metric | Description | Target |
|--------|-------------|--------|
| Loss | Average cross-entropy loss | < 1.0 |
| Accuracy | Token-level accuracy | > 0.5 |
| Perplexity | exp(loss) | < 10 |
| World Validity | % valid world letters | > 95% |
| Noetic Validity | % valid noetic indices | > 95% |
| Operator Validity | % valid operators | > 98% |
| Full Validity | % fully valid expressions | > 90% |

### 13.4 Usage Examples

#### Full Training Run

```bash
# Generate augmented data (if not already done)
python scripts/generate_augmented_data.py \
    --input data/original.jsonl \
    --output output/augmented.jsonl \
    --axes W N F \
    --use-anti-attractor

# Run smoke test first
python scripts/train_with_augmented.py \
    --data output/augmented.jsonl \
    --test

# Full training run
python scripts/train_with_augmented.py \
    --data output/augmented.jsonl \
    --epochs 20 \
    --batch-size 32 \
    --learning-rate 5e-5 \
    --output-dir output/models

# Evaluate trained model
python scripts/evaluate_model.py \
    --checkpoint output/models/best_model.pt \
    --data output/augmented.jsonl \
    --output output/eval_report.json
```

#### Quick Pipeline Validation

```bash
# Smoke test with existing sample data
python scripts/smoke_test_training.py

# Dry run (single batch)
python scripts/train_with_augmented.py \
    --data output/sample_augmented.jsonl \
    --dry-run

# Evaluate with dummy model (for testing)
python scripts/evaluate_model.py \
    --checkpoint output/models/final_model.pt \
    --data output/sample_augmented.jsonl \
    --use-dummy
```

### 13.5 Integration Architecture

```
Augmented JSONL (from Phase 1.5)
    |
    v
TKSAugmentedDataset (PyTorch Dataset)
    |
    v
DataLoader (batched, shuffled)
    |
    v
TKSLLMCorePipeline (5-layer model)
    |-- NoeticEmbedding
    |-- NoeticProcessor
    |-- FractalAttention
    |-- AttractorComputation
    |-- RPMGating
    |
    v
TKSLoss (multi-component)
    |-- TaskLoss (cross-entropy)
    |-- RPMLoss (D/W/P alignment)
    |-- AttractorLoss (convergence)
    |-- InvolutionLoss (pair composition)
    |-- SpectralLoss (eigenvalue)
    |-- CascadeLoss (world flow)
    |
    v
Optimizer (AdamW) + Scheduler (CosineAnnealing)
    |
    v
Checkpoints + Metrics (JSON export)
```

### 13.6 Canonical Guardrails Enforced

The implementation strictly adheres to TKS canonical constraints:

| Guardrail | Implementation | Validation |
|-----------|----------------|------------|
| Worlds A/B/C/D only | `WORLD_CODES = {'A', 'B', 'C', 'D'}` | CanonicalValidator |
| Noetics 1-10 | `NOETIC_INDICES = set(range(1, 11))` | CanonicalValidator |
| Involution pairs | `(2,3), (5,6), (8,9)` | TKSLoss.InvolutionLoss |
| Self-duals | `{1, 4, 7, 10}` | TKSLoss |
| Foundations 1-7 | `NUM_FOUNDATIONS = 7` | CanonicalValidator |
| Allowed operators | `{+, -, +T, -T, ->, <-, *T, /T, o}` | TKSTokenizer |

### 13.7 Files Created/Modified

**New Files:**
- `scripts/evaluate_model.py` - Comprehensive evaluation script
- `scripts/smoke_test_training.py` - End-to-end smoke test

**Modified Files:**
- `scripts/train_with_augmented.py` - Enhanced with real model integration
- `docs/TRAINING_INTEGRATION_PLAN.md` - Added Phase 3 documentation

### 13.8 Phase 3 Success Criteria

**Deliverables:**
- [x] TKSLLMCorePipeline correctly wired to training loop
- [x] Multi-component TKSLoss integrated
- [x] CLI configuration for epochs, batch size, learning rate, data paths
- [x] Metrics logging: loss curve, validator pass-rate, augmentation usage
- [x] Evaluation script for held-out accuracy
- [x] Smoke test for end-to-end validation
- [x] Documentation updated with Phase 3 section

**Validation Checkpoints:**
- [x] Smoke test passes all 11 tests
- [x] Forward pass produces valid logits shape
- [x] Loss computation returns finite, positive values
- [x] Backward pass computes gradients
- [x] Checkpoint save/load preserves model state
- [x] Evaluation metrics computed correctly

### 13.9 Next Steps (Phase 4)

**Short-term:**
1. Scale training to larger augmented corpus (1000+ entries)
2. Implement early stopping based on validation loss
3. Add learning rate warmup schedule
4. Integrate with wandb/tensorboard for live monitoring

**Medium-term:**
1. Implement contrastive learning for inversion pairs
2. Add TKS-aware beam search for generation
3. Multi-task training (stories + expressions jointly)
4. Foundation-aware attention mechanisms

**Long-term:**
1. Interpretability tools for TKS reasoning traces
2. Downstream task fine-tuning pipelines
3. Model distillation for deployment
4. Continuous training with new augmented data

---

## 14. Phase 4 Implementation: Data Quality & Sanitization

**Implementation Date:** 2025-12-14
**Agent:** TKS Agent 2
**Status:** Complete

### 14.1 Overview

Phase 4 introduces a comprehensive data sanitization tool to ensure quality and canonical compliance of augmented training data. The sanitizer detects and handles:

- Duplicate entries (by ID and content hash)
- Invalid operators, worlds, noetics, and foundations
- Missing required fields
- Structural inconsistencies

This optional pipeline step provides quality assurance before training, enabling early detection of data issues and improving model reliability.

### 14.2 Implementation Details

#### 14.2.1 Sanitizer Script

**File:** `scripts/sanitize_augmented.py`

**Core Features:**
- Scans JSONL files for data quality issues
- Reports issues with severity levels (error/warning/info)
- Supports multiple operation modes:
  - `--flag-only`: Report without modifying data
  - `--drop-invalid`: Remove invalid entries
  - `--report FILE`: Export detailed JSON report
- Enforces canonical constraints (A/B/C/D worlds, 1-10 noetics, allowed operators)
- Detects duplicate IDs and content hashes
- Validates structural consistency (operator/element counts)

**Usage Examples:**

```bash
# Scan and report only
python scripts/sanitize_augmented.py \
    --input data/pilot/augmented.jsonl \
    --flag-only

# Clean and save
python scripts/sanitize_augmented.py \
    --input data/pilot/augmented.jsonl \
    --output data/pilot/augmented_clean.jsonl \
    --drop-invalid

# Generate detailed report
python scripts/sanitize_augmented.py \
    --input data/pilot/augmented.jsonl \
    --output data/pilot/augmented_clean.jsonl \
    --drop-invalid \
    --report data/pilot/sanitization_report.json
```

#### 14.2.2 Test Suite

**File:** `tests/test_sanitize_augmented.py`

**Test Coverage:**
- Valid entry validation (pass-through)
- Invalid operator detection
- Invalid world detection (non-A/B/C/D)
- Invalid noetic detection (not 1-10)
- Missing required fields detection
- Structural error detection (operator/element mismatch)
- Duplicate ID detection
- Content hash duplicate detection
- Multiple issues per entry
- Full pipeline integration tests

**Running Tests:**

```bash
pytest tests/test_sanitize_augmented.py -v
```

#### 14.2.3 Canonical Constraints Enforced

**Worlds:** Only A, B, C, D
- A: Spiritual
- B: Mental
- C: Emotional
- D: Physical

**Noetics:** Only 1-10
- Involution pairs: 2↔3, 5↔6, 8↔9
- Self-duals: 1, 4, 7, 10

**Foundations:** Only 1-7
- 1: Unity, 2: Wisdom, 3: Life, 4: Companionship
- 5: Power, 6: Material, 7: Lust

**Operators:** Only {+, -, +T, -T, ->, <-, *T, /T, o}

**Structure:** len(ops) = len(elements) - 1

#### 14.2.4 Report Format

The sanitizer generates comprehensive reports:

```json
{
  "summary": {
    "total_entries": 500,
    "clean_entries": 485,
    "duplicate_entries": 5,
    "invalid_operators": 3,
    "invalid_worlds": 2,
    "invalid_noetics": 4,
    "missing_fields": 1,
    "structural_errors": 0,
    "pass_rate": 0.97
  },
  "issues": [
    {
      "entry_id": "entry_042",
      "issue_type": "invalid_operator",
      "description": "Invalid operator '**' at position 0",
      "severity": "error",
      "field": "expr_ops"
    }
  ],
  "duplicates": {
    "by_id": {"entry_001": 2},
    "by_hash": {"hash123": ["entry_010", "entry_011"]}
  }
}
```

### 14.3 Integration into Pipeline

The sanitizer integrates as an optional post-processing step:

```bash
# Step 1: Generate augmented data
python scripts/generate_augmented_data.py \
    --input data/pilot/stories.jsonl \
    --output data/pilot/augmented.jsonl \
    --axes W N \
    --use-anti-attractor

# Step 2: Sanitize and clean (PHASE 4)
python scripts/sanitize_augmented.py \
    --input data/pilot/augmented.jsonl \
    --output data/pilot/augmented_clean.jsonl \
    --drop-invalid \
    --report data/pilot/quality_report.json

# Step 3: Use cleaned data for training
python scripts/train_with_augmented.py \
    --data data/pilot/augmented_clean.jsonl \
    --model-name tks-pilot
```

### 14.4 Quality Metrics

**Key Metrics Tracked:**
- **Pass rate:** Percentage passing all validations (target: ≥95%)
- **Duplicate rate:** Percentage with duplicate IDs/content (target: ≤1%)
- **Validation breakdown:** Per-validator pass rates
- **Issue distribution:** Count of each issue type

**Production Thresholds:**
- Pass rate: ≥ 95%
- Duplicate rate: ≤ 1%
- Missing fields: 0%

### 14.5 Files Created

**New Files:**
- `scripts/sanitize_augmented.py` - Main sanitization script (460 lines)
- `tests/test_sanitize_augmented.py` - Comprehensive test suite (570 lines)

**Modified Files:**
- `scripts/AUGMENTATION_PIPELINE_SPEC.md` - Added Phase 4 documentation
- `docs/TRAINING_INTEGRATION_PLAN.md` - Added Phase 4 section (this document)

### 14.6 Phase 4 Success Criteria

**Deliverables:**
- [x] Sanitizer script with duplicate detection
- [x] Canonical constraint validation (worlds, noetics, operators, foundations)
- [x] Multiple operation modes (flag-only, drop-invalid, report)
- [x] Comprehensive test suite (15+ test cases)
- [x] Documentation in AUGMENTATION_PIPELINE_SPEC.md
- [x] Documentation in TRAINING_INTEGRATION_PLAN.md

**Validation Checkpoints:**
- [x] Detects invalid operators
- [x] Detects invalid worlds (non-A/B/C/D)
- [x] Detects invalid noetics (not 1-10)
- [x] Detects duplicate IDs
- [x] Detects content hash duplicates
- [x] Detects missing required fields
- [x] Detects structural errors
- [x] Generates summary reports
- [x] Exports detailed JSON reports
- [x] All tests pass

### 14.7 Usage Recommendations

**For Development/Testing:**
```bash
# Use --flag-only to inspect without modifying
python scripts/sanitize_augmented.py \
    --input data/test/augmented.jsonl \
    --flag-only
```

**For Production Training:**
```bash
# Use --drop-invalid to ensure clean data
python scripts/sanitize_augmented.py \
    --input data/prod/augmented.jsonl \
    --output data/prod/augmented_clean.jsonl \
    --drop-invalid \
    --report data/prod/quality_report.json
```

**For Quality Analysis:**
```bash
# Generate report for analysis
python scripts/sanitize_augmented.py \
    --input data/analysis/augmented.jsonl \
    --report data/analysis/quality_metrics.json \
    --flag-only
```

### 14.8 Next Steps (Phase 5)

**Potential Enhancements:**
1. Add automatic deduplication by content hash
2. Implement severity-based filtering (remove errors, keep warnings)
3. Add statistics on world/noetic/foundation distributions
4. Integrate with augmentation pipeline for inline validation
5. Add performance metrics (entries processed per second)
6. Implement parallel processing for large datasets
7. Add schema validation for JSONL structure

---

**Document Status:** Phase 4 Implementation Complete
**Next Action:** Integrate sanitizer into production pipeline and scale training
