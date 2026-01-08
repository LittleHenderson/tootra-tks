# Phase 1 Training Data Specification

## Overview

Phase 1 training requires a mix of data types to enable both TKS mastery and natural language understanding. The goal is to create a model that can:

1. **Understand TKS notation** (current capability)
2. **Explain TKS in natural language** (new)
3. **Generate TKS from natural language descriptions** (new)
4. **Reason about TKS concepts** (new)
5. **Engage in dialogue about TKS** (new)

## Data Mix Ratios

| Data Type | Ratio | Purpose |
|-----------|-------|---------|
| `tks_notation` | 40% | Pure TKS (current data) - maintain syntax mastery |
| `tks_explanation` | 30% | TKS with NL explanations - bridge understanding |
| `nl_with_tks` | 20% | NL text mentioning TKS - contextual learning |
| `general_nl` | 10% | General English - language model bootstrap |

## Data Formats

### 1. TKS Notation (Current)

```json
{
  "input": "Goal: A0.1 Divine Blueprint",
  "output": "(A0:B2)",
  "type": "tks_notation",
  "subtype": "goal"
}
```

Subtypes: `mapping`, `goal`, `operator`, `rpm`, `governance`, `regulator`

### 2. TKS Explanation

```json
{
  "input": "Explain the TKS expression (A0:B2)",
  "output": "The expression (A0:B2) represents a pathway from A0 (Divine Blueprint in the Spiritual world) to B2 (Positive Belief in the Cognitive world). This shows how divine intention manifests through mental reception. The colon ':' indicates a direct connection or pathway between these two sephirots.",
  "type": "tks_explanation",
  "tks_expr": "(A0:B2)"
}
```

### 3. TKS Reasoning Chain

```json
{
  "input": "How would you express 'transforming negative thoughts into physical health' in TKS?",
  "output": "Let me break this down:\n\n1. Negative thoughts = B3 (Cognitive world, negative polarity)\n2. Physical health = D2 (Physical world, positive state)\n3. Transformation = *T operator (multiplicative transform)\n\nThe expression would be: (B3^2) *T (D2^2)\n\nThis shows the cognitive pattern (^2 = interaction level) being transformed into physical manifestation.",
  "type": "tks_reasoning",
  "final_expr": "(B3^2) *T (D2^2)"
}
```

### 4. TKS Dialogue

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What does the ^ symbol mean in TKS?"
    },
    {
      "role": "assistant",
      "content": "The ^ symbol in TKS indicates the noetic level, which represents the depth or intensity of engagement:\n\n- ^0 = Pure mapping (static reference)\n- ^1 = Awareness level\n- ^2 = Interaction level\n- ^3 = Integration level\n- ^4 = Expression level\n- ^5 = Transformation level\n\nFor example, (A0^0) is a simple reference to Divine Blueprint, while (A0^5) indicates active transformation at that sephirot."
    },
    {
      "role": "user",
      "content": "So (B2^3) would be...?"
    },
    {
      "role": "assistant",
      "content": "(B2^3) represents Positive Belief (B2) at integration level (^3). This means the positive belief is not just present but being actively integrated into the cognitive framework. It's deeper than mere interaction (^2) but not yet at the expression level (^4)."
    }
  ],
  "type": "tks_dialogue"
}
```

### 5. NL with Embedded TKS

```json
{
  "text": "The Tree of Knowledge System (TKS) provides a symbolic framework for representing transformations. For instance, when we want to express emotional healing, we might write (C3^4) +T (D2^4), showing how emotional patterns (C3) transform into physical well-being (D2). The +T operator indicates forward transformation, while the ^4 noetic level shows this is happening at the expression level.",
  "type": "nl_with_tks"
}
```

### 6. TKS Definition

```json
{
  "input": "Define: A0",
  "output": "A0 (Divine Blueprint) is the first sephirot in the Spiritual world (A). It represents the original divine intention or cosmic pattern. In TKS expressions, A0 often appears as a source or origin point, such as (A0:B2) showing divine intention flowing into cognitive belief.",
  "type": "tks_definition",
  "sephirot": "A0"
}
```

### 7. TKS Comparison

```json
{
  "input": "What's the difference between (A0:B2) and (A0^3) + (B2^3)?",
  "output": "These expressions represent different relationships:\n\n1. (A0:B2) - A pathway/connection. Shows direct flow from A0 to B2.\n\n2. (A0^3) + (B2^3) - A combination. Shows both sephirots at integration level being added together.\n\nThe first is about connection/flow, the second is about combination/synthesis. Use ':' for pathways, '+' for additions.",
  "type": "tks_comparison"
}
```

### 8. World Context

```json
{
  "input": "Explain the four worlds in TKS",
  "output": "TKS organizes reality into four worlds, each represented by a letter:\n\n**A - Spiritual World (Atziluth)**\nThe realm of divine intention, pure consciousness, and spiritual patterns. Sephirots A0-A9.\n\n**B - Cognitive World (Briah)**\nThe realm of thought, belief, and mental structures. Sephirots B0-B9.\n\n**C - Emotional World (Yetzirah)**\nThe realm of feelings, desires, and emotional patterns. Sephirots C0-C9.\n\n**D - Physical World (Assiah)**\nThe realm of matter, body, and physical manifestation. Sephirots D0-D9.\n\nTransformations between worlds show how energy/information moves across these planes.",
  "type": "tks_world_context"
}
```

## Data Generation Guidelines

### Quality Requirements

1. **Accuracy**: All TKS expressions must be syntactically valid
2. **Consistency**: Explanations must match the canonical TKS documentation
3. **Diversity**: Cover all sephirots, operators, and noetic levels
4. **Naturalness**: NL should sound fluent, not templated

### Coverage Requirements

For Phase 1, ensure coverage of:

- [ ] All 40 sephirots (A0-A9, B0-B9, C0-C9, D0-D9)
- [ ] All operators (+, -, +T, -T, *T, /T, ->, <-)
- [ ] All noetic levels (^0 through ^9)
- [ ] Pathway expressions (X:Y)
- [ ] Multi-term expressions ((X) + (Y) - (Z))
- [ ] Nested expressions
- [ ] Cross-world transformations

### Augmentation Strategies

1. **Paraphrase**: Multiple ways to ask the same question
2. **Difficulty progression**: Simple → Complex expressions
3. **Error examples**: Include "What's wrong with (A0B2)?" type questions
4. **Contrastive pairs**: Similar expressions with different meanings

## File Format

Training data should be in JSONL format:

```
{"input": "...", "output": "...", "type": "...", ...}
{"input": "...", "output": "...", "type": "...", ...}
```

## Recommended Dataset Sizes

| Phase | Total Samples | TKS Notation | Explanations | Dialogues | General |
|-------|---------------|--------------|--------------|-----------|---------|
| 1a (10M model) | 100K | 40K | 30K | 20K | 10K |
| 1b (25M model) | 500K | 200K | 150K | 100K | 50K |
| 1c (50M model) | 1M | 400K | 300K | 200K | 100K |

## Validation

Before training, validate:

1. All TKS expressions parse correctly
2. No data leakage between train/val/test
3. Type distribution matches target ratios
4. Tokenizer handles all examples without UNK overflow
