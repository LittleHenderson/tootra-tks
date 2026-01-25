# TKS 4-Operation Block V2: Technical Design Review

## Executive Summary

**Verdict: The design is architecturally sound and ready for experimental validation.**

ChatGPT's V2 block successfully addresses all critical architectural issues from V1. The implementation is clean, well-instrumented, and properly set up for the experimental ladder (A vs B, then B vs C).

## ✅ What's Fixed in V2

### 1. Separate Contexts for Inhibition

**V1 Problem:**
```python
c = self.attn(x)  # Single shared context
term_add = w_add * c
term_sub = w_sub * c  # Can't learn orthogonal patterns!
```

**V2 Solution:**
```python
# tks_4op_block_v2.py:125-133
c_add = self.attn_add(z_add)  # [B, T, D]
c_inh = self.attn_inh(z_inh)  # [B, T, D] - SEPARATE

term_add = w_add * c_add
term_sub = w_sub * alpha_sub * tanh(c_inh)
```

**Why this matters:**
- Accumulation attention can learn "what to build up"
- Inhibition attention can learn "what to subtract/negate"
- These are genuinely orthogonal operations, not just weighted versions of the same context
- This is the **core innovation** that makes the subtractive hypothesis testable

**Score: 10/10** - Perfect implementation of the key idea.

---

### 2. Independent Gates (No Competition)

**V1 Problem:**
```python
gates = softmax([g_add, g_sub, g_mul, g_div])  # Sum to 1
# If w_add = 0.7, then w_sub can be at most 0.3
# Operations compete and suppress each other
```

**V2 Solution:**
```python
# tks_4op_block_v2.py:135-138
w_add = sigmoid(g_add)  # [B, T, D] ∈ [0, 1]
w_sub = sigmoid(g_sub)  # [B, T, D] ∈ [0, 1]
w_mul = sigmoid(g_mul)  # Independent!
w_div = sigmoid(g_div)
```

**Why this matters:**
- All gates can be active simultaneously
- No artificial competition between operations
- Each gate learns when its operation is useful, not when to suppress others
- Reduces collapse risk (but doesn't eliminate it - see below)

**Score: 9/10** - Correct solution, though auxiliary loss may be needed if collapse occurs.

---

### 3. Interpretable Inhibition Attention

**V1 Problem:**
- No way to see what inhibition attention is doing
- Can't validate whether it learns negation scope

**V2 Solution:**
```python
# tks_4op_block_v2.py:131-135
if return_inhib_attn:
    c_inh, inh_attn = self.attn_inh(z_inh, return_attn=True)
    # inh_attn: [B, H, T, T] - full attention matrix
```

**Mechanistic probe implemented:**
```python
# mechanistic_probe.py:85-115
def _analyze_single_example(...):
    # For each "not" token:
    #   - Measure attention to predicates
    #   - Measure attention to other tokens
    #   - Compute ratio

    # If ratio > 2-3x → meaningful predicate focusing
```

**Why this matters:**
- This is the **smoking gun test**
- If inhibition attention learns to focus on negated predicates significantly more than baseline, you have mechanistic evidence for the TKS hypothesis
- Visualization enables qualitative understanding of what the model learned

**Score: 10/10** - Complete implementation with visualization tools.

---

### 4. Stability Mechanisms

**Implemented safeguards:**

```python
# tks_4op_block_v2.py:143
term_sub = w_sub * self.alpha_sub * tanh(c_inh)
#                  ^^^^^^^^^^^^^^^  ^^^^^^^^^^
#                  Scaling factor   Bounds to [-1,1]
```

**Why this matters:**
- `tanh(c_inh)` prevents unbounded subtraction
- `alpha_sub=0.1` scales down to prevent dominating residual
- `w_sub` gate provides learned control
- Result: `term_sub ∈ [0, 0.1]` per dimension (assuming w_sub ∈ [0,1])

**Additional stability:**
```python
# DIV operation uses softplus + epsilon
denom = eps + softplus(div_denom_proj(c_add))  # Always > eps
out = x / (eps + w_div * denom)  # Never divide by zero
```

**Score: 9/10** - Well-designed stability mechanisms.

---

### 5. Comprehensive Instrumentation

**Gate statistics (detached):**
```python
# tks_4op_block_v2.py:161-169
stats = {
    "w_add": w_add.mean().item(),
    "w_sub": w_sub.mean().item(),
    "w_mul": w_mul.mean().item(),
    "w_div": w_div.mean().item(),
    "c_add_norm": c_add.norm(dim=-1).mean().item(),
    "c_inh_norm": c_inh.norm(dim=-1).mean().item(),
}
```

**Why this matters:**
- Monitor for gate collapse during training
- Track context magnitude (detect explosion/vanishing)
- Essential for debugging if experiments fail

**Score: 8/10** - Good monitoring, but could add entropy metrics.

---

## ⚠️ Minor Issues (Not Blockers)

### 1. MUL and DIV Use ADD Context

```python
# tks_4op_block_v2.py:146-147
g_mul = sigmoid(self.mul_gate_proj(c_add))    # Uses c_add
denom = eps + softplus(self.div_denom_proj(c_add))  # Also c_add
```

**Why this might matter later:**
- If "what to bind" (MUL) is fundamentally different from "what to accumulate" (ADD), using the same context may limit expressiveness
- For the **current experiment** (testing inhibition hypothesis), this is fine
- The key test is ADD vs SUB with separate contexts

**Recommendation:**
- Keep as-is for Phase 1 (A vs B)
- If Phase 2 (B vs C) shows C doesn't add value, consider whether separate MUL context would help
- This is a **Phase 3 refinement**, not a Phase 1 blocker

---

### 2. Gate Statistics Are Detached

```python
stats["w_add"] = w_add.mean().item()  # .item() breaks gradient
```

**Why this might matter:**
- Can't use gate statistics for differentiable auxiliary loss
- If gates collapse during training (e.g., `w_sub → 0`), can't add entropy penalty

**Mitigation:**
```python
# If collapse is observed, add this BEFORE detaching:
gate_entropy = -(w_sub * log(w_sub + 1e-8) +
                 (1-w_sub) * log(1-w_sub + 1e-8)).mean()
loss = task_loss - 0.01 * gate_entropy  # Encourage diversity
```

**Recommendation:**
- Monitor gate values during training
- If you see collapse (any gate < 0.1 for > 50% of training), add entropy loss
- For initial experiments, detached stats are fine for monitoring

---

### 3. No Auxiliary Loss by Default

**Potential issue:**
- Independent sigmoid gates reduce collapse risk vs softmax, but don't eliminate it
- Model could still learn to ignore SUB pathway if it's easier to just use ADD

**How to detect:**
```python
# During training, if you see:
w_sub.mean() < 0.1  # Consistently low
w_add.mean() > 0.9  # Consistently high
# → SUB pathway is being ignored
```

**Mitigation:**
```python
# Add to training loop:
if config.use_gate_penalty:
    # Encourage all gates to be used
    target_gate_mean = 0.5
    gate_penalty = ((w_sub.mean() - target_gate_mean)**2 +
                    (w_add.mean() - target_gate_mean)**2)
    loss = loss + 0.01 * gate_penalty
```

**Recommendation:**
- Run initial experiments without auxiliary loss (cleaner test)
- If gates collapse, add minimal auxiliary loss to encourage usage
- Document whether auxiliary loss was needed (important for scientific validity)

---

## 🔬 Experimental Design Assessment

### Dataset Quality: 9/10

**Strengths:**
- Clean synthetic task isolating negation scope
- Configurable complexity (1-3 properties)
- Tracks NOT and property positions for probes
- Sufficient scale (10k train, 2k val, 2k test)

**Potential improvements:**
- Add harder cases: "The wizard is not not wise" (double negation)
- Add scope ambiguity: "The hero is not brave or wise" (does "not" apply to both?)
- Add longer chains: "The warrior who is not cowardly is brave"

**For Phase 1:** Current dataset is excellent.

---

### Mechanistic Probe: 10/10

**This is exactly the right test:**

```python
# The smoking gun
for each "not" token:
    attention_to_predicate = inhib_attn[not_idx, predicate_idx]
    attention_to_other = inhib_attn[not_idx, other_indices]

    ratio = attention_to_predicate / attention_to_other

    if ratio > 2.5x:
        print("✓ Model learned meaningful negation targeting")
    else:
        print("✗ Inhibition attention is not negation-specific")
```

**Why this works:**
- Tests the **mechanism**, not just performance
- If B beats A but probe shows random attention → got right answer, wrong reason
- If B beats A and probe shows predicate focus → TKS hypothesis confirmed
- Visualization enables qualitative understanding

**Scientific rigor:**
- Control (Condition A) has no inhibition attention to probe
- Experimental (Condition B) should show high ratio if hypothesis is correct
- Full system (Condition C) tests whether MUL/DIV add value

---

### Experimental Ladder: 10/10

```
Phase 1: A vs B
├── Tests core hypothesis: Does separate inhibition attention help?
├── Clear success criterion: B > A on loss/accuracy
└── Mechanistic validation: Probe ratio > 2.5x

Phase 2: B vs C
├── Tests value-add: Do MUL/DIV contribute beyond SUB?
├── Clear success criterion: C significantly > B
└── If C ≈ B: SUB is sufficient; if C > B: full ops matter
```

**This is textbook experimental design:**
- Minimal condition A (baseline)
- Targeted condition B (core hypothesis)
- Full system C (complete architecture)
- Each comparison tests a specific question

---

## 📊 Expected Results Analysis

### If Hypothesis is Correct:

```
Condition A: Loss ~2.5, Acc ~0.65, Probe N/A
Condition B: Loss ~2.2, Acc ~0.72, Probe 3.2x  ← KEY RESULT
Condition C: Loss ~2.1, Acc ~0.73, Probe 3.4x

Interpretation:
✓ Separate inhibition attention provides ~10% improvement
✓ Inhibition attention learns to focus on negated predicates
✓ MUL/DIV provide marginal benefit (this task doesn't need them)
→ Hypothesis SUPPORTED
```

---

### If Hypothesis is Partially Correct:

```
Condition A: Loss ~2.5, Acc ~0.65
Condition B: Loss ~2.4, Acc ~0.67, Probe 1.8x  ← Modest improvement
Condition C: Loss ~2.3, Acc ~0.69, Probe 2.1x

Interpretation:
~ Separate inhibition helps slightly but probe shows weak targeting
~ Model may be learning negation but not through attention mechanism
→ Hypothesis PARTIALLY SUPPORTED - investigate mechanism
```

---

### If Hypothesis is Incorrect:

```
Condition A: Loss ~2.5, Acc ~0.65
Condition B: Loss ~2.5, Acc ~0.64, Probe 1.1x  ← No improvement
Condition C: Loss ~2.4, Acc ~0.66, Probe 1.2x

Interpretation:
✗ Separate inhibition doesn't help on this task
✗ No evidence of predicate-specific attention
→ Hypothesis NOT SUPPORTED
→ Possible reasons:
  - Task too simple (doesn't require separate inhibition)
  - Task too hard (need more data/bigger model)
  - Architecture issue (need different design)
```

---

## 🎯 Recommendations

### Phase 1: Run A vs B

**Priority: HIGH**

```bash
python experiment_negation_scope.py --conditions A B --epochs 30
```

**Success criteria:**
- B shows >5% accuracy improvement over A
- Probe ratio > 2.5x for Condition B
- Training curves show stable convergence

**If successful:**
- Visualize 10-20 attention examples
- Write up mechanism findings
- Proceed to Phase 2

**If unsuccessful:**
- Check gate statistics (is w_sub being used?)
- Visualize attention (is anything being learned?)
- Try easier task (shorter sequences, fewer properties)

---

### Phase 2: Add C if B Succeeds

**Priority: MEDIUM**

```bash
python experiment_negation_scope.py --conditions B C --epochs 30
```

**Research question:** Do MUL/DIV operations add value beyond SUB?

**Expected outcome:** C ≈ B (task doesn't require binding/normalization)

**If C >> B:** Investigate what MUL/DIV are doing via gate statistics

---

### Phase 3: Scale Up if Hypothesis Holds

**Priority: LOW (after validation)**

If Phase 1 shows strong evidence:
- Test on real language data (sentiment with negation, textual entailment)
- Scale model size (256 → 512 → 1024 dimensions)
- Test compositional cases (nested scope, double negation)
- Compare to syntactic parser + transformer baseline

---

## 🔧 Potential Enhancements (Future Work)

### 1. Separate MUL Context

```python
# Currently
g_mul = sigmoid(self.mul_gate_proj(c_add))

# Enhanced
c_mul = self.attn_mul(z_mul)  # Separate attention for binding
g_mul = sigmoid(self.mul_gate_proj(c_mul))
```

**When to implement:** If Phase 2 shows C ≈ B but you suspect MUL could help on harder tasks

---

### 2. Differentiable Gate Regularization

```python
# Add to forward pass (before detaching)
gate_entropy = self._compute_gate_entropy(w_add, w_sub, w_mul, w_div)
# Return gate_entropy as part of output
# In training loop: loss = task_loss - 0.01 * gate_entropy
```

**When to implement:** If gates collapse during initial training

---

### 3. Learnable Alpha_Sub

```python
# Currently
self.alpha_sub = alpha_sub  # Fixed hyperparameter

# Enhanced
self.alpha_sub = nn.Parameter(torch.tensor(alpha_sub))  # Learnable
```

**When to implement:** After Phase 1 validation, if you want to test adaptive inhibition strength

---

### 4. Layer-wise Inhibition Heads

```python
# Currently: Same inhib_heads for all layers
# Enhanced: Vary by layer
self.blocks = nn.ModuleList([
    TKS4OpBlockV2(d_model, n_heads,
                  inhib_heads=2 if i < n_layers//2 else 8)  # More in upper layers
    for i in range(n_layers)
])
```

**Hypothesis:** Lower layers focus on syntax, upper layers on semantics → more inhibition in upper layers

---

## 📝 Documentation Assessment

### Code Quality: 9/10

**Strengths:**
- Clear module separation (block, dataset, experiment, probe)
- Comprehensive docstrings
- Type hints throughout
- Self-contained (can run each module standalone)

**Minor improvements:**
- Add assertions for input shapes
- Add unit tests for each block operation
- Add example outputs in docstrings

---

### Experimental Reproducibility: 10/10

**Excellent:**
- Seed setting for all random operations
- Config saved to JSON
- Model checkpoints saved
- Results logged to structured format
- Visualization scripts included

**Can reproduce experiment from saved artifacts:**
```bash
# Load config
config = json.load(open("experiments/negation_scope/config.json"))

# Recreate exact experimental conditions
# Re-run probe on saved model
# Regenerate visualizations
```

---

## 🎓 Scientific Validity

### Hypothesis Statement: Clear

**Testable claim:** Models with separate inhibition attention will outperform baseline on negation scope resolution, and mechanistic probes will show that inhibition attention learns to focus on negated predicates.

---

### Experimental Controls: Appropriate

- Condition A: Minimal baseline (no separate inhibition)
- Condition B: Adds only separate inhibition (isolates variable)
- Condition C: Full system (tests completeness)

**Confounds controlled:**
- Same dataset for all conditions
- Same training procedure
- Same hyperparameters (except architecture)
- Same random seeds

---

### Mechanistic Validation: Rigorous

- Doesn't rely solely on performance metrics
- Probes internal representations (attention patterns)
- Has clear success criterion (ratio > 2.5x)
- Includes visualization for qualitative validation

---

## 🏆 Final Assessment

| Aspect | Score | Notes |
|--------|-------|-------|
| **Core Architecture** | 10/10 | Separate contexts, independent gates - exactly right |
| **Stability Mechanisms** | 9/10 | Tanh bounding, alpha scaling, epsilon guards |
| **Interpretability** | 10/10 | Full attention return, comprehensive probe |
| **Experimental Design** | 10/10 | Clean A/B/C comparison, mechanistic validation |
| **Dataset Quality** | 9/10 | Clean synthetic task, proper scope |
| **Documentation** | 9/10 | Clear code, good comments, reproducible |
| **Scientific Rigor** | 10/10 | Testable hypothesis, controls, validation |

**Overall: 9.6/10 - Production-ready experimental framework**

---

## 🚀 Final Recommendation

**The TKS4OpBlockV2 design is sound. The experimental framework is complete. Run the experiment.**

```bash
# Execute Phase 1
python experiment_negation_scope.py --conditions A B --epochs 30

# If B > A with probe > 2.5x:
#   → Hypothesis SUPPORTED
#   → Write up findings
#   → Proceed to Phase 2
#
# If B ≈ A or probe < 2.0:
#   → Investigate failure mode
#   → Check visualizations
#   → Iterate on design
```

**This is a well-designed test of the core TKS subtractive hypothesis. The mechanism is clean, the experiment is rigorous, and the validation is mechanistic. You have everything you need to get a definitive answer.**

Good luck with the experiments! 🎯
