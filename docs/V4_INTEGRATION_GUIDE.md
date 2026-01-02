# TKS-LLM v4 Integration Guide

**Version**: 4.0
**Date**: 2025-12-22
**Status**: Integration Phase
**Supervisor Agent**: tks-supervisor

---

## Overview

This document provides a step-by-step integration guide for TKS-LLM v4 components. Previous agents have created:

- **StableAttractorLayer**: Fixed-point iteration with guaranteed convergence (spectral normalization)
- **WorldClassificationLoss**: Auxiliary loss for world separation (fixes A/B/C/D confusion)
- **RPMDifferentiationLoss**: Auxiliary loss for D/W/P differentiation (fixes 0.94 collapse)
- **CI Regression Tests**: Automated tests in `tests/` directory

These components must now be integrated into the actual training pipeline (`scripts/train_cuda.py`).

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Integration Checklist](#2-integration-checklist)
3. [Step 1: Enable StableAttractorLayer](#3-step-1-enable-stableattractorlayer)
4. [Step 2: Integrate TKSLoss](#4-step-2-integrate-tksloss)
5. [Step 3: Add World/RPM Supervision](#5-step-3-add-worldrpm-supervision)
6. [Step 4: Enable Curriculum Scheduling](#6-step-4-enable-curriculum-scheduling)
7. [Step 5: Add Convergence Monitoring](#7-step-5-add-convergence-monitoring)
8. [Validation Checkpoints](#8-validation-checkpoints)
9. [Dependency Graph](#9-dependency-graph)
10. [Handoff Notes](#10-handoff-notes)

---

## 1. Prerequisites

### Required Files

| File | Purpose | Status |
|------|---------|--------|
| `tks_llm_core_v2.py` | StableAttractorLayer, TKSLLMCorePipeline | Complete |
| `training/losses.py` | TKSLoss, WorldClassificationLoss, RPMDifferentiationLoss | Complete |
| `scripts/train_cuda.py` | Main training script (target for integration) | Needs modification |
| `tks_rules/noetics.py` | INVOLUTION_PAIRS constant | Must exist |

### Verify Imports Work

```bash
# Test that all components can be imported
python -c "
from tks_llm_core_v2 import TKSLLMCorePipeline, StableAttractorLayer
from training.losses import TKSLoss, TKSLossConfig, WorldClassificationLoss, RPMDifferentiationLoss
print('All imports successful')
"
```

---

## 2. Integration Checklist

### Phase A: Core Pipeline (No Data Changes)

- [ ] **A.1** Verify `TKSLLMCorePipeline` uses `StableAttractorLayer` by default
- [ ] **A.2** Add `return_full_trace=True` to model forward calls
- [ ] **A.3** Replace `CrossEntropyLoss` with `TKSLoss`
- [ ] **A.4** Add loss config CLI arguments

### Phase B: Enhanced Losses (Requires Labels)

- [ ] **B.1** Add world labels to dataset (`world_label` field)
- [ ] **B.2** Add RPM labels to dataset (`rpm_label` field)
- [ ] **B.3** Integrate `WorldClassificationLoss`
- [ ] **B.4** Integrate `RPMDifferentiationLoss`

### Phase C: Curriculum and Monitoring

- [ ] **C.1** Add `CurriculumLossScheduler`
- [ ] **C.2** Add attractor convergence monitoring
- [ ] **C.3** Log loss components to metrics
- [ ] **C.4** Add TensorBoard/WandB logging (optional)

### Phase D: CI/CD Integration

- [ ] **D.1** Ensure `tests/test_regression_gate.py` passes
- [ ] **D.2** Add integration test for full training loop
- [ ] **D.3** Update CI workflow

---

## 3. Step 1: Enable StableAttractorLayer

### Current State (train_cuda.py line 683-690)

```python
model = TKSLLMCorePipeline(
    vocab_size=tokenizer.actual_vocab_size,
    hidden_dim=args.hidden_dim,
    noetic_dim=40,
    num_scales=3,
    max_attractor_iter=args.num_layers,
    contraction_factor=0.5
)
```

### Required Change

The `TKSLLMCorePipeline` constructor already defaults to `use_stable_attractor=True`, so no change is strictly needed. However, to make it explicit and configurable:

```python
# In argument parser (around line 621)
parser.add_argument('--use-stable-attractor', action='store_true', default=True,
                    help='Use StableAttractorLayer with guaranteed convergence')
parser.add_argument('--no-stable-attractor', action='store_false', dest='use_stable_attractor',
                    help='Use legacy AttractorComputationLayer')

# In model creation (around line 683)
model = TKSLLMCorePipeline(
    vocab_size=tokenizer.actual_vocab_size,
    hidden_dim=args.hidden_dim,
    noetic_dim=40,
    num_scales=3,
    max_attractor_iter=args.num_layers,
    contraction_factor=0.5,
    use_stable_attractor=args.use_stable_attractor  # ADD THIS
)
```

### Validation

```python
# Verify attractor type
print(f"Attractor type: {type(model.attractor).__name__}")
# Should print: StableAttractorLayer
```

---

## 4. Step 2: Integrate TKSLoss

### Current State (train_cuda.py line 326)

```python
# Loss function
self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)
```

### Required Changes

#### 4.1 Add imports at top of file

```python
# Add after line 36
from training.losses import (
    TKSLoss,
    TKSLossConfig,
    CurriculumLossScheduler,
    WorldClassificationLoss,
    RPMDifferentiationLoss,
    WorldRPMSupervisedLoss,
)
```

#### 4.2 Add CLI arguments

```python
# Add to argument parser (after line 628)
parser.add_argument('--use-tks-loss', action='store_true', default=False,
                    help='Use full TKS loss function (requires return_full_trace)')
parser.add_argument('--lambda-task', type=float, default=1.0,
                    help='Weight for task loss')
parser.add_argument('--lambda-rpm', type=float, default=0.5,
                    help='Weight for RPM loss')
parser.add_argument('--lambda-attractor', type=float, default=0.3,
                    help='Weight for attractor convergence loss')
parser.add_argument('--lambda-involution', type=float, default=0.2,
                    help='Weight for involution constraint loss')
parser.add_argument('--lambda-spectral', type=float, default=0.1,
                    help='Weight for spectral radius loss')
parser.add_argument('--lambda-cascade', type=float, default=0.2,
                    help='Weight for cascade flow loss')
```

#### 4.3 Modify CUDATrainer.__init__

```python
# Replace self.loss_fn initialization (around line 326)
if config.get('use_tks_loss', False):
    loss_config = TKSLossConfig(
        lambda_task=config.get('lambda_task', 1.0),
        lambda_rpm=config.get('lambda_rpm', 0.5),
        lambda_attractor=config.get('lambda_attractor', 0.3),
        lambda_involution=config.get('lambda_involution', 0.2),
        lambda_spectral=config.get('lambda_spectral', 0.1),
        lambda_cascade=config.get('lambda_cascade', 0.2),
    )
    self.loss_fn = TKSLoss(loss_config)
    self.use_tks_loss = True
    logger.info(f"Using TKSLoss with config: {loss_config.to_dict()}")
else:
    self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    self.use_tks_loss = False
```

#### 4.4 Modify train_epoch forward pass

```python
# In train_epoch (around line 411-431)
with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
    # Must use return_full_trace=True for TKS losses
    output = self.model(input_ids, return_full_trace=self.use_tks_loss)
    logits = self._get_logits(output)

    if self.use_tks_loss:
        # Get the raw model (unwrap DataParallel if needed)
        raw_model = self.model.module if hasattr(self.model, 'module') else self.model

        # Compute TKS loss (requires pipeline reference)
        loss_dict = self.loss_fn(
            pipeline_output=output,
            targets=targets,
            pipeline=raw_model,
            compute_all=True
        )
        loss = loss_dict['total']

        # Log individual loss components (every N steps)
        if batch_idx % 100 == 0:
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor) and k != 'total':
                    logger.debug(f"  {k}: {v.item():.4f}")
    else:
        # Original loss computation
        if 'loss_mask' in batch:
            loss_mask = batch['loss_mask'].to(self.device, non_blocking=True)
            per_token_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction='none'
            )
            masked_loss = per_token_loss * loss_mask.view(-1)
            loss = masked_loss.sum() / (loss_mask.sum() + 1e-8)
        else:
            loss = self.loss_fn(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
```

#### 4.5 Update config dict (around line 697)

```python
config = {
    # ... existing config ...
    'use_tks_loss': args.use_tks_loss,
    'lambda_task': args.lambda_task,
    'lambda_rpm': args.lambda_rpm,
    'lambda_attractor': args.lambda_attractor,
    'lambda_involution': args.lambda_involution,
    'lambda_spectral': args.lambda_spectral,
    'lambda_cascade': args.lambda_cascade,
}
```

---

## 5. Step 3: Add World/RPM Supervision

This requires labeled data with `world_label` and `rpm_label` fields.

### 5.1 Update TKSDataset

```python
# In TKSDataset.__getitem__ (around line 152)
def __getitem__(self, idx):
    entry = self.entries[idx]

    # ... existing code ...

    result = {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'targets': torch.tensor(targets, dtype=torch.long),
    }

    # Add world label if present (0=A, 1=B, 2=C, 3=D)
    if 'world_label' in entry:
        result['world_label'] = torch.tensor(entry['world_label'], dtype=torch.long)

    # Add RPM label if present (0=desire, 1=wisdom, 2=power)
    if 'rpm_label' in entry:
        result['rpm_label'] = torch.tensor(entry['rpm_label'], dtype=torch.long)

    # Add soft labels if present
    if 'world_weights' in entry:
        result['world_weights'] = torch.tensor(entry['world_weights'], dtype=torch.float)

    if 'rpm_weights' in entry:
        result['rpm_weights'] = torch.tensor(entry['rpm_weights'], dtype=torch.float)

    return result
```

### 5.2 Add WorldRPMSupervisedLoss to CUDATrainer

```python
# In CUDATrainer.__init__
if config.get('use_world_rpm_supervision', False):
    self.world_rpm_loss = WorldRPMSupervisedLoss(
        lambda_world=config.get('lambda_world', 1.0),
        lambda_rpm=config.get('lambda_rpm_diff', 1.0),
    )
    self.use_world_rpm = True
else:
    self.world_rpm_loss = None
    self.use_world_rpm = False
```

### 5.3 Add to training loop

```python
# In train_epoch, after computing main loss
if self.use_world_rpm and 'world_label' in batch:
    world_labels = batch['world_label'].to(self.device)
    rpm_labels = batch.get('rpm_label')
    if rpm_labels is not None:
        rpm_labels = rpm_labels.to(self.device)

    # Get noetic embedding from trace
    noetic_embedding = output['trace']['embedding']
    dwp_scores = output['trace']['dwp_scores']

    world_rpm_losses = self.world_rpm_loss(
        noetic_embedding=noetic_embedding,
        dwp_scores=dwp_scores,
        world_labels=world_labels,
        rpm_labels=rpm_labels,
        world_weights=batch.get('world_weights'),
        rpm_weights=batch.get('rpm_weights'),
    )

    # Add to total loss
    loss = loss + config.get('lambda_world_rpm', 0.5) * world_rpm_losses['total']
```

---

## 6. Step 4: Enable Curriculum Scheduling

The `CurriculumLossScheduler` allows staged training where different losses are introduced progressively.

### 6.1 Add CLI argument

```python
parser.add_argument('--curriculum-stage', type=int, default=5,
                    help='Curriculum stage (1-5). Stage 5 = all losses.')
```

### 6.2 Integrate scheduler

```python
# In CUDATrainer.__init__
if config.get('use_tks_loss', False):
    self.curriculum_scheduler = CurriculumLossScheduler(
        base_config=loss_config,
        warmup_steps=config.get('warmup_steps', 1000)
    )
    self.current_stage = config.get('curriculum_stage', 5)
else:
    self.curriculum_scheduler = None
```

### 6.3 Use in training loop

```python
# At start of train_epoch
if self.curriculum_scheduler is not None:
    current_config = self.curriculum_scheduler.step(
        global_step=epoch * len(self.train_loader) + batch_idx,
        stage=self.current_stage
    )
    # Update loss function config dynamically
    self.loss_fn.config = current_config
```

### Curriculum Stages Reference

| Stage | Task | RPM | Attractor | Involution | Spectral | Cascade |
|-------|------|-----|-----------|------------|----------|---------|
| 1     | 1.0  | 0.0 | 0.0       | 0.0        | 0.0      | 0.0     |
| 2     | 1.0  | 0.0 | 0.0       | 0.2        | 0.1      | 0.0     |
| 3     | 1.0  | 0.3 | 0.0       | 0.2        | 0.1      | 0.1     |
| 4     | 1.0  | 0.4 | 0.2       | 0.2        | 0.1      | 0.2     |
| 5     | Full config (user-specified weights)                      |

---

## 7. Step 5: Add Convergence Monitoring

### 7.1 Track attractor convergence in metrics

```python
# In train_epoch, after forward pass
if self.use_tks_pipeline and 'trace' in output:
    trace = output['trace']
    if 'attractor_converged' in trace:
        self.metrics.setdefault('attractor_convergence', []).append(
            1.0 if trace['attractor_converged'] else 0.0
        )
    if 'attractor_iterations' in trace:
        self.metrics.setdefault('attractor_iterations', []).append(
            trace['attractor_iterations']
        )
```

### 7.2 Log convergence rate per epoch

```python
# At end of train_epoch
if 'attractor_convergence' in self.metrics:
    conv_rate = sum(self.metrics['attractor_convergence'][-len(self.train_loader):]) / len(self.train_loader)
    logger.info(f"  Attractor convergence rate: {conv_rate*100:.1f}%")
```

### 7.3 Add Lipschitz penalty (optional)

```python
from tks_llm_core_v2 import compute_lipschitz_penalty

# In training loop (periodically, e.g., every 100 steps)
if batch_idx % 100 == 0 and hasattr(self.model, 'attractor'):
    raw_model = self.model.module if hasattr(self.model, 'module') else self.model
    lip_penalty = compute_lipschitz_penalty(raw_model.attractor, target_lipschitz=0.9)
    if lip_penalty > 0:
        loss = loss + 0.01 * lip_penalty  # Small weight
```

---

## 8. Validation Checkpoints

### Checkpoint 1: Basic TKSLoss Integration

```bash
python scripts/train_cuda.py \
    --data output/teacher_augmented.jsonl \
    --epochs 1 \
    --batch-size 4 \
    --use-tks-loss \
    --output-dir output/v4_test_1
```

**Expected**: Training runs without errors. Loss decreases. Log shows individual loss components.

### Checkpoint 2: StableAttractorLayer Convergence

```bash
python -c "
from tks_llm_core_v2 import TKSLLMCorePipeline, run_stable_attractor_test
results = run_stable_attractor_test()
assert results['stable_convergence_rate'] >= 90, 'Convergence rate too low'
print('PASS: StableAttractorLayer convergence verified')
"
```

**Expected**: Convergence rate >= 90%.

### Checkpoint 3: World/RPM Losses Compute

```bash
python -c "
import torch
from training.losses import WorldClassificationLoss, RPMDifferentiationLoss

# Test WorldClassificationLoss
wcl = WorldClassificationLoss()
embedding = torch.randn(4, 8, 40)
labels = torch.randint(0, 4, (4,))
out = wcl(embedding, labels)
print(f'WorldClassificationLoss: {out[\"total\"].item():.4f}')

# Test RPMDifferentiationLoss
rdl = RPMDifferentiationLoss()
dwp = torch.rand(4, 8, 7, 3)
rpm_labels = torch.randint(0, 3, (4,))
out = rdl(dwp, rpm_labels)
print(f'RPMDifferentiationLoss: {out[\"total\"].item():.4f}')

print('PASS: World/RPM losses compute correctly')
"
```

### Checkpoint 4: CI Tests Pass

```bash
pytest tests/test_regression_gate.py -v
pytest tests/ -k "loss or attractor" -v
```

**Expected**: All tests pass.

### Checkpoint 5: Full Training Run

```bash
python scripts/train_cuda.py \
    --data output/teacher_augmented.jsonl \
    --epochs 5 \
    --batch-size 8 \
    --use-tks-loss \
    --lambda-task 1.0 \
    --lambda-rpm 0.5 \
    --lambda-attractor 0.3 \
    --output-dir output/v4_full_test
```

**Expected**:
- Training completes
- Eval loss decreases over epochs
- Attractor convergence rate > 80%
- No NaN/Inf in losses

---

## 9. Dependency Graph

```
                    +-----------------------+
                    |   tks_rules/noetics   |
                    |   (INVOLUTION_PAIRS)  |
                    +-----------+-----------+
                                |
                                v
+---------------------------+   |   +---------------------------+
|     tks_llm_core.py       |<--+-->|    training/losses.py     |
| - NoeticEmbeddingLayer    |       | - TKSLoss                 |
| - NoeticProcessor         |       | - WorldClassificationLoss |
| - FractalAttention        |       | - RPMDifferentiationLoss  |
+-------------+-------------+       | - CurriculumLossScheduler |
              |                     +-------------+-------------+
              v                                   |
+---------------------------+                     |
|   tks_llm_core_v2.py      |                     |
| - AttractorComputationLyr |                     |
| - StableAttractorLayer    |                     |
| - RPMGatingMechanism      |                     |
| - TKSLLMCorePipeline      |                     |
+-------------+-------------+                     |
              |                                   |
              +----------------+------------------+
                               |
                               v
                    +---------------------+
                    | scripts/train_cuda.py|
                    |   (INTEGRATION)     |
                    +---------------------+
```

### Task Dependencies

| Task | Depends On | Blocks |
|------|------------|--------|
| A.1 (StableAttractor) | tks_llm_core_v2.py | A.2, A.3 |
| A.2 (return_full_trace) | A.1 | A.3, B.3, B.4 |
| A.3 (TKSLoss) | training/losses.py, A.2 | C.1 |
| B.1 (world labels) | Dataset spec | B.3 |
| B.2 (rpm labels) | Dataset spec | B.4 |
| B.3 (WorldClassLoss) | A.2, B.1 | C.2 |
| B.4 (RPMDiffLoss) | A.2, B.2 | C.2 |
| C.1 (Curriculum) | A.3 | - |
| C.2 (Monitoring) | A.3, B.3, B.4 | - |
| D.1 (CI Tests) | All above | - |

---

## 10. Handoff Notes

### For Next Agent (tks-compiler or integration specialist)

1. **Immediate Next Steps**:
   - Implement Phase A changes in `scripts/train_cuda.py`
   - Run Checkpoint 1 to verify basic integration
   - Run existing CI tests to ensure no regressions

2. **Dataset Requirements for Phase B**:
   - Training data needs `world_label` (int 0-3) field
   - Training data needs `rpm_label` (int 0-2) field
   - Consider running a labeling pass on existing `teacher_augmented.jsonl`

3. **Known Issues**:
   - Legacy `AttractorComputationLayer` has 0% convergence rate (fixed by StableAttractorLayer)
   - World B activation bleeding into World A expressions (fixed by WorldClassificationLoss)
   - D/W/P scores clustering at 0.94 (fixed by RPMDifferentiationLoss)

4. **Testing Priority**:
   - `tests/test_regression_gate.py` must pass before merge
   - Add new test: `tests/test_tks_loss_integration.py`

5. **Documentation Updates Needed**:
   - Update `docs/TRAINING_INTEGRATION_PLAN.md` with v4 status
   - Update `README.md` with new CLI arguments

### File Modification Summary

| File | Changes Required |
|------|-----------------|
| `scripts/train_cuda.py` | Major changes (imports, args, loss, training loop) |
| `training/datasets.py` | Minor (add world/rpm label extraction) |
| `tests/test_tks_loss_integration.py` | New file |
| `.github/workflows/ci.yaml` | Add integration test job |

---

## Appendix: Quick Reference Code Snippets

### Complete TKSLoss Usage

```python
from training.losses import TKSLoss, TKSLossConfig
from tks_llm_core_v2 import TKSLLMCorePipeline

# Configure
config = TKSLossConfig(
    lambda_task=1.0,
    lambda_rpm=0.5,
    lambda_attractor=0.3,
    lambda_involution=0.2,
    lambda_spectral=0.1,
    lambda_cascade=0.2,
)
loss_fn = TKSLoss(config)

# Model
model = TKSLLMCorePipeline(vocab_size=1000, use_stable_attractor=True)

# Forward with trace
tokens = torch.randint(0, 1000, (4, 32))
targets = tokens[:, 1:]
output = model(tokens[:, :-1], return_full_trace=True)

# Compute loss
losses = loss_fn(
    pipeline_output=output,
    targets=targets,
    pipeline=model,
    compute_all=True
)

# Backward
losses['total'].backward()
```

### Verify StableAttractorLayer

```python
from tks_llm_core_v2 import StableAttractorLayer

attractor = StableAttractorLayer(dim=40, contraction_factor=0.5)

# Verify contraction
check = attractor.verify_contraction(num_samples=100)
print(f"Hutchinson Lipschitz: {check['hutchinson_lipschitz']:.4f}")
print(f"Is contraction: {check['hutchinson_is_contraction']}")

# Test convergence
x = torch.randn(4, 8, 40)
out = attractor(x, return_metrics=True)
print(f"Converged: {out['converged']}")
print(f"Iterations: {out['iterations']}")
print(f"Final delta: {out['final_delta']:.6f}")
```

---

**Document End**

*Generated by tks-supervisor agent for TKS v4 integration phase.*
