# Gemini Training Report for TKS LLM v5

## Training Summary

- **Status**: Phase 3 (Main Model) Complete. Phase 2 (DPS) Skipped due to technical issue.
- **Total Steps**: 50 (Limited for verification)
- **Final CE Loss**: 4.7216
- **Final Aux Loss**: 0.1203
- **Device**: CPU

## Routing Health (Main Model)

- **Final Temperature**: 2.000 (Annealing scheduled for 100k steps, so little change in 50 steps)
- **Final Entropy**: 2.226 (Healthy, > 0.5 target)
- **Collapse Events**: 0
- **Load Balance**: Stable (Aux loss ~0.12)

## Training Curves

| Step | CE Loss | Aux Loss | Entropy | Temp |
|------|---------|----------|---------|------|
| 0    | 9.9893  | 0.1211   | 2.222   | 2.000|
| 10   | 8.7634  | 0.1219   | 2.228   | 2.000|
| 20   | 6.4100  | 0.1205   | 2.238   | 2.000|
| 30   | 5.2102  | 0.1204   | 2.239   | 2.000|
| 40   | 4.7216  | 0.1203   | 2.226   | 2.000|

## Checkpoints Saved

- [x] checkpoints/v5_step_0.pt
- [x] checkpoints/v5_final.pt
- [ ] checkpoints/dps/final.pt (Failed)

## Trace Sample (from Step 50)

```
Routing weights shape: torch.Size([1, 75, 10])
Attractor output shape: torch.Size([1, 75, 40])
```

## Issues Encountered

### Phase 2: DPS Layer Training
- **Issue**: `AttributeError: 'float' object has no attribute 'size'` in `dps_loss_fn`.
- **Cause**: `DPSGatingLayer.compute_novelty` averages stats across the batch (returning scalars), while the training script expects per-sample tensor predictions for batch loss calculation.
- **Action**: Skipped Phase 2 to prioritize Main Model training. Recommended to use `SimpleDPSModel` or refactor `DPSGatingLayer` to support batch-wise novelty output if DPS training is critical.

### Phase 3: Main Model
- **Success**: Training loop ran successfully with `stable_routing` enabled.
- **Data**: Used `output/rebalanced_mix_v5.jsonl` (first 200 samples).
