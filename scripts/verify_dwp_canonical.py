#!/usr/bin/env python3
"""Quick verification of D/W/P canonical extraction."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tks_llm_core_v2 import (
    RPMGatingMechanism, TKSLLMCorePipeline,
    DESIRE_INDICES, WISDOM_INDICES, POWER_INDICES, TOTAL_DIM,
    FOUNDATION_NAMES
)


def main():
    print("=" * 60)
    print("D/W/P Canonical Extraction Verification")
    print("=" * 60)

    # 1. Index verification
    print("\n--- Canonical Index Check ---")
    print(f"DESIRE (nu1,nu4,nu7 MVR): {DESIRE_INDICES}")
    d_expected = [1,4,7,11,14,17,21,24,27,31,34,37]  # Mind,Vibration,Rhythm × 4 worlds
    print(f"  Expected: {d_expected}")
    d_ok = DESIRE_INDICES == d_expected
    print(f"  Match: {'PASS' if d_ok else 'FAIL'}")

    print(f"\nWISDOM (nu5,nu6): {WISDOM_INDICES}")
    w_expected = [5,6,15,16,25,26,35,36]  # Female,Male × 4 worlds
    print(f"  Expected: {w_expected}")
    w_ok = WISDOM_INDICES == w_expected
    print(f"  Match: {'PASS' if w_ok else 'FAIL'}")

    print(f"\nPOWER (nu8,nu9): {POWER_INDICES}")
    p_expected = [8,9,18,19,28,29,38,39]  # Cause,Effect × 4 worlds
    print(f"  Expected: {p_expected}")
    p_ok = POWER_INDICES == p_expected
    print(f"  Match: {'PASS' if p_ok else 'FAIL'}")

    total_dims = len(DESIRE_INDICES) + len(WISDOM_INDICES) + len(POWER_INDICES)
    print(f"\nTotal D/W/P dims: {total_dims}/40 (covers 28 dims; nu0,2,3,10 excluded)")

    # 2. Buffer check
    print("\n--- RPM Buffer Check ---")
    rpm = RPMGatingMechanism(dim=TOTAL_DIM)
    print(f"desire_idx registered: {hasattr(rpm, 'desire_idx')}")
    print(f"wisdom_idx registered: {hasattr(rpm, 'wisdom_idx')}")
    print(f"power_idx registered: {hasattr(rpm, 'power_idx')}")

    # 3. Forward pass bounds check
    print("\n--- Forward Pass Bounds Check ---")
    thought = torch.randn(4, 8, TOTAL_DIM)
    out = rpm(thought, target_foundation=0)
    dwp = out['dwp_scores']
    gate = out['rpm_gate']

    print(f"DWP shape: {dwp.shape}")
    print(f"D range: [{dwp[:,:,:,0].min():.4f}, {dwp[:,:,:,0].max():.4f}]")
    print(f"W range: [{dwp[:,:,:,1].min():.4f}, {dwp[:,:,:,1].max():.4f}]")
    print(f"P range: [{dwp[:,:,:,2].min():.4f}, {dwp[:,:,:,2].max():.4f}]")
    print(f"Gate range: [{gate.min():.4f}, {gate.max():.4f}]")

    bounds_ok = (
        dwp.min() >= 0 and dwp.max() <= 1 and
        gate.min() >= 0 and gate.max() <= 1
    )
    print(f"All in [0,1]: {'PASS' if bounds_ok else 'FAIL'}")

    # 4. Gate = D * W * P verification
    print("\n--- Gate = D x W x P Check ---")
    for f_idx in [0, 3, 6]:
        out_f = rpm(thought, target_foundation=f_idx)
        gate_f = out_f['rpm_gate']
        dwp_f = out_f['dwp_scores'][:, :, f_idx, :]
        expected_gate = dwp_f[:, :, 0] * dwp_f[:, :, 1] * dwp_f[:, :, 2]
        diff = (gate_f - expected_gate).abs().max().item()
        status = "PASS" if diff < 1e-6 else "FAIL"
        print(f"  Foundation {f_idx}: max_diff={diff:.2e} [{status}]")

    # 5. Full pipeline test
    print("\n--- Full Pipeline Test ---")
    pipeline = TKSLLMCorePipeline(vocab_size=100)
    tokens = torch.randint(0, 100, (4, 8))
    out = pipeline(tokens, target_foundation=3, return_full_trace=True)

    dwp = out['trace']['dwp_scores']
    gate = out['rpm_gate']
    print(f"Pipeline DWP shape: {dwp.shape}")
    print(f"Pipeline gate shape: {gate.shape}")

    pipeline_bounds_ok = (
        torch.isfinite(dwp).all() and
        dwp.min() >= 0 and dwp.max() <= 1 and
        gate.min() >= 0 and gate.max() <= 1
    )
    print(f"Pipeline bounds: {'PASS' if pipeline_bounds_ok else 'FAIL'}")

    # 6. Quick eval metric
    print("\n--- Quick D/W/P Eval Metric ---")
    batch = torch.randint(0, 100, (16, 16))  # 16 samples
    out = pipeline(batch, target_foundation=None, return_full_trace=True)
    dwp = out['trace']['dwp_scores']

    print("Per-foundation means:")
    for f_idx, f_name in enumerate(FOUNDATION_NAMES):
        d = dwp[:, :, f_idx, 0].mean().item()
        w = dwp[:, :, f_idx, 1].mean().item()
        p = dwp[:, :, f_idx, 2].mean().item()
        g = (d * w * p)
        print(f"  {f_name:15s}: D={d:.3f} W={w:.3f} P={p:.3f} Gate={g:.3f}")

    # Overall
    print("\n" + "=" * 60)
    all_pass = d_ok and w_ok and p_ok and bounds_ok and pipeline_bounds_ok
    print(f"OVERALL: {'ALL CHECKS PASS' if all_pass else 'SOME CHECKS FAILED'}")
    print("=" * 60)

    return all_pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
