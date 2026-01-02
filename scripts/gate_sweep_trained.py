#!/usr/bin/env python3
"""
Gate sweep on trained v4 model with operator core.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tks_llm_core_v4 import TKSNoeticLM, TKSNoeticLMConfig
from scripts.operator_core_gate_sweep import (
    SimpleTokenizer,
    run_full_metrics,
    set_gate_value,
    EQUATION_WORLD_TEXTS,
    NL_WORLD_TEXTS,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="output/v4_with_operator_core.pt")
    parser.add_argument("--gates", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3])
    parser.add_argument("--output", default="output/gate_sweep_trained_results.json")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load trained checkpoint
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    cfg = checkpoint['config']

    tokenizer = SimpleTokenizer()

    config = TKSNoeticLMConfig(
        vocab_size=cfg['vocab_size'],
        max_seq_len=cfg['max_seq_len'],
        num_layers=cfg['num_layers'],
        use_operator_core=cfg['use_operator_core'],
        operator_gate_init=cfg['operator_gate_init'],
    )

    model = TKSNoeticLM(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded trained model from: {args.model_path}")
    print(f"  Training world_acc: {checkpoint.get('world_acc', 'N/A')}")

    print("\n" + "=" * 80)
    print("GATE SWEEP ON TRAINED MODEL")
    print("=" * 80)

    results = {}

    for gate in args.gates:
        print(f"\n--- Gate = {gate:.2f} ---")
        set_gate_value(model, gate)

        actual_gate = torch.sigmoid(model.operator_core.gate.gate_logits).mean().item()
        print(f"  Actual gate: {actual_gate:.4f}")

        metrics = run_full_metrics(model, tokenizer, device, use_triplet=(gate > 0))
        results[gate] = metrics.to_dict()

        print(f"  World Sep (EQ): {metrics.world_separation_eq:.1%}")
        print(f"  World Sep (NL): {metrics.world_separation_nl:.1%}")
        print(f"  Noetic Cons (EQ): {metrics.noetic_consistency_eq:.4f}")
        print(f"  Noetic Cons (NL): {metrics.noetic_consistency_nl:.4f}")
        print(f"  Attractor Conv: {metrics.attractor_convergence:.0%}")
        print(f"  Sym Violation: {metrics.symmetric_violation:.4f}")
        print(f"  Asym Violation: {metrics.antisymmetric_violation:.4f}")

    # Analysis
    print("\n" + "=" * 80)
    print("COMPARISON TO GATE=0")
    print("=" * 80)

    baseline = results[0.0]
    for gate in sorted(results.keys()):
        if gate == 0.0:
            continue
        r = results[gate]
        print(f"\nGate = {gate:.2f} vs 0.0:")
        print(f"  World Sep (EQ): {baseline['world_separation_eq']:.1%} -> {r['world_separation_eq']:.1%} ({r['world_separation_eq'] - baseline['world_separation_eq']:+.1%})")
        print(f"  World Sep (NL): {baseline['world_separation_nl']:.1%} -> {r['world_separation_nl']:.1%}")
        print(f"  Noetic Cons (EQ): {baseline['noetic_consistency_eq']:.4f} -> {r['noetic_consistency_eq']:.4f} ({r['noetic_consistency_eq'] - baseline['noetic_consistency_eq']:+.4f})")
        print(f"  Attractor: {baseline['attractor_convergence']:.0%} -> {r['attractor_convergence']:.0%}")

    # Find optimal
    def score(gate):
        r = results[gate]
        return (
            r['world_separation_eq'] * 2.0 +
            r['noetic_consistency_eq'] * 1.0 +
            r['attractor_convergence'] * 1.5 +
            (1.0 - r['symmetric_violation']) * 0.5 +
            r['world_separation_nl'] * 0.5
        )

    scores = [(g, score(g)) for g in results.keys()]
    scores.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 80)
    print("OPTIMAL GATE")
    print("=" * 80)
    print("\nScores (higher = better):")
    for g, s in scores:
        print(f"  gate={g:.2f}: {s:.3f}")

    print(f"\nOptimal: gate={scores[0][0]:.2f}")

    # Save
    with open(args.output, 'w') as f:
        json.dump({
            'results': {str(k): v for k, v in results.items()},
            'scores': {str(g): s for g, s in scores},
            'optimal': scores[0][0],
        }, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
