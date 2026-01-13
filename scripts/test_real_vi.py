#!/usr/bin/env python3
"""
REAL Test of Trained V/I Networks

This tests whether the trained V/I networks ACTUALLY produce
different outputs for valid vs invalid reasoning.

This is NOT a simulation - it uses the actual DPSGatingLayer
with loaded pretrained weights.
"""

import sys
import os
import json

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tks_features.dps_gating import DPSGatingLayer, DPSConfig, DPSState

# =============================================================================
# TEXT ENCODER (must match training)
# =============================================================================

class SimpleTextEncoder(nn.Module):
    """Same encoder used during training."""

    def __init__(self, vocab_size: int = 256, embed_dim: int = 64, hidden_dim: int = 128, output_dim: int = 40):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        _, (h, _) = self.lstm(emb)
        h = torch.cat([h[0], h[1]], dim=-1)
        return self.proj(h)


def text_to_ids(text: str, max_len: int = 256) -> torch.Tensor:
    """Convert text to character IDs."""
    ids = [ord(c) % 256 for c in text[:max_len]]
    ids = ids + [0] * (max_len - len(ids))
    return torch.tensor([ids], dtype=torch.long)


def main():
    print("="*60)
    print("REAL V/I NETWORK TEST")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load text encoders (with trained weights)
    v_checkpoint = torch.load('checkpoints/dps/validity_net.pt', map_location=device)
    i_checkpoint = torch.load('checkpoints/dps/impact_net.pt', map_location=device)

    v_encoder = SimpleTextEncoder(output_dim=40).to(device)
    i_encoder = SimpleTextEncoder(output_dim=40).to(device)

    v_encoder.load_state_dict(v_checkpoint['encoder'])
    i_encoder.load_state_dict(i_checkpoint['encoder'])

    # Create DPS layer and load pretrained V/I
    config = DPSConfig(
        validity_dim=32,
        impact_dim=32,
        hidden_dim=64,
    )
    dps = DPSGatingLayer(config, noetic_dim=40).to(device)

    # Load pretrained weights
    result = dps.load_pretrained_vi(
        validity_path='checkpoints/dps/validity_net.pt',
        impact_path='checkpoints/dps/impact_net.pt',
        device=device,
    )
    print(f"Load result: {result}")

    # =================================================================
    # TEST 1: Valid vs Invalid Reasoning
    # =================================================================
    print("\n" + "-"*60)
    print("TEST 1: VALID vs INVALID REASONING")
    print("-"*60)

    valid_texts = [
        "Step 1: CHECK 10D (Physical Desire) = 0.85 [STRONG] -> PASS",
        "RECURSE -> Sub-goal: Develop spiritual discipline",
        "RESOLVE 2W (Spiritual Wisdom) = 0.92 -> RESOLVED",
        "Goal: achieve wisdom | CHECK 3P = 0.7 -> PASS | RESOLVE -> ACHIEVED",
    ]

    invalid_texts = [
        "CHECK 10D = 0.8 -> PASS | CHECK 10D = 0.2 -> PASS | Contradiction!",
        "Because wisdom is strong, wisdom must be strong. QED.",
        "BLORK at FLARB -> ZZZZ -> INVALID SYNTAX 42",
        "Obviously everything is sufficient. No evidence needed.",
    ]

    print("\nVALID reasoning V scores:")
    valid_v_scores = []
    for text in valid_texts:
        ids = text_to_ids(text).to(device)
        with torch.no_grad():
            emb = v_encoder(ids)  # [1, 40]
            emb_3d = emb.unsqueeze(1)  # [1, 1, 40] for DPS
            v_out = dps.validity_net(emb)  # [1, 3]
            v_score = v_out.mean().item()
            valid_v_scores.append(v_score)
            print(f"  V={v_score:.3f} | {text[:50]}...")

    print("\nINVALID reasoning V scores:")
    invalid_v_scores = []
    for text in invalid_texts:
        ids = text_to_ids(text).to(device)
        with torch.no_grad():
            emb = v_encoder(ids)
            v_out = dps.validity_net(emb)
            v_score = v_out.mean().item()
            invalid_v_scores.append(v_score)
            print(f"  V={v_score:.3f} | {text[:50]}...")

    avg_valid_v = sum(valid_v_scores) / len(valid_v_scores)
    avg_invalid_v = sum(invalid_v_scores) / len(invalid_v_scores)
    v_separation = avg_valid_v - avg_invalid_v

    print(f"\nAvg Valid V:   {avg_valid_v:.3f}")
    print(f"Avg Invalid V: {avg_invalid_v:.3f}")
    print(f"Separation:    {v_separation:.3f}")
    v_test_pass = v_separation > 0.1
    print(f"TEST 1 {'PASS' if v_test_pass else 'FAIL'}: Valid V > Invalid V by {v_separation:.3f}")

    # =================================================================
    # TEST 2: Breakthrough vs Trivial Impact
    # =================================================================
    print("\n" + "-"*60)
    print("TEST 2: BREAKTHROUGH vs TRIVIAL IMPACT")
    print("-"*60)

    breakthrough_texts = [
        "KEY INSIGHT: the recursive structure mirrors the goal hierarchy",
        "BREAKTHROUGH: After 5 recursive attempts, discovered the pattern",
        "RESOLVE 2W -> Goal ACHIEVED. This changes everything!",
        "Simplification found: reduces problem to base case",
    ]

    trivial_texts = [
        "Step N: checking the next node. Step N+1: checking the next node.",
        "As mentioned before, still working on it. Repeating: still working.",
        "BLOCK -> BLOCK -> BLOCK (no progress)",
        "Clearly, more of the same. Obviously, more of the same.",
    ]

    print("\nBREAKTHROUGH I scores:")
    breakthrough_i_scores = []
    for text in breakthrough_texts:
        ids = text_to_ids(text).to(device)
        with torch.no_grad():
            emb = i_encoder(ids)
            # Impact net expects [emb, context]
            context = torch.zeros(1, 4, device=device)
            impact_input = torch.cat([emb, context], dim=-1)
            i_out = dps.impact_net(impact_input)
            i_score = i_out.mean().item()
            breakthrough_i_scores.append(i_score)
            print(f"  I={i_score:.3f} | {text[:50]}...")

    print("\nTRIVIAL I scores:")
    trivial_i_scores = []
    for text in trivial_texts:
        ids = text_to_ids(text).to(device)
        with torch.no_grad():
            emb = i_encoder(ids)
            context = torch.zeros(1, 4, device=device)
            impact_input = torch.cat([emb, context], dim=-1)
            i_out = dps.impact_net(impact_input)
            i_score = i_out.mean().item()
            trivial_i_scores.append(i_score)
            print(f"  I={i_score:.3f} | {text[:50]}...")

    avg_breakthrough_i = sum(breakthrough_i_scores) / len(breakthrough_i_scores)
    avg_trivial_i = sum(trivial_i_scores) / len(trivial_i_scores)
    i_separation = avg_breakthrough_i - avg_trivial_i

    print(f"\nAvg Breakthrough I: {avg_breakthrough_i:.3f}")
    print(f"Avg Trivial I:      {avg_trivial_i:.3f}")
    print(f"Separation:         {i_separation:.3f}")
    i_test_pass = i_separation > 0.1
    print(f"TEST 2 {'PASS' if i_test_pass else 'FAIL'}: Breakthrough I > Trivial I by {i_separation:.3f}")

    # =================================================================
    # TEST 3: Full NW Computation
    # =================================================================
    print("\n" + "-"*60)
    print("TEST 3: FULL NW COMPUTATION (V x I x (1-S))")
    print("-"*60)

    # Good thought: valid AND breakthrough
    good_text = "KEY INSIGHT: CHECK 2W = 0.95 -> PASS | RESOLVE -> Goal achieved!"
    # Bad thought: invalid AND trivial
    bad_text = "BLORK BLORK checking... checking... no progress"

    ids_good = text_to_ids(good_text).to(device)
    ids_bad = text_to_ids(bad_text).to(device)

    with torch.no_grad():
        # Good thought
        emb_good = v_encoder(ids_good)
        v_good = dps.validity_net(emb_good).mean().item()

        emb_good_i = i_encoder(ids_good)
        context = torch.zeros(1, 4, device=device)
        i_good = dps.impact_net(torch.cat([emb_good_i, context], dim=-1)).mean().item()

        # S = 0 (no memory)
        nw_good = v_good * i_good * 1.0

        # Bad thought
        emb_bad = v_encoder(ids_bad)
        v_bad = dps.validity_net(emb_bad).mean().item()

        emb_bad_i = i_encoder(ids_bad)
        i_bad = dps.impact_net(torch.cat([emb_bad_i, context], dim=-1)).mean().item()

        nw_bad = v_bad * i_bad * 1.0

    print(f"\nGOOD thought: '{good_text[:40]}...'")
    print(f"  V={v_good:.3f}, I={i_good:.3f}, S=0.0 -> NW={nw_good:.3f}")

    print(f"\nBAD thought: '{bad_text[:40]}...'")
    print(f"  V={v_bad:.3f}, I={i_bad:.3f}, S=0.0 -> NW={nw_bad:.3f}")

    nw_separation = nw_good - nw_bad
    print(f"\nNW Separation: {nw_separation:.3f}")
    nw_test_pass = nw_separation > 0.05
    print(f"TEST 3 {'PASS' if nw_test_pass else 'FAIL'}: Good NW > Bad NW by {nw_separation:.3f}")

    # =================================================================
    # SUMMARY
    # =================================================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_pass = v_test_pass and i_test_pass and nw_test_pass

    print(f"TEST 1 (V separates valid/invalid):     {'PASS' if v_test_pass else 'FAIL'}")
    print(f"TEST 2 (I separates breakthrough/trivial): {'PASS' if i_test_pass else 'FAIL'}")
    print(f"TEST 3 (NW separates good/bad thoughts):   {'PASS' if nw_test_pass else 'FAIL'}")
    print()

    if all_pass:
        print("VERDICT: REAL SUCCESS")
        print("The trained V/I networks genuinely distinguish:")
        print("  - Valid reasoning from invalid")
        print("  - Breakthroughs from trivial steps")
        print("  - Good thoughts get higher NW than bad thoughts")
        print("\nThis means TRUE earned depth is now possible!")
    else:
        print("VERDICT: NEEDS WORK")
        print("The V/I networks are not sufficiently separating the categories.")
        print("More/better training data may be needed.")


if __name__ == "__main__":
    main()
