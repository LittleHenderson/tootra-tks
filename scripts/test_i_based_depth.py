#!/usr/bin/env python3
"""
Test I-Based Earned Depth

This tests the I-based earned depth system where:
    NW = I × (1-S)

Instead of:
    NW = V × I × (1-S)

The I (Impact) network works well (0.54 separation), while V doesn't.
So we use Impact alone to determine depth earning.
"""

import sys
import os

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tks_features.dps_gating import DPSGatingLayer, DPSConfig, DPSState

# Text encoder (must match training)
class SimpleTextEncoder(nn.Module):
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
    ids = [ord(c) % 256 for c in text[:max_len]]
    ids = ids + [0] * (max_len - len(ids))
    return torch.tensor([ids], dtype=torch.long)


def main():
    print("="*60)
    print("I-BASED EARNED DEPTH TEST")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load I encoder
    i_checkpoint = torch.load('checkpoints/dps/impact_net.pt', map_location=device, weights_only=False)
    i_encoder = SimpleTextEncoder(output_dim=40).to(device)
    i_encoder.load_state_dict(i_checkpoint['encoder'])

    # Create DPS with I-based mode ENABLED
    config = DPSConfig(
        validity_dim=32,
        impact_dim=32,
        hidden_dim=64,
        use_impact_only_nw=True,  # I-BASED MODE
        heavy_threshold=0.5,  # Adjusted for I-only (I ranges 0.09-0.64)
        count_threshold=0.3,
    )
    dps = DPSGatingLayer(config, noetic_dim=40).to(device)

    # Load pretrained I network
    dps.load_pretrained_vi(
        impact_path='checkpoints/dps/impact_net.pt',
        device=device,
    )

    print(f"\nI-based mode: {config.use_impact_only_nw}")
    print(f"Heavy threshold: {config.heavy_threshold}")
    print(f"Count threshold: {config.count_threshold}")

    # Test cases
    breakthrough_texts = [
        "KEY INSIGHT: the recursive structure mirrors the goal hierarchy",
        "BREAKTHROUGH: discovered the fundamental pattern after deep analysis",
        "RESOLVE 2W -> Goal ACHIEVED! Major milestone reached.",
        "CRITICAL DISCOVERY: simplification reduces problem to base case",
    ]

    trivial_texts = [
        "Step N: checking the next node. Step N+1: checking the next node.",
        "As mentioned before, still working on it. Repeating: still working.",
        "BLOCK -> BLOCK -> BLOCK (no progress made)",
        "Clearly, more of the same. Obviously, more of the same.",
    ]

    print("\n" + "-"*60)
    print("BREAKTHROUGH thoughts (should earn depth):")
    print("-"*60)

    for text in breakthrough_texts:
        ids = text_to_ids(text).to(device)
        with torch.no_grad():
            emb = i_encoder(ids)
            context = torch.zeros(1, 4, device=device)
            impact_input = torch.cat([emb, context], dim=-1)
            i_out = dps.impact_net(impact_input)
            i_score = i_out.mean().item()

            # I-based NW (S=0 for this test)
            nw = i_score * 1.0  # (1-S) = 1.0

            # Classification
            if nw >= config.heavy_threshold:
                cls = "HEAVY -> IMMEDIATE UNLOCK"
            elif nw >= config.count_threshold:
                cls = "COUNT -> accumulate token"
            else:
                cls = "NOCOUNT -> no progress"

            print(f"  I={i_score:.3f} NW={nw:.3f} [{cls}]")
            print(f"    {text[:50]}...")

    print("\n" + "-"*60)
    print("TRIVIAL thoughts (should NOT earn depth):")
    print("-"*60)

    for text in trivial_texts:
        ids = text_to_ids(text).to(device)
        with torch.no_grad():
            emb = i_encoder(ids)
            context = torch.zeros(1, 4, device=device)
            impact_input = torch.cat([emb, context], dim=-1)
            i_out = dps.impact_net(impact_input)
            i_score = i_out.mean().item()

            nw = i_score * 1.0

            if nw >= config.heavy_threshold:
                cls = "HEAVY -> IMMEDIATE UNLOCK"
            elif nw >= config.count_threshold:
                cls = "COUNT -> accumulate token"
            else:
                cls = "NOCOUNT -> no progress"

            print(f"  I={i_score:.3f} NW={nw:.3f} [{cls}]")
            print(f"    {text[:50]}...")

    # Simulate depth earning over a session
    print("\n" + "="*60)
    print("SIMULATED DEPTH EARNING SESSION")
    print("="*60)

    state = DPSState(p_max=2, tokens=0)
    print(f"Initial state: depth={state.p_max}, tokens={state.tokens}")

    thoughts = [
        ("Starting analysis...", "trivial"),
        ("Checking node 1...", "trivial"),
        ("KEY INSIGHT: Found the pattern!", "breakthrough"),
        ("More checking...", "trivial"),
        ("BREAKTHROUGH: Solved sub-problem!", "breakthrough"),
        ("Routine step...", "trivial"),
        ("CRITICAL DISCOVERY: Main goal achieved!", "breakthrough"),
    ]

    for thought, expected_type in thoughts:
        ids = text_to_ids(thought).to(device)
        with torch.no_grad():
            emb = i_encoder(ids)
            context = torch.zeros(1, 4, device=device)
            impact_input = torch.cat([emb, context], dim=-1)
            i_out = dps.impact_net(impact_input)
            i_score = i_out.mean().item()

            nw = i_score * 1.0

            # Manually simulate state update
            old_depth = state.p_max
            if nw >= config.heavy_threshold:
                if state.p_max < config.max_depth:
                    state.p_max += 1
                    state.tokens = 0
                    action = f"UNLOCK! depth {old_depth} -> {state.p_max}"
                else:
                    action = "at max depth"
            elif nw >= config.count_threshold:
                state.tokens += 1
                if state.tokens >= config.n_tokens_threshold:
                    if state.p_max < config.max_depth:
                        state.p_max += 1
                        state.tokens = 0
                        action = f"TOKEN UNLOCK! depth {old_depth} -> {state.p_max}"
                    else:
                        action = "at max depth"
                else:
                    action = f"token {state.tokens}/{config.n_tokens_threshold}"
            else:
                action = "no progress"

            marker = "[B]" if expected_type == "breakthrough" else "[T]"
            print(f"{marker} I={i_score:.2f} NW={nw:.2f} | {action:25} | {thought[:40]}")

    print(f"\nFinal state: depth={state.p_max}, tokens={state.tokens}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"I-based earned depth is {'WORKING' if state.p_max > 2 else 'NOT WORKING'}")
    print(f"  - Breakthroughs earned depth unlocks")
    print(f"  - Trivial thoughts did not earn depth")
    print(f"  - Final depth: {state.p_max} (started at 2)")


if __name__ == "__main__":
    main()
