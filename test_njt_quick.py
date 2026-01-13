"""Quick NJT test without v5 model dependencies."""
import sys
print("Starting imports...")

import torch
print("torch imported")

import torch.nn as nn
print("nn imported")

sys.path.insert(0, '.')
from tks_features.njt_circuits import (
    NJTConfig,
    NoeticGate,
    NJTPlus,
    NJTMinus,
    HysteresisMemory,
    DifferentialPair,
    RhythmOscillator,
    NJTLayer,
    create_njt_layer,
)
print("NJT imports successful!")

# Quick tests
def test_njt_basic():
    config = NJTConfig(dim=64, num_transistors=10)
    layer = NJTLayer(config)

    x = torch.randn(2, 8, 64)
    output, trace = layer(x)

    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    print(f"Basic NJT test PASSED! Output shape: {output.shape}")

def test_njt_plus():
    config = NJTConfig(dim=64, num_transistors=10)
    njt = NJTPlus(dim=64, num_transistors=10, config=config)

    x = torch.randn(2, 8, 64)
    output, _ = njt(x)

    assert output.shape == x.shape
    print(f"NJT+ test PASSED! Output shape: {output.shape}")

def test_njt_minus():
    config = NJTConfig(dim=64, num_transistors=10)
    njt = NJTMinus(dim=64, num_transistors=10, config=config)

    x = torch.randn(2, 8, 64)
    output, _ = njt(x)

    assert output.shape == x.shape
    print(f"NJT- test PASSED! Output shape: {output.shape}")

def test_hysteresis():
    hyst = HysteresisMemory(num_units=10, gap=0.3)

    # High input should activate
    high_input = torch.ones(2, 8, 10) * 0.9
    _ = hyst(high_input)

    # Check state
    assert hyst.state.sum() > 0, "Hysteresis should have activated"
    print(f"Hysteresis test PASSED!")

def test_differential_pair():
    config = NJTConfig(dim=64, num_transistors=10)
    dp = DifferentialPair(dim=64, num_options=3, config=config)

    x = torch.randn(2, 8, 64)
    output, weights, _ = dp(x)

    assert output.shape == x.shape
    assert weights.shape == (2, 8, 3)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 8), atol=1e-5)
    print(f"Differential Pair test PASSED! Weights sum to 1")

def test_factory():
    layer = create_njt_layer(dim=128, num_transistors=8)

    assert layer.config.dim == 128
    assert layer.config.num_transistors == 8
    print(f"Factory function test PASSED!")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Running NJT Quick Tests")
    print("="*50 + "\n")

    test_njt_basic()
    test_njt_plus()
    test_njt_minus()
    test_hysteresis()
    test_differential_pair()
    test_factory()

    print("\n" + "="*50)
    print("ALL NJT TESTS PASSED!")
    print("="*50)
