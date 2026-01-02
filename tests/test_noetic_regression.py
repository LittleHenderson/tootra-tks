"""
Noetic Coherence Regression Tests for TKS CI Gate

This module implements pytest-based regression tests for noetic coherence metrics
that must pass for model checkpoints to be accepted. These tests enforce the
mathematical constraints from TKS v7.x specifications.

Metrics tested:
1. World Separation - Target world should have highest activation (60%+)
2. RPM Differentiation - D/W/P categories should have distinct values (variance > 0.01)
3. Attractor Convergence - Attractor should converge on 80%+ of inputs
4. Noetic Consistency - Similar texts should cluster, dissimilar should separate (> 0.05)

Usage:
    # Run all tests (both equation and NL prompts)
    pytest tests/test_noetic_regression.py -v

    # Run with specific model
    pytest tests/test_noetic_regression.py --model output/model.pt -v

    # Run with equation-style prompts only (for CI)
    pytest tests/test_noetic_regression.py --model output/model.pt --test-style equation -v

    # Run with NL prompts only (generalization test)
    pytest tests/test_noetic_regression.py --model output/model.pt --test-style nl -v

Test Styles:
    - equation: Training format (A1 + A2 + A3, etc.) - should pass quickly
    - nl: Natural language (Divine realm, Mental focus, etc.) - generalization test
    - both: Combined (default) - comprehensive validation

References:
    - TKS_LLM_Noetic_Mathematics_v1.0.md Section 7
    - TKS_FORMAL_MATHEMATICAL_MANUAL_v7.2_COMPLETE.tex
    - scripts/noetic_coherence_test.py (metric computation)

Author: TKS CI/Evaluation Specialist (Agent C)
Date: 2025-12-22
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Optional imports - tests skip if torch/model not available
try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from tks_llm_core_v2 import TKSLLMCorePipeline, TOTAL_DIM
    from scripts.train_cuda import SimpleTokenizer
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

# v4 model support
try:
    from tks_llm_core_v4 import TKSNoeticLM, TKSNoeticLMConfig
    V4_AVAILABLE = True
except ImportError:
    V4_AVAILABLE = False


# =============================================================================
# THRESHOLDS - Based on TKS Mathematical Requirements
# =============================================================================

@dataclass
class NoeticThresholds:
    """
    Pass/fail thresholds for noetic coherence metrics.

    These values are derived from TKS v7.x mathematical specifications:
    - World separation reflects 4-world orthogonality requirement
    - RPM variance ensures D/W/P gating is differentiated
    - Attractor convergence follows Banach fixed-point theorem
    - Noetic consistency enforces semantic clustering
    """

    # World Separation: Target world should win 60%+ of tests
    # Rationale: 4 worlds = 25% baseline, 60% = significantly above random
    WORLD_SEPARATION_THRESHOLD: float = 0.60

    # RPM Variance: D/W/P means should differ by > 0.01
    # Rationale: D/W/P are distinct noetic indices, must show differentiation
    RPM_VARIANCE_THRESHOLD: float = 0.01

    # Attractor Convergence: 80%+ should converge
    # Rationale: Banach fixed-point guarantees convergence for contractions
    ATTRACTOR_CONVERGENCE_THRESHOLD: float = 0.80

    # Noetic Consistency: Separation score > 0.05
    # Rationale: Similar inputs must map to similar noetic states
    NOETIC_CONSISTENCY_THRESHOLD: float = 0.05

    # Warning thresholds (between pass and fail)
    WORLD_SEPARATION_WARNING: float = 0.50
    RPM_VARIANCE_WARNING: float = 0.005
    ATTRACTOR_CONVERGENCE_WARNING: float = 0.70
    NOETIC_CONSISTENCY_WARNING: float = 0.02


THRESHOLDS = NoeticThresholds()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_model(checkpoint_path: str, device: Optional[torch.device] = None, model_type: str = 'v2'):
    """
    Load TKS model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint (.pt file)
        device: Target device (auto-detected if None)
        model_type: 'v2' for TKSLLMCorePipeline, 'v4' for TKSNoeticLM

    Returns:
        Tuple of (model, tokenizer, device, model_type)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Default expressiveness flags (will be overridden by config or inferred from state_dict)
    use_subfoundation_mixer = False
    use_acquisition_program = False
    use_foundation_graph = False

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        config = checkpoint.get('config', {})
        hidden_dim = config.get('hidden_dim', 256)
        num_layers = config.get('num_layers', 4)
        vocab_size = config.get('vocab_size', None)
        num_scales = config.get('num_scales', 3)
        contraction_factor = config.get('contraction_factor', 0.5)
        max_seq_len = config.get('max_seq_len', 256)
        use_stable_attractor = config.get('use_stable_attractor', True)
        # Read max_attractor_iter from config, fallback to num_layers
        max_attractor_iter = config.get('max_attractor_iter', num_layers)
        # Auto-detect model type from config if present
        model_type = config.get('model_type', model_type)
        # Expressiveness config
        use_subfoundation_mixer = config.get('use_subfoundation_mixer', False)
        use_acquisition_program = config.get('use_acquisition_program', False)
        use_foundation_graph = config.get('use_foundation_graph', False)
    else:
        state_dict = checkpoint
        hidden_dim = 256
        num_layers = 4
        vocab_size = None
        num_scales = 3
        contraction_factor = 0.5
        max_seq_len = 256
        use_stable_attractor = True
        max_attractor_iter = num_layers

    tokenizer = SimpleTokenizer(max_length=max_seq_len)

    # Determine vocab size from state dict if not in config
    if vocab_size is None:
        # Try to infer from embedding weight
        for key in state_dict:
            if 'embedding' in key and 'weight' in key:
                vocab_size = state_dict[key].shape[0]
                break
        if vocab_size is None:
            vocab_size = tokenizer.actual_vocab_size

    # Infer num_layers from state_dict keys if not in config
    inferred_layers = set()
    for key in state_dict:
        if key.startswith('blocks.'):
            parts = key.split('.')
            if len(parts) > 1 and parts[1].isdigit():
                inferred_layers.add(int(parts[1]))
    if inferred_layers:
        num_layers = max(inferred_layers) + 1

    # Infer expressiveness config from state_dict (if not already set from config)
    if not use_subfoundation_mixer:
        use_subfoundation_mixer = any('subfoundation_mixer' in k for k in state_dict)
    if not use_acquisition_program:
        use_acquisition_program = any('acquisition_program' in k for k in state_dict)
    if not use_foundation_graph:
        use_foundation_graph = any('foundation_diffusion' in k for k in state_dict)

    # Auto-detect model type from state_dict structure
    if any('blocks.' in k and '.expressiveness.' in k for k in state_dict):
        model_type = 'v4'
    elif any('transformer.layers' in k for k in state_dict):
        model_type = 'v2'

    if model_type == 'v4' and V4_AVAILABLE:
        # Load v4 TKSNoeticLM
        v4_config = TKSNoeticLMConfig(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            num_scales=num_scales,
            use_attractor=True,
            use_stable_attractor=use_stable_attractor,
            contraction_factor=contraction_factor,
            max_attractor_iter=max_attractor_iter,
            use_rpm=True,
            # Expressiveness stack (inferred from state_dict)
            use_subfoundation_mixer=use_subfoundation_mixer,
            use_acquisition_program=use_acquisition_program,
            use_foundation_graph=use_foundation_graph,
        )
        model = TKSNoeticLM(v4_config)
    else:
        # Default to v2 TKSLLMCorePipeline
        model = TKSLLMCorePipeline(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            noetic_dim=TOTAL_DIM,
            num_scales=num_scales,
            max_attractor_iter=max_attractor_iter,
            contraction_factor=contraction_factor,
        )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, tokenizer, device, model_type


def get_noetic_state(model, tokenizer, text: str, device: torch.device) -> Dict:
    """Extract noetic state for input text."""
    input_ids = tokenizer.tokenize(text)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        output = model(input_tensor)

    gated_output = output['gated_output'][0]  # [seq_len, 40]
    rpm_gate = output['rpm_gate'][0]  # [seq_len]
    attractor_converged = output['attractor_converged']

    content_len = min(len(text) + 2, gated_output.shape[0])
    noetic_state = gated_output[:content_len].mean(dim=0).cpu().numpy()
    rpm_mean = rpm_gate[:content_len].mean().item()

    world_states = {
        'A': noetic_state[0:10],
        'B': noetic_state[10:20],
        'C': noetic_state[20:30],
        'D': noetic_state[30:40],
    }

    return {
        'noetic_state': noetic_state,
        'world_states': world_states,
        'rpm_gate': rpm_mean,
        'attractor_converged': bool(attractor_converged),
    }


def cosine_similarity(a, b) -> float:
    """Compute cosine similarity between two vectors."""
    import numpy as np
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# =============================================================================
# TEST DATA - Canonical Test Cases
# =============================================================================

# Similar pairs: should have high noetic similarity (NL-style)
NL_SIMILAR_PAIRS = [
    ("Mental focus and clarity", "Clear thinking and concentration"),
    ("Emotional warmth and love", "Feeling affection and care"),
    ("Physical energy and power", "Bodily strength and vitality"),
    ("Spiritual awareness", "Divine consciousness"),
]

# Similar pairs: equation-style (same world = similar)
EQUATION_SIMILAR_PAIRS = [
    ("A1+A2+A3", "A4+A5+A6"),
    ("B1+B2+B3", "B4+B5+B6"),
    ("C1+C2+C3", "C4+C5+C6"),
    ("D1+D2+D3", "D4+D5+D6"),
]

# Dissimilar pairs: should have lower noetic similarity (NL-style)
NL_DISSIMILAR_PAIRS = [
    ("Mental focus and clarity", "Physical energy and power"),
    ("Emotional warmth and love", "Spiritual awareness"),
    ("1:7:9", "Fear and anxiety"),
    ("A1 + B2 + C3", "The sun sets in the west"),
]

# Dissimilar pairs: equation-style (different worlds = dissimilar)
EQUATION_DISSIMILAR_PAIRS = [
    ("A1+A2+A3", "D1+D2+D3"),
    ("B1+B2+B3", "C1+C2+C3"),
    ("A7+A8+A9", "B7+B8+B9"),
    ("C4+C5+C6", "D4+D5+D6"),
]

# Legacy aliases for backward compatibility
SIMILAR_PAIRS = NL_SIMILAR_PAIRS
DISSIMILAR_PAIRS = NL_DISSIMILAR_PAIRS

# World-specific texts for world separation testing

# Equation-style prompts (matches training format - no spaces around +)
EQUATION_WORLD_TEXTS = {
    'A': ["A1+A2+A3", "A4+A5+A6", "A7+A8+A9"],
    'B': ["B1+B2+B3", "B4+B5+B6", "B7+B8+B9"],
    'C': ["C1+C2+C3", "C4+C5+C6", "C7+C8+C9"],
    'D': ["D1+D2+D3", "D4+D5+D6", "D7+D8+D9"],
}

# NL-style prompts (generalization test - current prompts)
NL_WORLD_TEXTS = {
    'A': ["Divine realm", "Spiritual awareness", "Sacred consciousness"],
    'B': ["Mental focus", "Cognitive clarity", "Thinking deeply"],
    'C': ["Emotional flow", "Feeling warmth", "Heart connection"],
    'D': ["Physical power", "Bodily energy", "Material strength"],
}

# Legacy combined format (for backward compatibility)
WORLD_TEXTS = {
    'A': ["A1 + A2 + A3", "Divine realm", "Spiritual awareness A"],
    'B': ["B4 + B5 + B6", "Mental focus", "Cognitive clarity B"],
    'C': ["C7 + C8 + C9", "Emotional flow", "Feeling warmth C"],
    'D': ["D10 + D1 + D2", "Physical power", "Bodily energy D"],
}

# RPM test cases by D/W/P category (NL-style)
NL_RPM_TEST_CASES = {
    'desire': [
        "I want power and control",
        "Desire for manifestation",
        "Vibration of intention",
    ],
    'wisdom': [
        "Understanding the balance",
        "Male and female principles",
        "Wisdom of equilibrium",
    ],
    'power': [
        "Cause and effect in action",
        "The power of consequence",
        "Forces shaping reality",
    ],
}

# RPM test cases by D/W/P category (equation-style - no spaces around +)
# Desire = nu1, nu4, nu7 (indices 1, 4, 7 in each world)
# Wisdom = nu5, nu6 (indices 5, 6 in each world)
# Power = nu8, nu9 (indices 8, 9 in each world)
EQUATION_RPM_TEST_CASES = {
    'desire': [
        "A1+A4+A7",  # Desire noetics in World A
        "B1+B4+B7",  # Desire noetics in World B
        "C1+C4+C7",  # Desire noetics in World C
    ],
    'wisdom': [
        "A5+A6",     # Wisdom noetics in World A
        "B5+B6",     # Wisdom noetics in World B
        "C5+C6",     # Wisdom noetics in World C
    ],
    'power': [
        "A8+A9",     # Power noetics in World A
        "B8+B9",     # Power noetics in World B
        "C8+C9",     # Power noetics in World C
    ],
}

# Legacy alias for backward compatibility
RPM_TEST_CASES = NL_RPM_TEST_CASES

# Attractor test texts (NL-style)
NL_ATTRACTOR_TEXTS = [
    "The rhythm of cause and effect",
    "Desire leads to manifestation through wisdom",
    "Dynamic: Awareness of pattern triggered",
    "Mental focus brings clarity",
    "Physical power manifests change",
]

# Attractor test texts (equation-style - no spaces around +)
EQUATION_ATTRACTOR_TEXTS = [
    "A1+A2+A3",
    "B4+B5+B6",
    "C7+C8+C9",
    "D1+D4+D7",
    "A1+B2+C3+D4",
]

# Legacy alias for backward compatibility
ATTRACTOR_TEXTS = NL_ATTRACTOR_TEXTS


# =============================================================================
# METRIC COMPUTATION FUNCTIONS
# =============================================================================

def compute_noetic_consistency(model, tokenizer, device, similar_pairs=None, dissimilar_pairs=None) -> Dict:
    """
    Compute noetic consistency metric.

    Args:
        model: TKS model
        tokenizer: Tokenizer
        device: Torch device
        similar_pairs: List of (text1, text2) pairs that should be similar. Defaults to NL_SIMILAR_PAIRS.
        dissimilar_pairs: List of (text1, text2) pairs that should be dissimilar. Defaults to NL_DISSIMILAR_PAIRS.

    Returns:
        Dict with 'separation' score and 'passed' boolean
    """
    import numpy as np

    if similar_pairs is None:
        similar_pairs = NL_SIMILAR_PAIRS
    if dissimilar_pairs is None:
        dissimilar_pairs = NL_DISSIMILAR_PAIRS

    similar_scores = []
    for text1, text2 in similar_pairs:
        state1 = get_noetic_state(model, tokenizer, text1, device)
        state2 = get_noetic_state(model, tokenizer, text2, device)
        sim = cosine_similarity(state1['noetic_state'], state2['noetic_state'])
        similar_scores.append(sim)

    dissimilar_scores = []
    for text1, text2 in dissimilar_pairs:
        state1 = get_noetic_state(model, tokenizer, text1, device)
        state2 = get_noetic_state(model, tokenizer, text2, device)
        sim = cosine_similarity(state1['noetic_state'], state2['noetic_state'])
        dissimilar_scores.append(sim)

    mean_similar = np.mean(similar_scores)
    mean_dissimilar = np.mean(dissimilar_scores)
    separation = mean_similar - mean_dissimilar

    return {
        'similar_scores': similar_scores,
        'dissimilar_scores': dissimilar_scores,
        'mean_similar': float(mean_similar),
        'mean_dissimilar': float(mean_dissimilar),
        'separation': float(separation),
        'passed': separation > THRESHOLDS.NOETIC_CONSISTENCY_THRESHOLD,
        'warning': separation < THRESHOLDS.NOETIC_CONSISTENCY_WARNING,
    }


def compute_rpm_differentiation(model, tokenizer, device, rpm_test_cases=None) -> Dict:
    """
    Compute RPM differentiation metric.

    Args:
        model: TKS model
        tokenizer: Tokenizer
        device: Torch device
        rpm_test_cases: Dict mapping 'desire', 'wisdom', 'power' to list of test texts.
                       Defaults to NL_RPM_TEST_CASES.

    Returns:
        Dict with category means, variance, and 'passed' boolean
    """
    import numpy as np

    if rpm_test_cases is None:
        rpm_test_cases = NL_RPM_TEST_CASES

    category_means = {}
    for category, texts in rpm_test_cases.items():
        scores = []
        for text in texts:
            state = get_noetic_state(model, tokenizer, text, device)
            scores.append(state['rpm_gate'])
        category_means[category] = float(np.mean(scores))

    means = list(category_means.values())
    variance = float(np.var(means))

    # Check pairwise differences
    min_diff = min(
        abs(category_means['desire'] - category_means['wisdom']),
        abs(category_means['desire'] - category_means['power']),
        abs(category_means['wisdom'] - category_means['power']),
    )

    return {
        'category_means': category_means,
        'variance': variance,
        'min_pairwise_diff': float(min_diff),
        'passed': variance > THRESHOLDS.RPM_VARIANCE_THRESHOLD,
        'warning': variance < THRESHOLDS.RPM_VARIANCE_WARNING,
    }


def compute_world_separation(model, tokenizer, device, world_texts=None) -> Dict:
    """
    Compute world separation metric.

    Args:
        model: TKS model
        tokenizer: Tokenizer
        device: Torch device
        world_texts: Optional dict of world texts. Defaults to WORLD_TEXTS if None.

    Returns:
        Dict with correct_rate and 'passed' boolean
    """
    import numpy as np

    if world_texts is None:
        world_texts = WORLD_TEXTS

    correct_count = 0
    total_count = 0
    details = []

    for target_world, texts in world_texts.items():
        for text in texts:
            state = get_noetic_state(model, tokenizer, text, device)

            # Get activation for each world
            world_activations = {}
            for w in 'ABCD':
                world_activations[w] = float(np.linalg.norm(state['world_states'][w]))

            # Check if target world has highest activation
            max_world = max(world_activations, key=world_activations.get)
            is_correct = (max_world == target_world)

            if is_correct:
                correct_count += 1
            total_count += 1

            details.append({
                'text': text[:30],
                'target': target_world,
                'predicted': max_world,
                'correct': is_correct,
                'activations': world_activations,
            })

    correct_rate = correct_count / total_count if total_count > 0 else 0.0

    return {
        'correct_count': correct_count,
        'total_count': total_count,
        'correct_rate': float(correct_rate),
        'details': details,
        'passed': correct_rate >= THRESHOLDS.WORLD_SEPARATION_THRESHOLD,
        'warning': correct_rate < THRESHOLDS.WORLD_SEPARATION_WARNING,
    }


def compute_attractor_convergence(model, tokenizer, device, attractor_texts=None) -> Dict:
    """
    Compute attractor convergence metric.

    Args:
        model: TKS model
        tokenizer: Tokenizer
        device: Torch device
        attractor_texts: List of texts to test attractor convergence. Defaults to NL_ATTRACTOR_TEXTS.

    Returns:
        Dict with convergence_rate and 'passed' boolean
    """
    import numpy as np

    if attractor_texts is None:
        attractor_texts = NL_ATTRACTOR_TEXTS

    results = []
    for text in attractor_texts:
        state = get_noetic_state(model, tokenizer, text, device)
        noetic_norm = float(np.linalg.norm(state['noetic_state']))
        results.append({
            'text': text[:40],
            'converged': state['attractor_converged'],
            'noetic_norm': noetic_norm,
        })

    converged_count = sum(1 for r in results if r['converged'])
    convergence_rate = converged_count / len(results) if results else 0.0
    mean_norm = float(np.mean([r['noetic_norm'] for r in results]))

    return {
        'results': results,
        'converged_count': converged_count,
        'total_count': len(results),
        'convergence_rate': float(convergence_rate),
        'mean_norm': mean_norm,
        'passed': convergence_rate >= THRESHOLDS.ATTRACTOR_CONVERGENCE_THRESHOLD,
        'warning': convergence_rate < THRESHOLDS.ATTRACTOR_CONVERGENCE_WARNING,
    }


def compute_all_metrics(
    model, tokenizer, device,
    world_texts=None,
    similar_pairs=None,
    dissimilar_pairs=None,
    rpm_test_cases=None,
    attractor_texts=None
) -> Dict:
    """
    Compute all noetic coherence metrics.

    Args:
        model: TKS model
        tokenizer: Tokenizer
        device: Torch device
        world_texts: Optional dict of world texts for world separation test
        similar_pairs: Optional list of similar pairs for noetic consistency
        dissimilar_pairs: Optional list of dissimilar pairs for noetic consistency
        rpm_test_cases: Optional dict of RPM test cases
        attractor_texts: Optional list of attractor test texts

    Returns:
        Dict with all metric results
    """
    return {
        'noetic_consistency': compute_noetic_consistency(
            model, tokenizer, device, similar_pairs, dissimilar_pairs
        ),
        'rpm_differentiation': compute_rpm_differentiation(
            model, tokenizer, device, rpm_test_cases
        ),
        'world_separation': compute_world_separation(
            model, tokenizer, device, world_texts
        ),
        'attractor_convergence': compute_attractor_convergence(
            model, tokenizer, device, attractor_texts
        ),
    }


# =============================================================================
# PYTEST FIXTURES
# =============================================================================
# Note: pytest_addoption is defined in conftest.py to avoid conflicts


@pytest.fixture(scope="module")
def model_path(request):
    """Get model path from command line or environment."""
    cli_path = request.config.getoption("--model")
    if cli_path:
        return cli_path

    env_path = os.environ.get("TKS_MODEL_PATH")
    if env_path:
        return env_path

    # Check common locations
    common_paths = [
        "output/teacher_model_v3/final_model.pt",
        "output/teacher_model_long_v4/final_model.pt",
        "output/phase5_models/final_model.pt",
        "output/models/final_model.pt",
    ]

    for path in common_paths:
        if Path(path).exists():
            return path

    return None


@pytest.fixture(scope="module")
def model_type(request):
    """Get model type from command line (v2 or v4)."""
    return request.config.getoption("--model-type", default="v2")


@pytest.fixture(scope="module")
def loaded_model(model_path, model_type):
    """Load model and tokenizer for tests."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")
    if not MODEL_AVAILABLE:
        pytest.skip("TKSLLMCorePipeline not available")
    if model_type == 'v4' and not V4_AVAILABLE:
        pytest.skip("TKSNoeticLM (v4) not available")
    if model_path is None or not Path(model_path).exists():
        pytest.skip(f"Model checkpoint not found: {model_path}")

    model, tokenizer, device, actual_type = load_model(model_path, model_type=model_type)
    return {'model': model, 'tokenizer': tokenizer, 'device': device, 'path': model_path, 'type': actual_type}


@pytest.fixture(scope="module")
def metrics_output_path(request):
    """Get metrics output path from command line."""
    return request.config.getoption("--noetic-output")


@pytest.fixture(scope="module")
def test_style(request):
    """Get test style from command line."""
    return request.config.getoption("--test-style", default="both")


@pytest.fixture(scope="module")
def world_texts(test_style):
    """
    Get world-specific texts based on test style.

    Args:
        test_style: 'nl', 'equation', or 'both'

    Returns:
        Dict mapping world ('A', 'B', 'C', 'D') to list of test texts
    """
    if test_style == 'equation':
        return EQUATION_WORLD_TEXTS
    elif test_style == 'nl':
        return NL_WORLD_TEXTS
    else:  # 'both'
        # Combine both sets for comprehensive testing
        combined = {}
        for world in 'ABCD':
            combined[world] = EQUATION_WORLD_TEXTS[world] + NL_WORLD_TEXTS[world]
        return combined


@pytest.fixture(scope="module")
def similar_pairs(test_style):
    """Get similar pairs based on test style."""
    if test_style == 'equation':
        return EQUATION_SIMILAR_PAIRS
    elif test_style == 'nl':
        return NL_SIMILAR_PAIRS
    else:  # 'both'
        return EQUATION_SIMILAR_PAIRS + NL_SIMILAR_PAIRS


@pytest.fixture(scope="module")
def dissimilar_pairs(test_style):
    """Get dissimilar pairs based on test style."""
    if test_style == 'equation':
        return EQUATION_DISSIMILAR_PAIRS
    elif test_style == 'nl':
        return NL_DISSIMILAR_PAIRS
    else:  # 'both'
        return EQUATION_DISSIMILAR_PAIRS + NL_DISSIMILAR_PAIRS


@pytest.fixture(scope="module")
def rpm_test_cases(test_style):
    """Get RPM test cases based on test style."""
    if test_style == 'equation':
        return EQUATION_RPM_TEST_CASES
    elif test_style == 'nl':
        return NL_RPM_TEST_CASES
    else:  # 'both'
        # Combine both sets
        combined = {}
        for cat in ['desire', 'wisdom', 'power']:
            combined[cat] = EQUATION_RPM_TEST_CASES[cat] + NL_RPM_TEST_CASES[cat]
        return combined


@pytest.fixture(scope="module")
def attractor_texts(test_style):
    """Get attractor test texts based on test style."""
    if test_style == 'equation':
        return EQUATION_ATTRACTOR_TEXTS
    elif test_style == 'nl':
        return NL_ATTRACTOR_TEXTS
    else:  # 'both'
        return EQUATION_ATTRACTOR_TEXTS + NL_ATTRACTOR_TEXTS


# =============================================================================
# PYTEST TESTS
# =============================================================================

@pytest.mark.noetic
@pytest.mark.requires_model
class TestNoeticRegression:
    """
    Noetic coherence regression tests.

    These tests enforce the mathematical constraints from TKS specifications
    and serve as a CI gate for model quality.

    Markers:
        noetic: Part of noetic coherence regression suite
        requires_model: Requires a trained model checkpoint
    """

    @pytest.mark.noetic
    @pytest.mark.requires_model
    def test_world_separation(self, loaded_model, metrics_output_path, world_texts, test_style):
        """
        Test: Target world should have highest activation.

        Threshold: 60% correct rate (baseline random = 25%)

        Mathematical basis: TKS 4-world structure (A, B, C, D) with
        orthogonal noetic subspaces. World-specific inputs should
        preferentially activate their target world.
        """
        model = loaded_model['model']
        tokenizer = loaded_model['tokenizer']
        device = loaded_model['device']

        result = compute_world_separation(model, tokenizer, device, world_texts)

        print(f"\n{'='*70}")
        print("WORLD SEPARATION TEST")
        print(f"{'='*70}")
        print(f"Test style: {test_style}")
        print(f"Correct rate: {result['correct_rate']:.1%}")
        print(f"Threshold: {THRESHOLDS.WORLD_SEPARATION_THRESHOLD:.0%}")
        print(f"Details:")
        for d in result['details']:
            status = "OK" if d['correct'] else "MISS"
            print(f"  [{status}] '{d['text']}' target={d['target']} predicted={d['predicted']}")

        # Save metrics if output path specified
        if metrics_output_path:
            self._save_metric(metrics_output_path, 'world_separation', result)

        if result['warning'] and not result['passed']:
            pytest.fail(
                f"World separation failed: {result['correct_rate']:.1%} < "
                f"{THRESHOLDS.WORLD_SEPARATION_THRESHOLD:.0%} threshold"
            )

        assert result['passed'], (
            f"World separation too low: {result['correct_rate']:.1%} < "
            f"{THRESHOLDS.WORLD_SEPARATION_THRESHOLD:.0%}"
        )

    @pytest.mark.noetic
    @pytest.mark.requires_model
    def test_rpm_differentiation(self, loaded_model, metrics_output_path, rpm_test_cases, test_style):
        """
        Test: D/W/P categories should have distinct RPM values.

        Threshold: Variance > 0.01

        Mathematical basis: TKS RPM gating uses canonical noetic indices:
        - Desire: nu1, nu4, nu7 (Mind, Vibration, Rhythm)
        - Wisdom: nu5, nu6 (Female, Male)
        - Power: nu8, nu9 (Cause, Effect)

        Different content types should produce measurably different D/W/P scores.
        """
        model = loaded_model['model']
        tokenizer = loaded_model['tokenizer']
        device = loaded_model['device']

        result = compute_rpm_differentiation(model, tokenizer, device, rpm_test_cases)

        print(f"\n{'='*70}")
        print("RPM DIFFERENTIATION TEST")
        print(f"{'='*70}")
        print(f"Test style: {test_style}")
        print(f"Category means:")
        for cat, mean in result['category_means'].items():
            print(f"  {cat}: {mean:.4f}")
        print(f"Variance: {result['variance']:.6f}")
        print(f"Min pairwise diff: {result['min_pairwise_diff']:.4f}")
        print(f"Threshold: variance > {THRESHOLDS.RPM_VARIANCE_THRESHOLD}")

        if metrics_output_path:
            self._save_metric(metrics_output_path, 'rpm_differentiation', result)

        assert result['passed'], (
            f"RPM variance too low: {result['variance']:.6f} < "
            f"{THRESHOLDS.RPM_VARIANCE_THRESHOLD}"
        )

    @pytest.mark.noetic
    @pytest.mark.requires_model
    def test_attractor_convergence(self, loaded_model, metrics_output_path, attractor_texts, test_style):
        """
        Test: Attractor should converge on 80%+ of inputs.

        Threshold: 80% convergence rate

        Mathematical basis: Banach Fixed-Point Theorem guarantees
        convergence for contraction mappings with Lipschitz constant L < 1.
        The attractor layer uses iterative contraction to find fixed points.
        """
        model = loaded_model['model']
        tokenizer = loaded_model['tokenizer']
        device = loaded_model['device']

        result = compute_attractor_convergence(model, tokenizer, device, attractor_texts)

        print(f"\n{'='*70}")
        print("ATTRACTOR CONVERGENCE TEST")
        print(f"{'='*70}")
        print(f"Test style: {test_style}")
        print(f"Convergence rate: {result['convergence_rate']:.0%}")
        print(f"Threshold: {THRESHOLDS.ATTRACTOR_CONVERGENCE_THRESHOLD:.0%}")
        print(f"Mean noetic norm: {result['mean_norm']:.4f}")
        print(f"Details:")
        for r in result['results']:
            status = "CONVERGED" if r['converged'] else "NOT CONVERGED"
            print(f"  [{status}] norm={r['noetic_norm']:.4f} '{r['text']}'")

        if metrics_output_path:
            self._save_metric(metrics_output_path, 'attractor_convergence', result)

        assert result['passed'], (
            f"Attractor convergence too low: {result['convergence_rate']:.0%} < "
            f"{THRESHOLDS.ATTRACTOR_CONVERGENCE_THRESHOLD:.0%}"
        )

    @pytest.mark.noetic
    @pytest.mark.requires_model
    def test_noetic_consistency(self, loaded_model, metrics_output_path, similar_pairs, dissimilar_pairs, test_style):
        """
        Test: Similar texts should cluster, dissimilar should separate.

        Threshold: Separation score > 0.05

        Mathematical basis: Noetic embeddings should preserve semantic
        similarity. Similar inputs map to similar 40-dim noetic states.
        """
        model = loaded_model['model']
        tokenizer = loaded_model['tokenizer']
        device = loaded_model['device']

        result = compute_noetic_consistency(model, tokenizer, device, similar_pairs, dissimilar_pairs)

        print(f"\n{'='*70}")
        print("NOETIC CONSISTENCY TEST")
        print(f"{'='*70}")
        print(f"Test style: {test_style}")
        print(f"Mean similar score: {result['mean_similar']:.4f}")
        print(f"Mean dissimilar score: {result['mean_dissimilar']:.4f}")
        print(f"Separation: {result['separation']:.4f}")
        print(f"Threshold: separation > {THRESHOLDS.NOETIC_CONSISTENCY_THRESHOLD}")

        if metrics_output_path:
            self._save_metric(metrics_output_path, 'noetic_consistency', result)

        assert result['passed'], (
            f"Noetic consistency separation too low: {result['separation']:.4f} < "
            f"{THRESHOLDS.NOETIC_CONSISTENCY_THRESHOLD}"
        )

    def _save_metric(self, output_path: str, metric_name: str, result: Dict):
        """Save metric to output JSON file."""
        output_path = Path(output_path)

        # Load existing metrics if file exists
        if output_path.exists():
            with open(output_path, 'r') as f:
                metrics = json.load(f)
        else:
            metrics = {}

        # Add/update metric
        metrics[metric_name] = result

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)


class TestNoeticThresholdsValidation:
    """Validate that threshold constants are sensible."""

    def test_world_separation_threshold_valid(self):
        """World separation threshold should be above random chance (25%)."""
        assert THRESHOLDS.WORLD_SEPARATION_THRESHOLD > 0.25
        assert THRESHOLDS.WORLD_SEPARATION_THRESHOLD <= 1.0
        assert THRESHOLDS.WORLD_SEPARATION_WARNING < THRESHOLDS.WORLD_SEPARATION_THRESHOLD

    def test_rpm_variance_threshold_valid(self):
        """RPM variance threshold should be positive."""
        assert THRESHOLDS.RPM_VARIANCE_THRESHOLD > 0
        assert THRESHOLDS.RPM_VARIANCE_WARNING < THRESHOLDS.RPM_VARIANCE_THRESHOLD

    def test_attractor_convergence_threshold_valid(self):
        """Attractor convergence threshold should be reasonable."""
        assert 0.5 <= THRESHOLDS.ATTRACTOR_CONVERGENCE_THRESHOLD <= 1.0
        assert THRESHOLDS.ATTRACTOR_CONVERGENCE_WARNING < THRESHOLDS.ATTRACTOR_CONVERGENCE_THRESHOLD

    def test_noetic_consistency_threshold_valid(self):
        """Noetic consistency threshold should be positive."""
        assert THRESHOLDS.NOETIC_CONSISTENCY_THRESHOLD > 0
        assert THRESHOLDS.NOETIC_CONSISTENCY_WARNING < THRESHOLDS.NOETIC_CONSISTENCY_THRESHOLD


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

def run_standalone(model_path: str, output_path: Optional[str] = None, model_type: str = 'v2', test_style: str = 'both') -> Dict:
    """
    Run noetic regression tests standalone (without pytest).

    Args:
        model_path: Path to model checkpoint
        output_path: Optional path to save results JSON
        model_type: 'v2' for TKSLLMCorePipeline, 'v4' for TKSNoeticLM
        test_style: 'nl', 'equation', or 'both' for test prompt style

    Returns:
        Dict with all metric results and overall pass/fail
    """
    print(f"\n{'='*70}")
    print("NOETIC COHERENCE REGRESSION TESTS")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Type: {model_type}")
    print(f"Test style: {test_style}")
    print(f"Thresholds:")
    print(f"  World Separation: {THRESHOLDS.WORLD_SEPARATION_THRESHOLD:.0%}")
    print(f"  RPM Variance: {THRESHOLDS.RPM_VARIANCE_THRESHOLD}")
    print(f"  Attractor Convergence: {THRESHOLDS.ATTRACTOR_CONVERGENCE_THRESHOLD:.0%}")
    print(f"  Noetic Consistency: {THRESHOLDS.NOETIC_CONSISTENCY_THRESHOLD}")
    print(f"{'='*70}")

    # Select test data based on test style
    if test_style == 'equation':
        world_texts = EQUATION_WORLD_TEXTS
        similar_pairs = EQUATION_SIMILAR_PAIRS
        dissimilar_pairs = EQUATION_DISSIMILAR_PAIRS
        rpm_test_cases = EQUATION_RPM_TEST_CASES
        attractor_texts = EQUATION_ATTRACTOR_TEXTS
    elif test_style == 'nl':
        world_texts = NL_WORLD_TEXTS
        similar_pairs = NL_SIMILAR_PAIRS
        dissimilar_pairs = NL_DISSIMILAR_PAIRS
        rpm_test_cases = NL_RPM_TEST_CASES
        attractor_texts = NL_ATTRACTOR_TEXTS
    else:  # 'both'
        combined = {}
        for world in 'ABCD':
            combined[world] = EQUATION_WORLD_TEXTS[world] + NL_WORLD_TEXTS[world]
        world_texts = combined
        similar_pairs = EQUATION_SIMILAR_PAIRS + NL_SIMILAR_PAIRS
        dissimilar_pairs = EQUATION_DISSIMILAR_PAIRS + NL_DISSIMILAR_PAIRS
        rpm_test_cases = {cat: EQUATION_RPM_TEST_CASES[cat] + NL_RPM_TEST_CASES[cat] for cat in ['desire', 'wisdom', 'power']}
        attractor_texts = EQUATION_ATTRACTOR_TEXTS + NL_ATTRACTOR_TEXTS

    # Load model
    model, tokenizer, device, actual_type = load_model(model_path, model_type=model_type)
    print(f"Model loaded on device: {device} (type: {actual_type})")

    # Compute all metrics
    metrics = compute_all_metrics(
        model, tokenizer, device,
        world_texts=world_texts,
        similar_pairs=similar_pairs,
        dissimilar_pairs=dissimilar_pairs,
        rpm_test_cases=rpm_test_cases,
        attractor_texts=attractor_texts
    )

    # Print results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    all_passed = True
    for name, result in metrics.items():
        status = "PASS" if result['passed'] else "FAIL"
        if not result['passed']:
            all_passed = False

        print(f"\n{name.upper().replace('_', ' ')}: [{status}]")

        if name == 'world_separation':
            print(f"  Correct rate: {result['correct_rate']:.1%} (threshold: {THRESHOLDS.WORLD_SEPARATION_THRESHOLD:.0%})")
        elif name == 'rpm_differentiation':
            print(f"  Variance: {result['variance']:.6f} (threshold: {THRESHOLDS.RPM_VARIANCE_THRESHOLD})")
        elif name == 'attractor_convergence':
            print(f"  Convergence rate: {result['convergence_rate']:.0%} (threshold: {THRESHOLDS.ATTRACTOR_CONVERGENCE_THRESHOLD:.0%})")
        elif name == 'noetic_consistency':
            print(f"  Separation: {result['separation']:.4f} (threshold: {THRESHOLDS.NOETIC_CONSISTENCY_THRESHOLD})")

    # Summary
    print(f"\n{'='*70}")
    print(f"OVERALL: {'PASS' if all_passed else 'FAIL'}")
    print(f"{'='*70}")

    # Save results
    results = {
        'model_path': model_path,
        'thresholds': {
            'world_separation': THRESHOLDS.WORLD_SEPARATION_THRESHOLD,
            'rpm_variance': THRESHOLDS.RPM_VARIANCE_THRESHOLD,
            'attractor_convergence': THRESHOLDS.ATTRACTOR_CONVERGENCE_THRESHOLD,
            'noetic_consistency': THRESHOLDS.NOETIC_CONSISTENCY_THRESHOLD,
        },
        'metrics': metrics,
        'all_passed': all_passed,
    }

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Noetic Coherence Regression Tests')
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--model-type', default='v2', choices=['v2', 'v4'],
                        help="Model type: 'v2' for TKSLLMCorePipeline, 'v4' for TKSNoeticLM")
    parser.add_argument('--test-style', default='both', choices=['nl', 'equation', 'both'],
                        help="Test prompt style: 'nl' for natural language (generalization), 'equation' for training format, 'both' for all")
    parser.add_argument('--output', help='Path to save results JSON')
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)

    results = run_standalone(args.model, args.output, model_type=args.model_type, test_style=args.test_style)

    # Exit with appropriate code
    sys.exit(0 if results['all_passed'] else 1)
