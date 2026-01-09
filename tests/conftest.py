"""
Pytest configuration for TKS-LLM tests.
"""
import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Try to import torch for fixtures
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def pytest_addoption(parser):
    """Add custom command line options for TKS tests."""
    parser.addoption(
        "--model",
        action="store",
        default=None,
        help="Path to model checkpoint for regression tests"
    )
    parser.addoption(
        "--model-type",
        action="store",
        default="v2",
        choices=["v2", "v4"],
        help="Model type: v2 (TKSLLMCorePipeline) or v4 (TKSNoeticLM)"
    )
    parser.addoption(
        "--test-style",
        action="store",
        default="nl",
        choices=["nl", "equation", "both"],
        help="Test style: nl (natural language), equation, or both"
    )
    parser.addoption(
        "--metrics-output",
        action="store",
        default=None,
        help="Path to save metrics JSON output"
    )
    parser.addoption(
        "--noetic-output",
        action="store",
        default=None,
        help="Path to save noetic metrics output"
    )


@pytest.fixture(scope="session")
def torch_device():
    """Get the appropriate torch device."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def available_model_path(request):
    """Get model path from command line or find one in common locations."""
    cli_path = request.config.getoption("--model")
    if cli_path and Path(cli_path).exists():
        return cli_path

    # Check common locations
    common_paths = [
        "output/teacher_model_v3/final_model.pt",
        "output/teacher_model_long_v4/final_model.pt",
        "output/phase5_models/final_model.pt",
        "output/models/final_model.pt",
        "checkpoints/v5_final.pt",
    ]

    for path in common_paths:
        if Path(path).exists():
            return path

    return None
