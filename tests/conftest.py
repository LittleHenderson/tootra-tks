"""
Pytest configuration for TKS-LLM tests.
"""
import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


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
