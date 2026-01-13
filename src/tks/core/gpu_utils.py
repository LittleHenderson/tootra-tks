"""
TKS GPU Utilities

Provides functions to ensure GPU-only training and hardware safety.
"""

import torch
import sys
import logging

logger = logging.getLogger(__name__)

def get_target_device(force_gpu: bool = True) -> torch.device:
    """
    Returns the compute device.
    If force_gpu is True, raises an error if CUDA is not available.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"CUDA Device Detected: {torch.cuda.get_device_name(0)}")
        return device
    
    if force_gpu:
        print("\n" + "!" * 60)
        print("CRITICAL ERROR: GPU REQUIRED BUT NOT DETECTED")
        print("Exiting training to prevent CPU overheating as requested.")
        print("Check NVIDIA drivers and PyTorch CUDA installation.")
        print("!" * 60 + "\n")
        sys.exit(1)
    
    logger.warning("CUDA not available, falling back to CPU.")
    return torch.device("cpu")

def setup_cuda_optimizations():
    """Enables TF32 and other CUDA-specific speedups."""
    if torch.cuda.is_available():
        # Enable TF32 for Ampere+
        if torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("Enabled TF32 optimizations.")
        torch.backends.cudnn.benchmark = True
        logger.info("Enabled cuDNN benchmark mode.")

