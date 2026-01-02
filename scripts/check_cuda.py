#!/usr/bin/env python3
"""
CUDA Environment Checker

Checks your system for CUDA compatibility and provides
installation instructions if needed.

Usage:
    python scripts/check_cuda.py
"""

import sys
import subprocess
import shutil
import re
from pathlib import Path


def check_cuda():
    """Check CUDA environment and provide detailed report."""
    print("=" * 70)
    print("TKS CUDA Environment Check")
    print("=" * 70)
    print()

    # Check Python version
    print(f"Python Version: {sys.version}")
    print()

    # Check PyTorch
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")

        # CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_available}")

        if cuda_available:
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"GPU Count: {torch.cuda.device_count()}")
            print()

            # Device details
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {props.name}")
                print(f"  - Total Memory: {props.total_memory / 1e9:.2f} GB")
                print(f"  - Compute Capability: {props.major}.{props.minor}")
                print(f"  - Multi-Processors: {props.multi_processor_count}")

                # Feature support
                if props.major >= 8:
                    print(f"  - BF16 Support: Yes (Ampere+)")
                    print(f"  - TF32 Support: Yes")
                elif props.major >= 7:
                    print(f"  - BF16 Support: Limited (Volta/Turing)")
                    print(f"  - Tensor Cores: Yes")
                else:
                    print(f"  - BF16 Support: No")

            print()

            # Memory test
            print("Running memory test...")
            try:
                # Allocate test tensor
                test_size = 1024 * 1024 * 256  # 256MB
                x = torch.randn(test_size, device='cuda')
                del x
                torch.cuda.empty_cache()
                print("  Memory allocation: OK")

                # Matrix multiplication test
                a = torch.randn(1000, 1000, device='cuda')
                b = torch.randn(1000, 1000, device='cuda')
                c = torch.mm(a, b)
                del a, b, c
                torch.cuda.empty_cache()
                print("  Matrix operations: OK")

                # Mixed precision test
                with torch.cuda.amp.autocast():
                    x = torch.randn(1000, 1000, device='cuda')
                    y = x @ x.T
                del x, y
                torch.cuda.empty_cache()
                print("  Mixed precision: OK")

                print()
                print("✓ CUDA is properly configured!")

            except Exception as e:
                print(f"  ERROR: {e}")
                print()
                print("✗ CUDA test failed")

        else:
            print()
            print("CUDA is not available. Possible reasons:")
            print("  1. No NVIDIA GPU installed")
            print("  2. NVIDIA drivers not installed")
            print("  3. PyTorch installed without CUDA support")
            print()
            print("To install PyTorch with CUDA support:")
            print()
            print("  For CUDA 12.1:")
            print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            print()
            print("  For CUDA 11.8:")
            print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

    except ImportError:
        print("PyTorch is not installed!")
        print()
        print("To install PyTorch with CUDA support:")
        print()
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

    # Check NVIDIA driver
    print()
    print("-" * 70)
    print("NVIDIA Driver Check")
    print("-" * 70)

    try:
        nvidia_smi = shutil.which('nvidia-smi')
        if not nvidia_smi and Path('/usr/lib/wsl/lib/nvidia-smi').exists():
            nvidia_smi = '/usr/lib/wsl/lib/nvidia-smi'

        if not nvidia_smi:
            raise FileNotFoundError("nvidia-smi not found in PATH (and /usr/lib/wsl/lib/nvidia-smi not present)")

        result = subprocess.run(
            [nvidia_smi, '--query-gpu=driver_version,name,memory.total', '--format=csv,noheader'],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("nvidia-smi output:")
            for line in result.stdout.strip().split('\n'):
                parts = line.split(',')
                if len(parts) >= 3:
                    print(f"  Driver: {parts[0].strip()}")
                    print(f"  GPU: {parts[1].strip()}")
                    print(f"  Memory: {parts[2].strip()}")
        else:
            print("nvidia-smi failed - NVIDIA drivers may not be installed")

        # Show and parse CUDA version line for compatibility hints
        full = subprocess.run([nvidia_smi], capture_output=True, text=True)
        if full.returncode == 0:
            m = re.search(r'CUDA Version:\\s*([0-9]+\\.[0-9]+)', full.stdout)
            driver_cuda = m.group(1) if m else None
            if driver_cuda:
                print(f"  Driver CUDA (max): {driver_cuda}")

                try:
                    import torch
                    torch_cuda = torch.version.cuda
                    if torch_cuda:
                        def _v(s: str):
                            major, minor = s.split('.')[:2]
                            return int(major), int(minor)
                        if _v(torch_cuda) > _v(driver_cuda):
                            print()
                            print("WARNING: Your PyTorch CUDA runtime is newer than your driver supports.")
                            print(f"  PyTorch CUDA: {torch_cuda}")
                            print(f"  Driver CUDA:  {driver_cuda}")
                            print("Fix: either update the NVIDIA driver, or install a torch wheel built for an older CUDA (e.g. cu126/cu124).")
                except Exception:
                    pass
    except FileNotFoundError:
        print("nvidia-smi not found - NVIDIA drivers may not be installed")
    except Exception as e:
        print(f"Error checking NVIDIA driver: {e}")

    print()
    print("=" * 70)
    print("Recommendations for TKS Training")
    print("=" * 70)
    print()

    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            mem_gb = props.total_memory / 1e9

            print("Based on your GPU configuration:")
            print()

            if mem_gb >= 24:
                print(f"  GPU Memory: {mem_gb:.0f}GB - Excellent!")
                print("  Recommended batch size: 32-64")
                print("  Recommended hidden dim: 512")
                print("  Recommended layers: 8-12")
            elif mem_gb >= 12:
                print(f"  GPU Memory: {mem_gb:.0f}GB - Good")
                print("  Recommended batch size: 16-32")
                print("  Recommended hidden dim: 256-384")
                print("  Recommended layers: 4-8")
            elif mem_gb >= 8:
                print(f"  GPU Memory: {mem_gb:.0f}GB - Moderate")
                print("  Recommended batch size: 8-16")
                print("  Recommended hidden dim: 256")
                print("  Recommended layers: 4-6")
            else:
                print(f"  GPU Memory: {mem_gb:.0f}GB - Limited")
                print("  Recommended batch size: 4-8")
                print("  Recommended hidden dim: 128-256")
                print("  Recommended layers: 2-4")
                print("  Consider using --gradient-checkpointing")

            print()
            print("Sample training command:")
            print()
            if mem_gb >= 12:
                print("  python scripts/train_cuda.py \\")
                print("      --data output/teacher_augmented.jsonl \\")
                print("      --epochs 10 \\")
                print("      --batch-size 16 \\")
                print("      --hidden-dim 256 \\")
                print("      --use-bf16")
            else:
                print("  python scripts/train_cuda.py \\")
                print("      --data output/teacher_augmented.jsonl \\")
                print("      --epochs 10 \\")
                print("      --batch-size 8 \\")
                print("      --hidden-dim 128 \\")
                print("      --gradient-checkpointing")

    except:
        print("Install PyTorch with CUDA to see recommendations")

    print()


if __name__ == "__main__":
    check_cuda()
