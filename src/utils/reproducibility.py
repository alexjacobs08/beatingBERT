"""Reproducibility utilities for setting random seeds and deterministic behavior."""

import random
import os
import numpy as np
import torch


SEED = 99


def set_seed(seed: int = SEED) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed to use (default: 99)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior for CUDA operations
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set seed for transformers
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device() -> torch.device:
    """
    Get the best available device (MPS for M1, CUDA for GPU, CPU otherwise).
    
    Returns:
        torch.device: The device to use for computation
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def print_system_info() -> None:
    """Print system information for debugging."""
    print("System Information:")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  MPS available: {torch.backends.mps.is_available()}")
    print(f"  Device: {get_device()}")
    print(f"  Number of CPUs: {os.cpu_count()}")




