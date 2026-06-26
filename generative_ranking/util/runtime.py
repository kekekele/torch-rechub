import os
import random

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None


def set_seed(seed):
    if torch is not None:
        torch.manual_seed(seed)
    if np is not None:
        np.random.seed(seed)
    random.seed(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path