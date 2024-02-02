import torch
import numpy as np


def set_seed(seed: int) -> None:
    """Set seed for reproducibility

    Args:
        seed (int): Integer seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed
