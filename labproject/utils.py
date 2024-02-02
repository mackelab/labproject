import torch
import numpy as np
import random
import inspect

from omegaconf import OmegaConf


def set_seed(seed: int) -> None:
    """Set seed for reproducibility

    Args:
        seed (int): Integer seed
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed

def get_cfg() -> OmegaConf:
    """This function returns the configuration file for the current experiment run.
    
    The configuration file is expected to be located at ../configs/conf_{name}.yaml, where name will match the name of the run_{name}.py file.

    Raises:
        FileNotFoundError: If the configuration file is not found

    Returns:
        OmegaConf: Dictionary with the configuration parameters
    """
    caller_frame = inspect.currentframe().f_back
    filename = caller_frame.f_code.co_filename
    name = filename.split("/")[-1].split(".")[0].split("_")[-1]
    try:
        config = OmegaConf.load(f"../configs/conf_{name}.yaml")
    except FileNotFoundError:
        msg = f"Config file not found for {name}. Please create a config file at ../configs/conf_{name}.yaml"
        raise FileNotFoundError(msg)
    return config
