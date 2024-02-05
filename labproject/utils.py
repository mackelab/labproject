import torch
import numpy as np
import random
import inspect
import os
import datetime

from omegaconf import OmegaConf

CONF_PATH = STYLE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")
)


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
        config = OmegaConf.load(CONF_PATH + f"/conf_{name}.yaml")
        config.running_user = name
    except FileNotFoundError:
        msg = f"Config file not found for {name}. Please create a config file at ../configs/conf_{name}.yaml"
        raise FileNotFoundError(msg)
    return config


def get_log_path(cfg):
    """
    Get the log path for the current experiment run.
    This log path is then used to save the numerical results of the experiment.
    Import this function in the run_{name}.py file and call it to get the log path.
    """

    # get datetime string
    now = datetime.datetime.now()
    if "exp_log_name" not in cfg:
        exp_log_name = now.strftime("%Y-%m-%d_%H-%M-%S")
    else:
        exp_log_name = cfg.exp_log_name
        # add datetime to the name
        exp_log_name = exp_log_name + "_" + now.strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(f"results/{cfg.running_user}/{exp_log_name}.pkl")
    return log_path
