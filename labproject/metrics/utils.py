from typing import Optional
import torch
import functools
import warnings

METRICS = {}


def register_metric(name: str) -> callable:
    r"""This decorator wrapps a function that should return a dataset and ensures that the dataset is a PyTorch tensor, with the correct shape.

    Args:
        name (str): name of supported metric

    Returns:
        callable: metric function wrapper

    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Call the original function
            metric = func(*args, **kwargs)

            # Convert output to tensor
            metric = torch.tensor(metric)

            return metric

        METRICS[name] = wrapper
        return wrapper

    return decorator


def get_metric(name: str) -> callable:
    r"""Get a metric by name

    Args:
        name (str): Name of the metric

    Returns:
        callable: metric function

    Example:

    from labproject.metrics.utils import get_metric
    wasserstein_sinkhorn = get_metric("wasserstein_sinkhorn")
    dist = wasserstein_sinkhorn(real_samples, fake_samples, epsilon=1e-3, niter=1000, p=2)
    """
    assert name in METRICS, f"Distribution {name} not found, please register it first "
    return METRICS[name]
