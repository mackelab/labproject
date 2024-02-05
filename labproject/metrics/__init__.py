"""
Best practices for developing metrics:

1. Please do everything in torch, and if that is not possible, cast the output to torch.Tensor.
2. The function should be well-documented, including type hints.
3. The function should be tested with a simple example.
4. Add an assert at the beginning for shape checking (N,D), see examples. 
"""

from labproject.metrics.gaussian_kl import gaussian_kl_divergence
from labproject.metrics.sliced_wasserstein import sliced_wasserstein_distance

METRICS = {}
