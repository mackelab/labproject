from typing import Any

from torch import ones, zeros, eye, sum, Tensor, tensor, allclose, manual_seed
from torch.distributions import MultivariateNormal, Normal


def c2st_optimal(density1: Any, density2: Any, n_monte_carlo: int = 10_000) -> Tensor:
    """Return the c2st that can be achieved by an optimal classifier.

    This requires that both densities have `.log_prob()` functions.

    Args:
        density1: The first density. Must have `.sample()` and `.log_prob()`.
        density2: The second density. Must have `.sample()` and `.log_prob()`.

    Returns:
        The closed-form c2st (between 0.5 and 1.0).
    """
    assert getattr(density1, "log_prob", None), "density1 has no `.log_prob()`"
    assert getattr(density2, "log_prob", None), "density1 has no `.log_prob()`"

    d1_samples = density1.sample((n_monte_carlo // 2,))
    d2_samples = density2.sample((n_monte_carlo // 2,))

    density_ratios1 = density1.log_prob(d1_samples) >= density2.log_prob(d1_samples)
    density_ratios2 = density1.log_prob(d2_samples) < density2.log_prob(d2_samples)

    return (sum(density_ratios1) + sum(density_ratios2)) / n_monte_carlo


def test_optimal_c2st():
    """Tests the c2st on 1D Gaussians against the cdf of that Gaussian."""
    _ = manual_seed(0)
    dim = 1
    mean_diff = 4.0

    d1 = MultivariateNormal(0.0 * ones(dim), eye(dim))
    d2 = MultivariateNormal(mean_diff * ones(dim), eye(dim))

    c2st = c2st_optimal(d1, d2, 100_000)
    target = Normal(0.0, 1.0).cdf(tensor(mean_diff // 2))
    assert allclose(c2st, target, atol=1e-3)
