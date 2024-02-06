import torch
from torch import Tensor
import scipy
from labproject.metrics.utils import register_metric


@register_metric("wasserstein_gauss_squared")
def gaussian_squared_w2_distance(real_samples: Tensor, fake_samples: Tensor) -> Tensor:
    r"""
    Compute the squared Wasserstein distance between Gaussian approximations of real and fake samples.
    Dimensionality of the samples must be the same and >=2 (for covariance calculation).

    In detail, for each set of samples, we calculate the mean and covariance matrix.

    $$ \mu_{\text{real}} = \frac{1}{n} \sum_{i=1}^{n} x_i \qquad \mu_{\text{fake}} = \frac{1}{n} \sum_{i=1}^{n} y_i $$


    $$
    \Sigma_{\text{real}} = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \mu_{\text{real}})(x_i - \mu_{\text{real}})^T \qquad
    \Sigma_{\text{fake}} = \frac{1}{n-1} \sum_{i=1}^{n} (y_i - \mu_{\text{fake}})(y_i - \mu_{\text{fake}})^T
    $$

    Then we calculate the squared Wasserstein distance between the two Gaussian approximations:

    $$
    d_{W_2}^2(N(\mu_{\text{real}}, \Sigma_{\text{real}}), N(\mu_{\text{fake}}, \Sigma_{\text{fake}})) =
    \left\| \mu_{\text{real}} - \mu_{\text{fake}} \right\|^2 + \text{tr}(\Sigma_{\text{real}} + \Sigma_{\text{fake}} - 2 \sqrt{\Sigma_{\text{real}} \Sigma_{\text{fake}}})
    $$

    Args:
        real_samples (torch.Tensor): A tensor representing the real samples.
        fake_samples (torch.Tensor): A tensor representing the fake samples.

    Returns:
        torch.Tensor: The KL divergence between the two Gaussian approximations.

    References:
        [1] https://en.wikipedia.org/wiki/Wasserstein_metric
        [2] https://arxiv.org/pdf/1706.08500.pdf

    Examples:
        >>> real_samples = torch.randn(100, 2)  # 100 samples, 2-dimensional
        >>> fake_samples = torch.randn(100, 2)  # 100 samples, 2-dimensional
        >>> w2 = gaussian_squared_w2_distance(real_samples, fake_samples)
        >>> print(w2)
    """

    # check input (n,d only)
    assert len(real_samples.size()) == 2, "Real samples must be 2-dimensional, (n,d)"
    assert len(fake_samples.size()) == 2, "Fake samples must be 2-dimensional, (n,d)"

    # calculate mean and covariance of real and fake samples
    mu_real = real_samples.mean(dim=0)
    mu_fake = fake_samples.mean(dim=0)
    cov_real = torch.cov(real_samples.t())
    cov_fake = torch.cov(fake_samples.t())

    # ensure the covariance matrices are invertible
    eps = 1e-6
    cov_real += torch.eye(cov_real.size(0)) * eps
    cov_fake += torch.eye(cov_fake.size(0)) * eps

    # compute KL divergence
    mean_dist = torch.norm(mu_real - mu_fake, p=2)
    cov_sqrt = scipy.linalg.sqrtm((cov_real @ cov_fake).numpy())
    # print(cov_sqrt.real)
    cov_sqrt = torch.from_numpy(cov_sqrt.real)
    cov_dist = torch.trace(cov_real + cov_fake - 2 * cov_sqrt)
    w2_squared_dist = mean_dist**2 + cov_dist

    return w2_squared_dist


if __name__ == "__main__":
    # example usage
    real_samples = torch.randn(100, 2)  # 100 samples, 2-dimensional
    fake_samples = torch.randn(100, 2)  # 100 samples, 2-dimensional

    w2_dist = gaussian_squared_w2_distance(real_samples, fake_samples)
    print(w2_dist)

    # Fail case # TODO
    # real_samples = torch.randn(100, 1)  # 100 samples, 1-dimensional
    # fake_samples = torch.randn(100, 1)  # 100 samples, 1-dimensional

    # kl_div = gaussian_kl_divergence(real_samples, fake_samples)
