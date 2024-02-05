import torch
from torch import Tensor


def gaussian_kl_divergence(real_samples: Tensor, fake_samples: Tensor) -> Tensor:
    r"""
    Compute the KL divergence between Gaussian approximations of real and fake samples.
    Dimensionality of the samples must be the same and >=2 (for covariance calculation).

    In detail, for each set of samples, we calculate the mean and covariance matrix.

    $$ \mu_{\text{real}} = \frac{1}{n} \sum_{i=1}^{n} x_i \qquad \mu_{\text{fake}} = \frac{1}{n} \sum_{i=1}^{n} y_i $$


    $$
    \Sigma_{\text{real}} = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \mu_{\text{real}})(x_i - \mu_{\text{real}})^T \qquad
    \Sigma_{\text{fake}} = \frac{1}{n-1} \sum_{i=1}^{n} (y_i - \mu_{\text{fake}})(y_i - \mu_{\text{fake}})^T
    $$

    Then we calculate the KL divergence between the two Gaussian approximations:

    $$
    D_{KL}(N(\mu_{\text{real}}, \Sigma_{\text{real}}) || N(\mu_{\text{fake}}, \Sigma_{\text{fake}})) =
    \frac{1}{2} \left( \text{tr}(\Sigma_{\text{fake}}^{-1} \Sigma_{\text{real}}) + (\mu_{\text{fake}} - \mu_{\text{real}})^T \Sigma_{\text{fake}}^{-1} (\mu_{\text{fake}} - \mu_{\text{real}})
    - k + \log \frac{|\Sigma_{\text{fake}}|}{|\Sigma_{\text{real}}|} \right)
    $$

    Args:
        real_samples (torch.Tensor): A tensor representing the real samples.
        fake_samples (torch.Tensor): A tensor representing the fake samples.

    Returns:
        torch.Tensor: The KL divergence between the two Gaussian approximations.

    Examples:
        >>> real_samples = torch.randn(100, 2)  # 100 samples, 2-dimensional
        >>> fake_samples = torch.randn(100, 2)  # 100 samples, 2-dimensional
        >>> kl_div = gaussian_kl_divergence(real_samples, fake_samples)
        >>> print(kl_div)
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
    eps = 1e-8
    cov_real += torch.eye(cov_real.size(0)) * eps
    cov_fake += torch.eye(cov_fake.size(0)) * eps

    # compute KL divergence
    inv_cov_fake = torch.inverse(cov_fake)
    kl_div = 0.5 * (
        torch.trace(inv_cov_fake @ cov_real)
        + (mu_fake - mu_real).dot(inv_cov_fake @ (mu_fake - mu_real))
        - real_samples.size(1)
        + torch.log(torch.det(cov_fake) / torch.det(cov_real))
    )

    return kl_div


if __name__ == "__main__":
    # example usage
    real_samples = torch.randn(100, 2)  # 100 samples, 2-dimensional
    fake_samples = torch.randn(100, 2)  # 100 samples, 2-dimensional

    kl_div = gaussian_kl_divergence(real_samples, fake_samples)
    print(kl_div)

    # Fail case # TODO
    # real_samples = torch.randn(100, 1)  # 100 samples, 1-dimensional
    # fake_samples = torch.randn(100, 1)  # 100 samples, 1-dimensional

    # kl_div = gaussian_kl_divergence(real_samples, fake_samples)
