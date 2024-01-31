import torch


def gaussian_kl_divergence(real_samples, fake_samples):
    """
    Compute the KL divergence between Gaussian approximations of real and fake samples.

    Args:
        real_samples (torch.Tensor): A tensor representing the real samples.
        fake_samples (torch.Tensor): A tensor representing the fake samples.

    Returns:
        float: The KL divergence between the two Gaussian approximations.
    """
    # Calculate mean and covariance of real and fake samples
    # print(real_samples.shape, fake_samples.shape)
    mu_real = real_samples.mean(dim=0)
    mu_fake = fake_samples.mean(dim=0)
    cov_real = torch.cov(real_samples.t())
    cov_fake = torch.cov(fake_samples.t())

    # Ensure the covariance matrices are invertible
    eps = 1e-8
    cov_real += torch.eye(cov_real.size(0)) * eps
    cov_fake += torch.eye(cov_fake.size(0)) * eps

    # Compute KL divergence
    inv_cov_fake = torch.inverse(cov_fake)
    kl_div = 0.5 * (
        torch.trace(inv_cov_fake @ cov_real)
        + (mu_fake - mu_real).dot(inv_cov_fake @ (mu_fake - mu_real))
        - real_samples.size(1)
        + torch.log(torch.det(cov_fake) / torch.det(cov_real))
    )

    return kl_div


if __name__ == "__main__":
    # Example usage
    real_samples = torch.randn(100, 2)  # 100 samples, 2-dimensional
    fake_samples = torch.randn(100, 2)  # 100 samples, 2-dimensional

    kl_div = gaussian_kl_divergence(real_samples, fake_samples)
    print(kl_div)
