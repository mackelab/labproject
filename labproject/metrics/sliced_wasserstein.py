# this implementation is from https://github.com/mackelab/sourcerer/blob/main/sourcerer/sliced_wasserstein.py
# Removed numpy dependency

import torch
from torch import Tensor
from labproject.metrics.utils import register_metric


@register_metric("sliced_wasserstein")
def sliced_wasserstein_distance(
    encoded_samples: Tensor,
    distribution_samples: Tensor,
    num_projections: int = 50,
    p: int = 2,
    device: str = "cpu",
) -> Tensor:
    """
    Sliced Wasserstein distance between encoded samples and distribution samples.
    Note that the SWD does not converge to the true Wasserstein distance, but rather it is a different proper distance metric.

    Args:
        encoded_samples (torch.Tensor): tensor of encoded training samples
        distribution_samples (torch.Tensor): tensor drawn from the prior distribution
        num_projection (int): number of projections to approximate sliced wasserstein distance
        p (int): power of distance metric
        device (torch.device): torch device 'cpu' or 'cuda' gpu

    Return:
        torch.Tensor: Tensor of wasserstein distances of size (num_projections, 1)
    """

    # check input (n,d only)
    assert len(encoded_samples.size()) == 2, "Real samples must be 2-dimensional, (n,d)"
    assert len(distribution_samples.size()) == 2, "Fake samples must be 2-dimensional, (n,d)"

    embedding_dim = distribution_samples.size(-1)

    projections = rand_projections(embedding_dim, num_projections).to(device)

    encoded_projections = encoded_samples.matmul(projections.transpose(-2, -1))

    distribution_projections = distribution_samples.matmul(projections.transpose(-2, -1))

    wasserstein_distance = (
        torch.sort(encoded_projections.transpose(-2, -1), dim=-1)[0]
        - torch.sort(distribution_projections.transpose(-2, -1), dim=-1)[0]
    )

    wasserstein_distance = torch.pow(torch.abs(wasserstein_distance), p)

    return torch.pow(torch.mean(wasserstein_distance, dim=(-2, -1)), 1 / p)


def rand_projections(embedding_dim: int, num_samples: int):
    """
    This function generates num_samples random samples from the latent space's unti sphere.r

    Args:
        embedding_dim (int): dimention of the embedding
        sum_samples (int): number of samples

    Return :
        torch.tensor: tensor of size (num_samples, embedding_dim)
    """

    ws = torch.randn((num_samples, embedding_dim))
    projection = ws / torch.norm(ws, dim=-1, keepdim=True)
    return projection


if __name__ == "__main__":
    # Test
    # Generate random samples
    samples1 = torch.randn(100, 2)
    samples2 = torch.randn(100, 2)

    # Compute sliced wasserstein distance
    sw_distance = sliced_wasserstein_distance(samples1, samples2)
    print(sw_distance)
