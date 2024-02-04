# STOLEN from Julius: https://github.com/mackelab/wasserstein_source/blob/main/wasser/sliced_wasserstein.py
# Removed numpy dependency

import torch
from torch import Tensor


def sliced_wasserstein_distance(
    encoded_samples: Tensor, distribution_samples: Tensor, num_projections:int=50, p:int=2, device:str="cpu"
):
    """
    Sliced Wasserstein distance between encoded samples and distribution samples

    Args:
        encoded_samples (torch.Tensor): tensor of encoded training samples
        distribution_samples (torch.Tensor): tensor drawn from the prior distribution
        num_projection (int): number of projections to approximate sliced wasserstein distance
        p (int): power of distance metric
        device (torch.device): torch device 'cpu' or 'cuda' gpu

    Return:
        torch.Tensor: Tensor of wasserstein distances of size (num_projections, 1)
    """

    embedding_dim = distribution_samples.size(-1)

    projections = rand_projections(embedding_dim, num_projections).to(device)

    encoded_projections = encoded_samples.matmul(projections.transpose(-2, -1))

    distribution_projections = distribution_samples.matmul(projections.transpose(-2, -1))

    wasserstein_distance = (
        torch.sort(encoded_projections.transpose(-2, -1), dim=-1)[0]
        - torch.sort(distribution_projections.transpose(-2, -1), dim=-1)[0]
    )

    wasserstein_distance = torch.pow(torch.abs(wasserstein_distance), p)

    # NOTE: currently computes the "squared" wasserstein distance
    # No p-th root is applied

    # return torch.pow(torch.mean(wasserstein_distance, dim=(-2, -1)), 1 / p)
    return torch.mean(wasserstein_distance, dim=(-2, -1))



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