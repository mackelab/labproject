import torch

torch.manual_seed(0)


def random_dataset(n=1000, d=10):
    return torch.randn(n, d)
