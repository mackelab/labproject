import torch
from labproject.metrics.utils import register_metric


def sinkhorn_algorithm(
    x: torch.Tensor, y: torch.Tensor, epsilon: float = 1e-3, niter: int = 1000, p: int = 2
):
    r"""Compute the sinkhorn approximation to the Wasserstein-p distance between two sets of samples.
    The sinkhorn algorithm adds a small entropy regularization term to the empirical Wasserstein distance.
    Hence this function solves the modified optimal transport problem:

    $$ \text{maximize}_{\pi \in \Pi(a, b)} \sum_\limits_{ij} \pi_{ij}c_{ij} +\epsilon\sum\limits_{ij} \log \pi_{ij}
    \text{s.t} \, \pi 1 = a, \pi^T 1 = b
    $$
    Where $\{c_{ij}\}$ is the cost matrix, $\Pi(a, b)$ is the set of joint distributions with marginals $a$ and $b$.
    In the sample-based setting, all weights $a$ and $b$ are equal to $1/n$.

    Args:
        x (torch.Tensor): tensor of samples from one distribution
        y (torch.Tensor): tensor of samples from another distribution
        epsilon (float): entropy regularization strength
        niter (int): max number of iterations
        p (int): power of distance metric

    Source: https://personal.math.ubc.ca/~geoff/courses/W2019T1/Lecture13.pdf
    Code adapted from https://github.com/gpeyre/SinkhornAutoDiff
    """

    assert len(x.shape) == 2 and len(y.shape) == 2, "x and y must be 2D"
    n, d = x.shape

    # Compute pairwise p-distances
    cost_matrix = torch.cdist(x.double(), y.double(), p=p)
    K = torch.exp(-cost_matrix / epsilon)
    a = torch.ones(n, dtype=torch.double) / n
    b = torch.ones(n, dtype=torch.double) / n

    def MC(u, v):
        r"""Modified cost for logarithmic updates on u,v
        $M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"""
        return (-cost_matrix + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon

    err = 1e6
    actual_niter = 0  # count number of iterations
    thresh = 1e-6
    u, v = torch.zeros(n, dtype=torch.double), torch.zeros(n, dtype=torch.double)

    # Sinkhorn loop
    for actual_niter in range(niter):
        u1 = u
        v1 = v
        u = epsilon * (torch.log(a) - torch.logsumexp(MC(u, v), dim=1)) + u
        v = epsilon * (torch.log(b) - torch.logsumexp(MC(u, v).T, dim=1)) + v
        err = torch.max((u - u1).abs().sum(), (v1 - v).abs().sum())
        actual_niter += 1
        if err < thresh:
            break

    U, V = u, v
    transport = torch.exp(MC(U, V))  # Transport plan pi = diag(a)*K*diag(b)
    cost = torch.sum(transport * cost_matrix)  # Sinkhorn cost

    return cost, transport


@register_metric("wasserstein_sinkhorn")
def sinkhorn_loss(
    x: torch.Tensor, y: torch.Tensor, epsilon: float = 1e-3, niter: int = 1000, p: int = 2
):
    loss, transport = sinkhorn_algorithm(x, y, epsilon, niter, p)
    return loss


if __name__ == "__main__":
    # example usage
    real_samples = torch.randn(100, 2)  # 100 samples, 2-dimensional
    fake_samples = torch.randn(100, 2)  # 100 samples, 2-dimensional

    w2_dist = sinkhorn_loss(real_samples, fake_samples)
    print(w2_dist)
