import torch

from labproject.metrics.utils import register_metric


# NOTE: all tensors should be of shape (n_samples, n_features)


def rbf_kernel(x, y, bandwidth):
    dist = torch.cdist(x, y)
    return torch.exp(-(dist**2) / (2.0 * bandwidth**2))


def polynomial_kernel(x, y, degree, bias):
    return (x @ y.t() + bias) ** degree


def linear_kernel(x, y):
    return x @ y.t()


@register_metric("mmd_rbf")
def compute_rbf_mmd(x, y, bandwidth=1.0):
    x_kernel = rbf_kernel(x, x, bandwidth)
    y_kernel = rbf_kernel(y, y, bandwidth)
    xy_kernel = rbf_kernel(x, y, bandwidth)
    mmd = torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)
    return mmd


@register_metric("mmd_rbf_auto")
def compute_rbf_mmd_auto(x, y, bandwidth=1.0):
    dim = x.shape[1]
    x_kernel = rbf_kernel(x, x, dim * bandwidth)
    y_kernel = rbf_kernel(y, y, dim * bandwidth)
    xy_kernel = rbf_kernel(x, y, dim * bandwidth)
    mmd = torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)
    return mmd


@register_metric("mmd_polynomial")
def compute_polynomial_mmd(x, y, degree=2, bias=0):
    x_kernel = polynomial_kernel(x, x, degree, bias)
    y_kernel = polynomial_kernel(y, y, degree, bias)
    xy_kernel = polynomial_kernel(x, y, degree, bias)
    mmd = torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)
    return mmd


@register_metric("mmd_linear_naive")
def compute_linear_mmd_naive(x, y):
    x_kernel = linear_kernel(x, x)
    y_kernel = linear_kernel(y, y)
    xy_kernel = linear_kernel(x, y)
    mmd = torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)
    return mmd


@register_metric("mmd_linear")
def compute_linear_mmd(x, y):
    delta = torch.mean(x, 0) - torch.mean(y, 0)
    return torch.norm(delta, 2) ** 2
