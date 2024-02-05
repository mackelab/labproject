from typing import Optional
import torch
import requests
from requests.auth import HTTPBasicAuth
import os
import functools

from torch.distributions import MultivariateNormal, Categorical

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import inception_v3

from labproject.embeddings import FIDEmbeddingNet


STORAGEBOX_URL = os.getenv("HETZNER_STORAGEBOX_URL")
HETZNER_STORAGEBOX_USERNAME = os.getenv("HETZNER_STORAGEBOX_USERNAME")
HETZNER_STORAGEBOX_PASSWORD = os.getenv("HETZNER_STORAGEBOX_PASSWORD")


## Hetzner Storage Box API functions ----

DATASETS = {}
DISTRIBUTIONS = {}


def upload_file(local_path: str, remote_path: str):
    r"""
    Uploads a file to the Hetzner Storage Box.

    Args:
        local_path (str): The path to the local file to be uploaded.
        remote_path (str): The path where the file should be uploaded on the remote server.

    Returns:
        bool: True if the upload is successful, False otherwise.

    Example:
        >>> if upload_file('path/to/your/local/file.txt', 'path/to/remote/file.txt'):
        >>>     print("Upload successful")
        >>> else:
        >>>     print("Upload failed")
    """
    url = f"{STORAGEBOX_URL}/remote.php/dav/files/{HETZNER_STORAGEBOX_USERNAME}/{remote_path}"
    auth = HTTPBasicAuth(HETZNER_STORAGEBOX_USERNAME, HETZNER_STORAGEBOX_PASSWORD)
    with open(local_path, "rb") as f:
        data = f.read()
    response = requests.put(url, data=data, auth=auth)
    return response.status_code == 201


def download_file(remote_path, local_path):
    r"""
    Downloads a file from the Hetzner Storage Box.

    Args:
        remote_path (str): The path to the remote file to be downloaded.
        local_path (str): The path where the file should be saved locally.

    Returns:
        bool: True if the download is successful, False otherwise.

    Example:
        >>> if download_file('path/to/remote/file.txt', 'path/to/save/file.txt'):
        >>>     print("Download successful")
        >>> else:
        >>>     print("Download failed")
    """
    url = f"{STORAGEBOX_URL}/remote.php/dav/files/{HETZNER_STORAGEBOX_USERNAME}/{remote_path}"
    auth = HTTPBasicAuth(HETZNER_STORAGEBOX_USERNAME, HETZNER_STORAGEBOX_PASSWORD)
    response = requests.get(url, auth=auth)
    if response.status_code == 200:
        with open(local_path, "wb") as f:
            f.write(response.content)
        return True
    return False


def register_dataset(name: str) -> callable:
    r"""This decorator wrapps a function that should return a dataset and ensures that the dataset is a PyTorch tensor, with the correct shape.

    Args:
        func (callable): Dataset generator function

    Returns:
        callable: Dataset generator function wrapper

    Example:
        >>> @register_dataset("random")
        >>> def random_dataset(n=1000, d=10):
        >>>     return torch.randn(n, d)
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(n: int, d: Optional[int] = None, **kwargs):

            assert n > 0, "n must be a positive integer"
            assert d > 0, "d must be a positive integer"

            # Call the original function
            dataset = func(n, d, **kwargs)

            # Convert the dataset to a PyTorch tensor
            dataset = torch.Tensor(dataset) if not isinstance(dataset, torch.Tensor) else dataset

            assert dataset.shape == (n, d), f"Dataset shape must be {(n, d)}"

            return dataset

        DATASETS[name] = wrapper
        return wrapper

    return decorator


def get_dataset(name: str) -> torch.Tensor:
    r"""Get a dataset by name

    Args:
        name (str): Name of the dataset
        n (int): Number of samples
        d (int): Dimensionality of the samples

    Returns:
        torch.Tensor: Dataset
    """
    assert name in DATASETS, f"Dataset {name} not found, please register it first "
    return DATASETS[name]


def register_distribution(name: str) -> callable:
    r"""This decorator wrapps a function that should return a dataset and ensures that the dataset is a PyTorch tensor, with the correct shape.

    Args:
        func (callable): Dataset generator function

    Returns:
        callable: Dataset generator function wrapper

    Example:
        >>> @register_dataset("random")
        >>> def random_dataset(n=1000, d=10):
        >>>     return torch.randn(n, d)
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Call the original function
            distribution = func(*args, **kwargs)
            return distribution

        DISTRIBUTIONS[name] = wrapper
        return wrapper

    return decorator


def get_distribution(name: str) -> torch.Tensor:
    r"""Get a distribution by name

    Args:
        name (str): Name of the distribution

    Returns:
        torch.Tensor: Distribution
    """
    assert name in DISTRIBUTIONS, f"Distribution {name} not found, please register it first "
    return DISTRIBUTIONS[name]


def load_cifar10(
    n: int, save_path="data", train=True, batch_size=100, shuffle=False, num_workers=1, device="cpu"
) -> torch.Tensor:
    """Load a subset of cifar10

    Args:
        n (int): Number of samples to load
        save_path (str, optional): Path to save files. Defaults to "data".
        train (bool, optional): Train or test. Defaults to True.
        batch_size (int, optional): Batch size. Defaults to 100.
        shuffle (bool, optional): Shuffle. Defaults to False.
        num_workers (int, optional): Parallel workers. Defaults to 1.
        device (str, optional): Device. Defaults to "cpu".

    Returns:
        torch.Tensor: Cifar10 embeddings
    """
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            # normalize specific to inception model
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # Move to GPU if available
            transforms.Lambda(lambda x: x.to(device if torch.cuda.is_available() else "cpu")),
        ]
    )
    cifar10 = CIFAR10(root=save_path, train=train, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        cifar10, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    dataloader_subset = Subset(dataloader.dataset, range(n))
    dataloader = DataLoader(
        dataloader_subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    model = inception_v3(pretrained=True)
    model.fc = torch.nn.Identity()  # replace the classifier with identity to get features
    model.eval()
    model = model.to(device if torch.cuda.is_available() else "cpu")
    net = FIDEmbeddingNet(model)
    embeddings = net.get_embeddings(dataloader)
    return embeddings


# ------------------------------


## Data functions ----
# This will be an arbitrary function, returning a numric array and can be registered as a dataset as follows:


@register_dataset("random")
def random_dataset(n=1000, d=10):
    return torch.randn(n, d)

@register_distribution("normal")
def normal_distribution():
    return torch.distributions.Normal(0,1)


@register_distribution("normal")
def normal_distribution():
    return torch.distributions.Normal(0, 1)


@register_distribution("toy_2d")
def normal_distribution():
    class Toy2D:
        def __init__(self):
            self.means = torch.tensor(
                [
                    [0.0, 0.5],
                    [-3.0, -0.5],
                    [0.0, -1.0],
                    [-4.0, -3.0],
                ]
            )
            self.covariances = torch.tensor(
                [
                    [[1.0, 0.8], [0.8, 1.0]],
                    [[1.0, -0.5], [-0.5, 1.0]],
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[0.5, 0.0], [0.0, 0.5]],
                ]
            )
            self.weights = torch.tensor([0.2, 0.3, 0.3, 0.2])

            # Create a list of 2D Gaussian distributions
            self.gaussians = [
                MultivariateNormal(mean, covariance)
                for mean, covariance in zip(self.means, self.covariances)
            ]

        def sample(self, sample_shape):
            # Sample from the mixture
            categorical = Categorical(self.weights)
            sample_indices = categorical.sample(sample_shape)
            return torch.stack([self.gaussians[i].sample() for i in sample_indices])

        def log_prob(self, input):
            probs = torch.stack([g.log_prob(input).exp() for g in self.gaussians])
            probs = probs.T * self.weights
            return torch.sum(probs, dim=1).log()

    return Toy2D()


@register_dataset("cifar10_train")
def cifar10_train(n=1000, d=10, save_path="data", device="cpu"):

    assert d is None or d == 2048, "The dimensionality of the embeddings must be 2048"

    embeddings = load_cifar10(n, save_path=save_path, train=True, device=device)

    max_n = embeddings.shape[0]

    assert n <= max_n, f"Requested {n} samples, but only {max_n} are available"

    return embeddings[:n]


@register_dataset("cifar10_test")
def cifar10_test(n=1000, d=2048, save_path="data", device="cpu"):

    assert d == 2048, "The dimensionality of the embeddings must be 2048"

    embeddings = load_cifar10(n, save_path=save_path, train=False, device=device)

    max_n = embeddings.shape[0]

    assert n <= max_n, f"Requested {n} samples, but only {max_n} are available"

    return embeddings[:n]
