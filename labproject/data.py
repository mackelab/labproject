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

# from torchvision.models import inception_v3
from labproject.external.inception_v3 import InceptionV3

from labproject.embeddings import FIDEmbeddingNet

import warnings


STORAGEBOX_URL = os.getenv("HETZNER_STORAGEBOX_URL")
HETZNER_STORAGEBOX_USERNAME = os.getenv("HETZNER_STORAGEBOX_USERNAME")
HETZNER_STORAGEBOX_PASSWORD = os.getenv("HETZNER_STORAGEBOX_PASSWORD")

IMAGENET_UNCONDITIONAL_MODEL_EMBEDDING = (
    "https://drive.google.com/uc?id=1xsGlNig7pCQuMpsvN86hgTGLEDGi6fVD"
)
IMAGENET_CONDITIONAL_MODEL = "https://drive.google.com/uc?id=1FBVFiFcWnVs4i_LK4lAUemx83D7Hb_tU"
IMAGENET_TEST_EMBEDDING = "https://drive.google.com/uc?id=12B5Nkjr611WhXUafv08BciW7nsZ20Dfc"
IMAGENET_VALIDATION_EMBEDDING = "https://drive.google.com/uc?id=1Chc2ygs-Akw0Hlq-Nx7ykF2fp3SqV_aM"


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
            if d is not None:
                assert d > 0, "d must be a positive integer"
            else:
                warnings.warn("d is not specified, make sure you know what you're doing!")

            # Call the original function
            if d is not None:
                dataset = func(n, d, **kwargs)
            else:
                dataset = func(n, **kwargs)
            if isinstance(dataset, tuple):
                dataset = tuple(
                    torch.Tensor(data) if not isinstance(data, torch.Tensor) else data
                    for data in dataset
                )
            else:
                dataset = (
                    torch.Tensor(dataset) if not isinstance(dataset, torch.Tensor) else dataset
                )
            if d is not None:
                assert dataset.shape == (n, d), f"Dataset shape must be {(n, d)}"
            else:
                assert dataset.shape[0] == n, f"Dataset shape must be {(n, ...)}"

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
    n: int,
    save_path="data",
    train=True,
    batch_size=100,
    shuffle=False,
    num_workers=1,
    device="cpu",
    return_labels=False,
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
            transforms.ToTensor(),
        ]
    )
    cifar10 = CIFAR10(root=save_path, train=train, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        cifar10, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    dataset_subset = Subset(dataloader.dataset, range(n))
    dataloader = DataLoader(
        dataset_subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    net = FIDEmbeddingNet(device=device)
    if return_labels:
        embeddings, labels = net.get_embeddings_with_labels(dataloader)
        return embeddings, labels
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
    return torch.distributions.Normal(0, 1)


@register_distribution("normal")
def normal_distribution():
    return torch.distributions.Normal(0, 1)


@register_dataset("multivariate_normal")
def multivariate_normal(n=3000, dims=100, means=None, vars=None, distort=None):
    if means is None:
        means = torch.zeros(dims)
    else:
        assert (
            len(means) == dims
        ), "The length of the means vector must be equal to the number of dimensions"
    if vars is None:
        vars = torch.eye(dims)
    else:
        assert (
            len(vars) == dims
        ), "The length of the vars vector must be equal to the number of dimensions"
        vars = torch.diag(vars)

    samples = torch.distributions.MultivariateNormal(means, vars).sample((n,))
    if distort == "shift_all":
        shift = 1
        samples = samples + shift
    elif distort == "shift_one":
        idx = 0
        shift = torch.zeros(n) + 1
        samples[:, idx] = samples[:, idx] + shift
    print(f"First 5 rows of dataset distorted: {samples[:5, :5]}")
    return samples


@register_dataset("toy_2d")
def toy_2d(n=1000, d=2):
    return toy_mog_2d().sample(n)


@register_distribution("toy_2d")
def toy_mog_2d():
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
            if isinstance(sample_shape, int):
                sample_shape = (sample_shape,)
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
def cifar10_train(n=1000, d=2048, save_path="data", device="cpu", return_labels=False):

    assert d is None or d == 2048, "The dimensionality of the embeddings must be 2048"

    embeddings = load_cifar10(
        n, save_path=save_path, train=True, device=device, return_labels=return_labels
    )
    if return_labels:
        embeddings, labels = embeddings
    # to cpu if necessary
    if device == "cuda":
        embeddings = embeddings.cpu()

    max_n = embeddings.shape[0]

    assert n <= max_n, f"Requested {n} samples, but only {max_n} are available"

    if return_labels:
        return embeddings[:n], labels[:n]

    return embeddings[:n]


@register_dataset("cifar10_test")
def cifar10_test(n=1000, d=2048, save_path="data", device="cpu", return_labels=False):

    assert d == 2048, "The dimensionality of the embeddings must be 2048"

    assert d is None or d == 2048, "The dimensionality of the embeddings must be 2048"

    embeddings = load_cifar10(
        n, save_path=save_path, train=False, device=device, return_labels=return_labels
    )
    if return_labels:
        embeddings, labels = embeddings
    # to cpu if necessary
    if device == "cuda":
        embeddings = embeddings.cpu()

    max_n = embeddings.shape[0]

    assert n <= max_n, f"Requested {n} samples, but only {max_n} are available"

    if return_labels:
        return embeddings[:n], labels[:n]

    return embeddings[:n]


# TODO: Maybe chang
@register_dataset("imagenet_real_embeddings")
def imagenet_real_embeddings(n=1000, d=2048):

    assert d == 2048, "The dimensionality of the embeddings must be 2048"

    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/ImageNet/"
    )
    if not os.path.exists(os.path.join(data_dir, "imagenet_test_dataset_embeddings.pt")):
        raise FileNotFoundError(
            f"No file `data/ImageNet/imagenet_test_dataset_embeddings.pt` found"
        )
    embeddings = torch.load(os.path.join(data_dir, "imagenet_test_dataset_embeddings.pt"))

    max_n = embeddings.shape[0]
    assert n <= max_n, f"Requested {n} samples, but only {max_n} are available"

    return embeddings[:n]


@register_dataset("imagenet_uncond_embeddings")
def imagenet_uncond_embeddings(n=1000, d=2048):

    assert d == 2048, "The dimensionality of the embeddings must be 2048"

    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/ImageNet/"
    )
    if not os.path.exists(
        os.path.join(data_dir, "samples_50k_unconditional_moresteps_embeddings.pt")
    ):
        raise FileNotFoundError(
            f"No file `data/ImageNet/samples_50k_unconditional_moresteps_embeddings.pt` found"
        )
    embeddings = torch.load(
        os.path.join(data_dir, "samples_50k_unconditional_moresteps_embeddings.pt")
    )

    max_n = embeddings.shape[0]
    assert n <= max_n, f"Requested {n} samples, but only {max_n} are available"

    return embeddings[:n]


# @register_dataset("imagenet_cond_embeddings")
# def imagenet_cond_embeddings(keep_labels=[]):
#     data_dir = os.path.join(os.path.abspath(__file__).parent.parent, "data/")
#     if not os.path.exists(os.path.join(data_dir, "samples_50k_unconditional_moresteps_embeddings.pt")):
#         raise FileNotFoundError(f"No file `data/samples_50k_unconditional_moresteps_embeddings.pt` found")
#     data = torch.load(os.path.join(data_dir, "samples_50k_unconditional_moresteps_embeddings.pt"))
@register_dataset("imagenet_unconditional_model_embedding")
def imagenet_unconditional_model_embedding(n, d=2048, device="cpu", save_path="data"):
    r"""Get the unconditional model embeddings for ImageNet

    Args:
        n (int): Number of samples
        d (int, optional): Dimensionality of the embeddings. Defaults to 2048.
        device (str, optional): Device. Defaults to "cpu".

    Returns:
        torch.Tensor: ImageNet embeddings
    """
    assert d == 2048, "The dimensionality of the embeddings must be 2048"
    if not os.path.exists("imagenet_unconditional_model_embedding.pt"):
        import gdown

        gdown.download(
            IMAGENET_UNCONDITIONAL_MODEL_EMBEDDING,
            "imagenet_unconditional_model_embedding.pt",
            quiet=False,
        )
    unconditional_embeddigns = torch.load("imagenet_unconditional_model_embedding.pt")

    max_n = unconditional_embeddigns.shape[0]

    assert n <= max_n, f"Requested {n} samples, but only {max_n} are available"

    return unconditional_embeddigns[:n]


@register_dataset("imagenet_test_embedding")
def imagenet_test_embedding(n, d=2048, device="cpu", save_path="data"):
    r"""Get the test embeddings for ImageNet

    Args:
        n (int): Number of samples
        d (int, optional): Dimensionality of the embeddings. Defaults to 2048.
        device (str, optional): Device. Defaults to "cpu".

    Returns:
        torch.Tensor: ImageNet embeddings
    """
    assert d == 2048, "The dimensionality of the embeddings must be 2048"
    if not os.path.exists("imagenet_test_embedding.pt"):
        import gdown

        gdown.download(IMAGENET_TEST_EMBEDDING, "imagenet_test_embedding.pt", quiet=False)
    test_embeddigns = torch.load("imagenet_test_embedding.pt")

    max_n = test_embeddigns.shape[0]

    assert n <= max_n, f"Requested {n} samples, but only {max_n} are available"

    return test_embeddigns[:n]


@register_dataset("imagenet_validation_embedding")
def imagenet_validation_embedding(n, d=2048, device="cpu", save_path="data"):
    r"""Get the validation embeddings for ImageNet

    Args:
        n (int): Number of samples
        d (int, optional): Dimensionality of the embeddings. Defaults to 2048.
        device (str, optional): Device. Defaults to "cpu".

    Returns:
        torch.Tensor: ImageNet embeddings
    """
    assert d == 2048, "The dimensionality of the embeddings must be 2048"
    if not os.path.exists("imagenet_validation_embedding.pt"):
        import gdown

        gdown.download(
            IMAGENET_VALIDATION_EMBEDDING, "imagenet_validation_embedding.pt", quiet=False
        )
    validation_embeddigns = torch.load("imagenet_validation_embedding.pt")

    max_n = validation_embeddigns.shape[0]

    assert n <= max_n, f"Requested {n} samples, but only {max_n} are available"

    return validation_embeddigns[:n]


@register_dataset("imagenet_conditional_model")
def imagenet_conditional_model(
    n, d=2048, label: Optional[int] = None, device="cpu", save_path="data"
):
    r"""Get the conditional model embeddings for ImageNet

    Args:
        n (int): Number of samples
        d (int, optional): Dimensionality of the embeddings. Defaults to 2048.
        label (int, optional): Label, if None it takes random samples. Defaults to None.
        device (str, optional): Device. Defaults to "cpu".

    Returns:
        torch.Tensor: ImageNet embeddings
    """
    assert d == 2048, "The dimensionality of the embeddings must be 2048"
    if not os.path.exists("imagenet_conditional_model.pt"):
        import gdown

        gdown.download(IMAGENET_CONDITIONAL_MODEL, "imagenet_conditional_model.pt", quiet=False)
    conditional_embeddings = torch.load("imagenet_conditional_model.pt")

    if label is not None:
        conditional_embeddings = conditional_embeddings[label]
    else:
        conditional_embeddings = conditional_embeddings.flatten(0, 1)
        conditional_embeddings = conditional_embeddings[
            torch.randperm(conditional_embeddings.shape[0])
        ]

    max_n = conditional_embeddings.shape[0]

    assert n <= max_n, f"Requested {n} samples, but only {max_n} are available"

    return conditional_embeddings[:n]
