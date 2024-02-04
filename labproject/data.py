import torch
import requests
from requests.auth import HTTPBasicAuth
import os
import functools


STORAGEBOX_URL = os.getenv("HETZNER_STORAGEBOX_URL")
HETZNER_STORAGEBOX_USERNAME = os.getenv("HETZNER_STORAGEBOX_USERNAME")
HETZNER_STORAGEBOX_PASSWORD = os.getenv("HETZNER_STORAGEBOX_PASSWORD")


## Hetzner Storage Box API functions ----

DATASETS = {}


def upload_file(local_path: str, remote_path: str):
    """
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
    """
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
    """This decorator wrapps a function that should return a dataset and ensures that the dataset is a PyTorch tensor, with the correct shape.

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
        def wrapper(n: int, d: int, **kwargs):

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
    """Get a dataset by name

    Args:
        name (str): Name of the dataset
        n (int): Number of samples
        d (int): Dimensionality of the samples

    Returns:
        torch.Tensor: Dataset
    """
    assert name in DATASETS, f"Dataset {name} not found, please register it first "
    return DATASETS[name]


# ------------------------------


## Data functions ----
# This will be an arbitrary function, returning a numric array and can be registered as a dataset as follows:


@register_dataset("random")
def random_dataset(n=1000, d=10):
    return torch.randn(n, d)
