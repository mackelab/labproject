import torch
import requests
from requests.auth import HTTPBasicAuth
import os

STORAGEBOX_URL = os.getenv("HETZNER_STORAGEBOX_URL")
HETZNER_STORAGEBOX_USERNAME = os.getenv("HETZNER_STORAGEBOX_USERNAME")
HETZNER_STORAGEBOX_PASSWORD = os.getenv("HETZNER_STORAGEBOX_PASSWORD")

torch.manual_seed(0)

## Hetzner Storage Box API functions ----


def upload_file(local_path, remote_path):
    """
    Uploads a file to the Hetzner Storage Box.

    Example usage:

    ```python
    if upload_file('path/to/your/local/file.txt', 'path/to/remote/file.txt'):
        print("Upload successful")
    else:
        print("Upload failed")
    ```

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

    Example usage:

    ```python
    if download_file('path/to/remote/file.txt', 'path/to/save/file.txt'):
        print("Download successful")
    else:
        print("Download failed")
    ```
    """
    url = f"{STORAGEBOX_URL}/remote.php/dav/files/{HETZNER_STORAGEBOX_USERNAME}/{remote_path}"
    auth = HTTPBasicAuth(HETZNER_STORAGEBOX_USERNAME, HETZNER_STORAGEBOX_PASSWORD)
    response = requests.get(url, auth=auth)
    if response.status_code == 200:
        with open(local_path, "wb") as f:
            f.write(response.content)
        return True
    return False


# ------------------------------


## Data functions ----
def random_dataset(n=1000, d=10):
    return torch.randn(n, d)
