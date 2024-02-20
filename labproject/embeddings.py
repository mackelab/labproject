"""
This contains embedding nets, and auxilliary functions
for extracting (N,D) embeddings from respective data and models.
"""

import torch
import torch.nn as nn
import numpy as np

from labproject.external.inception_v3 import InceptionV3, get_inception_v3_activations


# Abstract class for embedding nets
class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.embedding_net = None

    def forward(self, x):
        return self.embedding_net(x)

    def get_embedding(self, x):
        raise NotImplementedError("Subclasses must implement this method")

    def get_embeddings(self, dataloader):
        raise NotImplementedError("Subclasses must implement this method")


class FIDEmbeddingNet(EmbeddingNet):
    def __init__(self, image_model=None, device="cpu"):
        super(FIDEmbeddingNet, self).__init__()
        self.embedding_net = image_model if image_model is not None else InceptionV3()
        self.device = device
        self.embedding_net = self.embedding_net.to(device)

    def get_embeddings(self, dataloader):
        embeddings = []
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            if isinstance(batch, dict):
                batch = batch["image"]
            embeddings += [get_inception_v3_activations(self.embedding_net, batch.to(self.device))]
        embeddings = torch.cat(embeddings, dim=0)
        return embeddings

    def get_embeddings_with_labels(self, dataloader):
        embeddings = []
        labels = []
        for batch in dataloader:
            embeddings += [
                get_inception_v3_activations(self.embedding_net, batch[0].to(self.device))
            ]
            labels += [batch[1]]
        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels, dim=0)
        return embeddings, labels

    # optional override of original methods


if __name__ == "__main__":
    pass
    # test embedding nets
