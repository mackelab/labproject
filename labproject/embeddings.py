"""
This contains embedding nets, and auxilliary functions
for extracting (N,D) embeddings from respective data and models.
"""

import torch
import torch.nn as nn
import numpy as np


# Abstract class for embedding nets
class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.embedding = None

    def forward(self, x):
        return self.embedding(x)

    def get_embedding(self, x):
        return self.forward(x).detach().cpu().numpy()

    def get_embeddings(self, dataloader):
        embeddings = []
        for batch in dataloader:
            embeddings.append(self.get_embedding(batch[0]))
        return np.concatenate(embeddings)


class FIDEmbeddingNet(EmbeddingNet):
    def __init__(self, image_model):
        super(FIDEmbeddingNet, self).__init__()
        self.embedding = image_model

    # optional override of original methods


if __name__ == "__main__":
    pass
    # test embedding nets
