from torch import nn
import torch


class FilmLayer(nn.Module):
    def __init__(self, positionalEncodingSize: int, nbFeatures: int, intermediarySize=256) -> None:
        super().__init__()
        self.positionalEncodingSize = positionalEncodingSize

        self.fimStacks = nn.Sequential(
            nn.Linear(positionalEncodingSize, intermediarySize),
            nn.ReLU(),
            nn.Linear(intermediarySize, 2*nbFeatures),
        )

    def forward(self, X) -> torch.Tensor:
        return self.fimStacks(X)
