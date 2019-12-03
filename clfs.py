import torch
from torch import nn
from torch import optim


class Baseline(nn.Module):
    def __init__(self, size):
        super().__init__()
        self._lin = nn.Linear(size, 1)
        self._optim = optim.SGD(self.parameters(), lr=0.01)

    def forward(self, batch):
        return self._lin(batch).squeeze()

