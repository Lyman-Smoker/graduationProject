import torch.nn as nn
import torch
import logging

class MLP(nn.Module):
    def __init__(self, in_dim):
        super(MLP, self).__init__()
        self.regressor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        score = self.regressor(x)
        return score
