import torch.nn as nn
import torch
import os


class LN_MLP(nn.Module):
    def __init__(self, dim, use_ln=False, use_mlp=True, use_residual=False, dropout=0.6):
        super(LN_MLP, self).__init__()
        self.use_ln = use_ln
        self.use_mlp = use_mlp
        self.use_residual = use_residual
        if use_ln:
            self.ln = nn.LayerNorm(dim)
        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
                nn.Dropout(dropout)
            )

    def forward(self, feature):
        if self.use_mlp:
            return feature + self.mlp(feature)
        else:
            return feature
