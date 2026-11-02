"""Temporal residual adapter on cached decoder inputs x: (B,C,T)."""
import torch
import torch.nn as nn


class TemporalAdapter(nn.Module):
    def __init__(self, channels=512, kernel=3):
        super().__init__()
        padding = kernel // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel, padding=padding, groups=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(channels, channels, kernel, padding=padding, groups=1)
        # near-identity: zero residual branch
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        nn.init.kaiming_uniform_(self.conv1.weight, a=1)
        nn.init.zeros_(self.conv1.bias)

    def forward(self, x):
        # x: (B,C,T)
        return x + self.conv2(self.act(self.conv1(x)))


def prepare_x(x, arm, adapter):
    if arm == "N":
        return x
    if arm == "T":
        return adapter(x)
    if arm == "X":
        return adapter(torch.flip(x, dims=[-1]))
    raise RuntimeError(arm)
