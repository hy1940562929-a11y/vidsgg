import torch
import torch.nn as nn

class TemporalAdapter(nn.Module):
    def __init__(self, channels=256, kernel=3):
        super().__init__()
        padding = kernel // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel, padding=padding)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(channels, channels, kernel, padding=padding)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        nn.init.kaiming_uniform_(self.conv1.weight, a=1)
        nn.init.zeros_(self.conv1.bias)

    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))


def prepare_f(f, arm, adapter):
    if arm == "N":
        return f
    if arm == "T":
        return adapter(f)
    if arm == "X":
        return adapter(torch.flip(f, dims=[-1]))
    raise RuntimeError(arm)
