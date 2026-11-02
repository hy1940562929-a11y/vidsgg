"""ActionFormer-style temporal FPN on cached f [B,C,T]=[B,256,512]."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalFPN(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv1d(channels, channels, 1) for _ in range(3)])
        self.fuse = nn.Conv1d(channels * 3, channels, 1)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)

    def forward(self, f):
        # f: [B, C, T]
        B, C, T = f.shape
        s1 = self.proj[0](f)
        s2 = self.proj[1](F.avg_pool1d(f, kernel_size=2, stride=2, ceil_mode=True))
        s4 = self.proj[2](F.avg_pool1d(f, kernel_size=4, stride=4, ceil_mode=True))
        s2 = F.interpolate(s2, size=T, mode="linear", align_corners=False)
        s4 = F.interpolate(s4, size=T, mode="linear", align_corners=False)
        return f + self.fuse(torch.cat([s1, s2, s4], dim=1))


def time_reverse_f(f):
    return torch.flip(f, dims=[-1])
