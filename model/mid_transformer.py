import torch
import torch.nn as nn
from .restormer_block import CrossRestormerBlock

class MidFrequencyTransformer(nn.Module):
    def __init__(self, in_channels, dim, num_blocks):
        super().__init__()
        self.proj_in = nn.Conv2d(in_channels * 2, dim, 3, padding=1)
        self.blocks = nn.ModuleList([CrossRestormerBlock(dim) for _ in range(num_blocks)])
        self.proj_out = nn.Conv2d(dim, in_channels * 2, 3, padding=1)

    def forward(self, lh, hl, ll_features):
        x = torch.cat([lh, hl], dim=1)
        x = self.proj_in(x)

        for block in self.blocks:
            x = block(x, ll_features)

        x = self.proj_out(x)

        lh_restored, hl_restored = torch.chunk(x, 2, dim=1)
        return lh_restored, hl_restored

