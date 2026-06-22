import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeEnhancedHHGate(nn.Module):

    def __init__(self, ll_dim: int, hh_dim: int):
        super().__init__()


        self.ms_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hh_dim, hh_dim, 3, padding=r, dilation=r, groups=hh_dim, bias=False),
                nn.Conv2d(hh_dim, hh_dim, 1, bias=False),
                nn.GELU(),
            )
            for r in (1, 2, 4)
        ])

        self.ms_fuse = nn.Sequential(
            nn.Conv2d(hh_dim * 3, hh_dim, 1, bias=False),
            nn.GELU(),
        )


        self.mask_net = nn.Sequential(
            nn.Conv2d(ll_dim, ll_dim // 2, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(ll_dim // 2, hh_dim, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hh_dim, hh_dim, 1, bias=False),
            nn.Sigmoid(),
        )


        self.res_scale = nn.Parameter(torch.ones(1, hh_dim, 1, 1) * 0.5)


        self.refine = nn.Sequential(
            nn.Conv2d(hh_dim, hh_dim, 3, padding=1, groups=hh_dim, bias=False),
            nn.Conv2d(hh_dim, hh_dim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hh_dim, hh_dim, 3, padding=1, groups=hh_dim, bias=False),
            nn.Conv2d(hh_dim, hh_dim, 1, bias=False),
        )

    def forward(self, hh: torch.Tensor, ll_features: torch.Tensor) -> torch.Tensor:

        ms_feats = torch.cat([conv(hh) for conv in self.ms_conv], dim=1)
        hh_ms = self.ms_fuse(ms_feats)


        mask = self.mask_net(ll_features)


        hh_gated = hh_ms * mask + hh * self.res_scale


        return hh_gated + self.refine(hh_gated)
