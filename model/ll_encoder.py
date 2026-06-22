import torch
import torch.nn as nn
import torch.nn.functional as F


class LLEncoder(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()


        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GELU(),
        )
        self.residual = nn.Conv2d(in_channels, out_channels, 1, bias=False)


        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        )

        sobel_kernel = torch.stack([sobel_x, sobel_y]).unsqueeze(1)


        sobel_kernel = sobel_kernel.repeat(in_channels, 1, 1, 1)
        self.register_buffer('sobel_kernel', sobel_kernel)
        self.sobel_in_ch   = in_channels
        self.sobel_out_ch  = 2 * in_channels


        self.sobel_proj = nn.Sequential(
            nn.Conv2d(self.sobel_out_ch, out_channels, 1, bias=False),
            nn.GELU(),
        )


        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
            nn.GELU(),
        )

    def forward(self, ll: torch.Tensor) -> torch.Tensor:

        cnn_feat = self.cnn(ll) + self.residual(ll)


        with torch.no_grad():
            grad = F.conv2d(ll, self.sobel_kernel, padding=1,
                            groups=self.sobel_in_ch)
        grad_feat = self.sobel_proj(grad)


        fused = self.fuse(torch.cat([cnn_feat, grad_feat], dim=1))
        return fused
