
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_sobel_kernels(device, dtype):
    kx = torch.tensor(
        [[-1., 0., 1.],
         [-2., 0., 2.],
         [-1., 0., 1.]],
        device=device, dtype=dtype,
    ).view(1, 1, 3, 3)

    ky = torch.tensor(
        [[-1., -2., -1.],
         [ 0.,  0.,  0.],
         [ 1.,  2.,  1.]],
        device=device, dtype=dtype,
    ).view(1, 1, 3, 3)

    return kx, ky


class SobelEdgeLoss(nn.Module):

    def __init__(self, fft_weight: float = 0.2, fft_radius: float = 0.25):
        super().__init__()
        self.fft_weight = fft_weight
        self.fft_radius = fft_radius


    def _gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        kx, ky = _get_sobel_kernels(x.device, x.dtype)
        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)


    def _hf_consistency(self, pred: torch.Tensor,
                        target: torch.Tensor) -> torch.Tensor:
        B, C, H, W = pred.shape
        pred_fft   = torch.fft.fftshift(torch.fft.fft2(pred.float()),   dim=(-2, -1))
        target_fft = torch.fft.fftshift(torch.fft.fft2(target.float()), dim=(-2, -1))

        cy, cx = H // 2, W // 2
        ys = torch.arange(H, device=pred.device).float().view(H, 1) - cy
        xs = torch.arange(W, device=pred.device).float().view(1, W) - cx
        r  = torch.sqrt(ys ** 2 + xs ** 2)
        cutoff  = self.fft_radius * min(H, W) / 2
        hf_mask = (r > cutoff).float().unsqueeze(0).unsqueeze(0)

        pred_mag   = pred_fft.abs()   * hf_mask
        target_mag = target_fft.abs() * hf_mask

        norm = (target_mag.abs().mean() + 1e-8)
        return F.l1_loss(pred_mag / norm, target_mag / norm)


    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_grad   = self._gradient_magnitude(pred)
        target_grad = self._gradient_magnitude(target)
        spatial_loss = F.l1_loss(pred_grad, target_grad)

        if self.fft_weight > 0:
            hf_loss = self._hf_consistency(pred, target)
            return spatial_loss + self.fft_weight * hf_loss.to(spatial_loss.dtype)

        return spatial_loss

