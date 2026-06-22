
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SWTProcessor(nn.Module):

    def __init__(self, level: int = 1):
        super().__init__()
        self.level = level


        lo = torch.tensor([1.0 / math.sqrt(2), 1.0 / math.sqrt(2)])
        hi = torch.tensor([1.0 / math.sqrt(2), -1.0 / math.sqrt(2)])

        self.register_buffer('lo', lo)
        self.register_buffer('hi', hi)


    def _build_filters(self, channels: int, device, dtype):
        lo = self.lo.to(device=device, dtype=dtype)
        hi = self.hi.to(device=device, dtype=dtype)


        ll_k = torch.outer(lo, lo)
        lh_k = torch.outer(hi, lo)
        hl_k = torch.outer(lo, hi)
        hh_k = torch.outer(hi, hi)

        def _expand(k):

            return k.unsqueeze(0).unsqueeze(0).expand(channels, -1, -1, -1)

        return _expand(ll_k), _expand(lh_k), _expand(hl_k), _expand(hh_k)


    @staticmethod
    def _pad_same(x, kH, kW):
        pad_h = kH - 1
        pad_w = kW - 1
        pad_top = pad_h // 2
        pad_bot = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        return F.pad(x, (pad_left, pad_right, pad_top, pad_bot), mode='reflect')

    def _swt_single(self, x: torch.Tensor):
        B, C, H, W = x.shape
        ll_k, lh_k, hl_k, hh_k = self._build_filters(C, x.device, x.dtype)

        kH, kW = ll_k.shape[-2], ll_k.shape[-1]

        x_padded = self._pad_same(x, kH, kW)

        LL = F.conv2d(x_padded, ll_k, groups=C)
        LH = F.conv2d(x_padded, lh_k, groups=C)
        HL = F.conv2d(x_padded, hl_k, groups=C)
        HH = F.conv2d(x_padded, hh_k, groups=C)

        return LL, LH, HL, HH


    def _iswt_single(self, LL, LH, HL, HH):
        B, C, H, W = LL.shape
        ll_k, lh_k, hl_k, hh_k = self._build_filters(C, LL.device, LL.dtype)

        kH, kW = ll_k.shape[-2], ll_k.shape[-1]

        def _adj_conv(band, kernel):

            out = F.conv_transpose2d(band, kernel, groups=C)

            crop_h = out.shape[2] - H
            crop_w = out.shape[3] - W
            top = crop_h // 2
            left = crop_w // 2
            return out[:, :, top:top+H, left:left+W]

        rec = (_adj_conv(LL, ll_k) +
               _adj_conv(LH, lh_k) +
               _adj_conv(HL, hl_k) +
               _adj_conv(HH, hh_k))

        return rec


    def forward_transform(self, x: torch.Tensor):
        LL, LH, HL, HH = self._swt_single(x)
        bands = {'LL': LL, 'LH': LH, 'HL': HL, 'HH': HH}

        stds = {}
        norm_bands = {}
        for key, band in bands.items():
            std = band.std(dim=[2, 3], keepdim=True) + 1e-8
            stds[key] = std
            norm_bands[key] = band / std

        return norm_bands, stds

    def inverse_transform(self, norm_bands: dict, stds: dict) -> torch.Tensor:
        destd = {k: norm_bands[k] * stds[k] for k in norm_bands}
        return self._iswt_single(
            destd['LL'], destd['LH'], destd['HL'], destd['HH']
        )
