import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossMDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.q_dwconv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.k_dwconv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

        self.v_proj = nn.Conv2d(channels, channels, 1)
        self.v_dwconv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

        self.project_out = nn.Conv2d(channels, channels, 1)

    def forward(self, q_x, kv_x):
        b, c, h, w = q_x.shape


        q = self.q_dwconv(self.q_proj(q_x))

        k = self.k_dwconv(self.k_proj(kv_x))
        v = self.v_dwconv(self.v_proj(kv_x))

        q = q.view(b, self.num_heads, c // self.num_heads, -1)
        k = k.view(b, self.num_heads, c // self.num_heads, -1)
        v = v.view(b, self.num_heads, c // self.num_heads, -1)

        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v).view(b, c, h, w)
        return self.project_out(out)

class GatedDconvFFN(nn.Module):
    def __init__(self, channels, expansion_factor=2.66):
        super().__init__()
        hidden = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden * 2, 1)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2)
        self.project_out = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class CrossRestormerBlock(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()

        self.norm_sa = nn.LayerNorm(channels)
        self.self_attn = CrossMDTA(channels, num_heads)


        self.norm1 = nn.LayerNorm(channels)
        self.attn = CrossMDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)


        self.ffn = GatedDconvFFN(channels)

    def forward(self, mid_x, ll_x):
        b, c, h, w = mid_x.shape


        norm_x = self.norm_sa(mid_x.flatten(2).transpose(1, 2)).transpose(1, 2).view(b, c, h, w)
        x = mid_x + self.self_attn(norm_x, norm_x)


        norm_mid = self.norm1(x.flatten(2).transpose(1, 2)).transpose(1, 2).view(b, c, h, w)
        attn_out = self.attn(norm_mid, ll_x)
        x = x + attn_out


        norm_x = self.norm2(x.flatten(2).transpose(1, 2)).transpose(1, 2).view(b, c, h, w)
        x = x + self.ffn(norm_x)
        return x

