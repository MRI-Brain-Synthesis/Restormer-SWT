import torch
import torch.nn as nn
from dwt_utils import SWTProcessor
from .ll_encoder import LLEncoder
from .mid_transformer import MidFrequencyTransformer
from .hh_gate import EdgeEnhancedHHGate


class HierarchicalRestormer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.swt_proc = SWTProcessor(level=config.SWT_LEVEL)

        c   = config.IN_CHANNELS
        dim = config.MID_DIM

        self.ll_encoder      = LLEncoder(c, dim)
        self.mid_transformer = MidFrequencyTransformer(c, dim, config.NUM_TRANSFORMER_BLOCKS)
        self.hh_gate         = EdgeEnhancedHHGate(dim, c)


        self.ll_out_proj = nn.Conv2d(dim, c, 1)


        self.refine = nn.Sequential(
            nn.Conv2d(c * 2, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, c, 3, padding=1),
        )

    def forward(self, x, return_bands=False):

        norm_bands, stds = self.swt_proc.forward_transform(x)


        ll_features = self.ll_encoder(norm_bands['LL'])
        ll_out      = self.ll_out_proj(ll_features)


        lh_restored, hl_restored = self.mid_transformer(
            norm_bands['LH'], norm_bands['HL'], ll_features
        )


        hh_clean = self.hh_gate(norm_bands['HH'], ll_features)

        restored_bands = {
            'LL': ll_out,
            'LH': lh_restored,
            'HL': hl_restored,
            'HH': hh_clean,
        }


        swt_output = self.swt_proc.inverse_transform(restored_bands, stds)


        output = swt_output + self.refine(torch.cat([swt_output, x], dim=1))

        if return_bands:
            return output, restored_bands, norm_bands
        return output
