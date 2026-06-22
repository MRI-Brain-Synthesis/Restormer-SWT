import torch
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure
from edge_loss import SobelEdgeLoss
import lpips as lpips_lib


class AdaptiveBandLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ssim_band = StructuralSimilarityIndexMeasure(data_range=1.0).to(config.DEVICE)
        self.ssim_img  = StructuralSimilarityIndexMeasure(data_range=1.0).to(config.DEVICE)
        self.edge      = SobelEdgeLoss(fft_weight=0.2)


        self.lpips_fn = lpips_lib.LPIPS(net="alex", verbose=False).to(config.DEVICE)
        for p in self.lpips_fn.parameters():
            p.requires_grad = False

        self.weights         = config.INITIAL_BAND_WEIGHTS.copy()
        self.edge_loss_w     = config.EDGE_LOSS_WEIGHT
        self.image_l1_w      = config.IMAGE_L1_WEIGHT
        self.image_ssim_w    = config.IMAGE_SSIM_WEIGHT
        self.image_lpips_w   = config.IMAGE_LPIPS_WEIGHT
        self.eps             = config.EPSILON
        self.charb_eps       = 1e-3

    def charbonnier(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.charb_eps ** 2))

    def forward(self, pred_bands, target_bands, pred_image=None, target_image=None):
        losses = {}
        total_loss = 0


        for band in ['LL', 'LH', 'HL', 'HH']:
            pred = pred_bands[band]
            targ = target_bands[band]

            loss = self.charbonnier(pred, targ)
            if band == 'LL':
                loss += (1 - self.ssim_band(pred.clamp(0, 1), targ.clamp(0, 1)))

            losses[band] = loss
            total_loss  += self.weights[band] * loss


        if pred_image is not None and target_image is not None:

            edge_l = self.edge(pred_image, target_image)
            losses['edge'] = edge_l
            total_loss += self.edge_loss_w * edge_l


            img_l1 = self.charbonnier(pred_image, target_image)
            losses['img_l1'] = img_l1
            total_loss += self.image_l1_w * img_l1


            pred_clamped = pred_image.clamp(0, 1)
            tgt_clamped  = target_image.clamp(0, 1)
            img_ssim_loss = 1.0 - self.ssim_img(pred_clamped, tgt_clamped)
            losses['img_ssim'] = img_ssim_loss
            total_loss += self.image_ssim_w * img_ssim_loss


            with torch.amp.autocast(device_type=pred_image.device.type, enabled=False):
                pred_3c = pred_clamped.float().repeat(1, 3, 1, 1) * 2.0 - 1.0
                tgt_3c  = tgt_clamped.float().repeat(1, 3, 1, 1) * 2.0 - 1.0
                img_lpips = self.lpips_fn(pred_3c, tgt_3c).mean()
            losses['img_lpips'] = img_lpips
            total_loss += self.image_lpips_w * img_lpips
        else:
            losses['edge']      = torch.tensor(0.0)
            losses['img_l1']    = torch.tensor(0.0)
            losses['img_ssim']  = torch.tensor(0.0)
            losses['img_lpips'] = torch.tensor(0.0)

        return total_loss, losses

    def update_weights(self, avg_epoch_losses):
        band_losses = {k: v for k, v in avg_epoch_losses.items()
                       if k in ('LL', 'LH', 'HL', 'HH')}
        inv_losses  = {k: 1.0 / (v + self.eps) for k, v in band_losses.items()}
        sum_inv     = sum(inv_losses.values())
        for k in self.weights:
            self.weights[k] = inv_losses[k] / sum_inv
