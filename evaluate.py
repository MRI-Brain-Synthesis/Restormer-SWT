"""
PraFormer Evaluation Script
Evaluate the trained model on the test set with PSNR, SSIM, LPIPS, and FID.

Usage:
    export SWT_DATA_ROOT=/path/to/data
    export SWT_OUTPUT_ROOT=/path/to/output
    python evaluate.py
"""
import os, sys, glob, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import lpips as lpips_lib
from torchmetrics.image.fid import FrechetInceptionDistance

from config import Config
from dataset import MRIDataset
from model.full_model import HierarchicalRestormer

device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")


# ── Metrics ───────────────────────────────────────────────────────────────────
lpips_fn = lpips_lib.LPIPS(net='alex').to(device).eval()

def compute_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return (10 * torch.log10(1.0 / (mse + 1e-10))).item()

def compute_ssim(pred, target, C1=0.01**2, C2=0.03**2, ks=11):
    pad = ks // 2
    mu_x = torch.nn.functional.avg_pool2d(pred, ks, 1, pad)
    mu_y = torch.nn.functional.avg_pool2d(target, ks, 1, pad)
    s_x = torch.nn.functional.avg_pool2d(pred * pred, ks, 1, pad) - mu_x * mu_x
    s_y = torch.nn.functional.avg_pool2d(target * target, ks, 1, pad) - mu_y * mu_y
    s_xy = torch.nn.functional.avg_pool2d(pred * target, ks, 1, pad) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + C1) * (2 * s_xy + C2)) / \
           ((mu_x**2 + mu_y**2 + C1) * (s_x + s_y + C2))
    return ssim.mean().item()

def compute_lpips(pred, target):
    return lpips_fn(pred.repeat(1, 3, 1, 1), target.repeat(1, 3, 1, 1)).mean().item()


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, split_name="test"):
    model.eval()
    psnr_in, psnr_out = [], []
    ssim_in, ssim_out = [], []
    lpips_in, lpips_out = [], []
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    for batch in tqdm(loader, desc=f"  Evaluating ({split_name})", leave=False):
        inp, tgt = batch[0].to(device), batch[1].to(device)
        out = model(inp).clamp(0, 1)

        for b in range(inp.shape[0]):
            psnr_in.append(compute_psnr(inp[b:b+1], tgt[b:b+1]))
            psnr_out.append(compute_psnr(out[b:b+1], tgt[b:b+1]))
            ssim_in.append(compute_ssim(inp[b:b+1], tgt[b:b+1]))
            ssim_out.append(compute_ssim(out[b:b+1], tgt[b:b+1]))
            lpips_in.append(compute_lpips(inp[b:b+1], tgt[b:b+1]))
            lpips_out.append(compute_lpips(out[b:b+1], tgt[b:b+1]))

        fid.update(tgt.repeat(1, 3, 1, 1), real=True)
        fid.update(out.repeat(1, 3, 1, 1), real=False)

    results = {
        "n_samples": len(psnr_in),
        "psnr_in": float(np.mean(psnr_in)), "psnr_in_std": float(np.std(psnr_in)),
        "psnr_out": float(np.mean(psnr_out)), "psnr_out_std": float(np.std(psnr_out)),
        "ssim_in": float(np.mean(ssim_in)), "ssim_in_std": float(np.std(ssim_in)),
        "ssim_out": float(np.mean(ssim_out)), "ssim_out_std": float(np.std(ssim_out)),
        "lpips_in": float(np.mean(lpips_in)), "lpips_in_std": float(np.std(lpips_in)),
        "lpips_out": float(np.mean(lpips_out)), "lpips_out_std": float(np.std(lpips_out)),
        "fid": float(fid.compute().item()),
    }
    return results


def main():
    ckpt_path = os.path.join(Config.OUTPUT_ROOT, "checkpoints", "best_model.pth")
    eval_dir = os.path.join(Config.OUTPUT_ROOT, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    print("=" * 70)
    print("  PraFormer Evaluation")
    print(f"  Checkpoint: {ckpt_path}")
    print("=" * 70, flush=True)

    # Load model
    model = HierarchicalRestormer(Config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    best_epoch = ckpt.get("epoch", "?")
    best_psnr = ckpt.get("best_psnr", "?")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,} ({n_params/1e6:.2f}M)")
    print(f"  Best epoch: {best_epoch} | Val PSNR: {best_psnr}", flush=True)

    # Evaluate on test set
    test_ds = MRIDataset('test')
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False,
                             num_workers=4, pin_memory=True)
    results = evaluate(model, test_loader, "test")

    # Print results
    print(f"\n  {'Metric':<16} {'Input':>20} {'Output':>20}")
    print(f"  {'-'*56}")
    print(f"  {'PSNR (dB)':<16} {results['psnr_in']:.4f} +/- {results['psnr_in_std']:.4f}   "
          f"{results['psnr_out']:.4f} +/- {results['psnr_out_std']:.4f}")
    print(f"  {'SSIM':<16} {results['ssim_in']:.4f} +/- {results['ssim_in_std']:.4f}   "
          f"{results['ssim_out']:.4f} +/- {results['ssim_out_std']:.4f}")
    print(f"  {'LPIPS':<16} {results['lpips_in']:.4f} +/- {results['lpips_in_std']:.4f}   "
          f"{results['lpips_out']:.4f} +/- {results['lpips_out_std']:.4f}")
    print(f"\n  FID: {results['fid']:.4f}")

    # Save results
    with open(os.path.join(eval_dir, "summary.txt"), "w") as f:
        f.write(f"PraFormer -- Test Evaluation\n{'='*60}\n")
        f.write(f"Best epoch: {best_epoch}  |  Params: {n_params/1e6:.2f}M\n\n")
        f.write(f"Samples: {results['n_samples']}\n\n")
        f.write(f"{'Metric':<24} {'Input':>20} {'Output':>20}\n{'-'*64}\n")
        f.write(f"{'PSNR (dB)':<24} {results['psnr_in']:.4f} +/- {results['psnr_in_std']:.4f}   "
                f"{results['psnr_out']:.4f} +/- {results['psnr_out_std']:.4f}\n")
        f.write(f"{'SSIM':<24} {results['ssim_in']:.4f} +/- {results['ssim_in_std']:.4f}   "
                f"{results['ssim_out']:.4f} +/- {results['ssim_out_std']:.4f}\n")
        f.write(f"{'LPIPS':<24} {results['lpips_in']:.4f} +/- {results['lpips_in_std']:.4f}   "
                f"{results['lpips_out']:.4f} +/- {results['lpips_out_std']:.4f}\n")
        f.write(f"\nFID: {results['fid']:.4f}\n")

    with open(os.path.join(eval_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  Results saved to {eval_dir}")
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
