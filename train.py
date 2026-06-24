"""
PraFormer Training Script
Train the HierarchicalRestormer model on paired MRI data.

Usage:
    export SWT_DATA_ROOT=/path/to/data
    export SWT_OUTPUT_ROOT=/path/to/output
    python train.py
"""
import os, sys, glob, random, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import Config
from dataset import MRIDataset
from model.full_model import HierarchicalRestormer
from losses import AdaptiveBandLoss
from dwt_utils import SWTProcessor

# ── Setup ─────────────────────────────────────────────────────────────────────
random.seed(Config.SEED)
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)
device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")

CKPT_DIR = os.path.join(Config.OUTPUT_ROOT, "checkpoints")
VIS_DIR = os.path.join(Config.OUTPUT_ROOT, "visualization")
for d in [CKPT_DIR, VIS_DIR]:
    os.makedirs(d, exist_ok=True)


# ── Warmup + Cosine Scheduler ────────────────────────────────────────────────
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr


# ── Visualization ─────────────────────────────────────────────────────────────
def save_epoch_vis(model, val_loader, epoch, vis_dir, n=4):
    model.eval()
    os.makedirs(os.path.join(vis_dir, f"epoch_{epoch:03d}"), exist_ok=True)
    with torch.no_grad():
        batch = next(iter(val_loader))
        inp, tgt = batch[0][:n].to(device), batch[1][:n].to(device)
        out = model(inp).clamp(0, 1)
        for i in range(min(n, inp.shape[0])):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for ax, img, title in zip(axes,
                [inp[i, 0].cpu(), out[i, 0].cpu(), tgt[i, 0].cpu()],
                ['Input (Noisy)', 'Output (Denoised)', 'Target']):
                ax.imshow(img, cmap='gray'); ax.set_title(title); ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"epoch_{epoch:03d}", f"sample_{i}.png"), dpi=100)
            plt.close()


# ── Training ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  PraFormer Training")
    print(f"  Data:   {Config.DATA_ROOT}")
    print(f"  Output: {Config.OUTPUT_ROOT}")
    print("=" * 70, flush=True)

    # Data
    train_ds = MRIDataset('train')
    val_ds = MRIDataset('val')
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE,
                            shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = HierarchicalRestormer(Config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.2f}M)", flush=True)

    # Loss, Optimizer, Scheduler
    criterion = AdaptiveBandLoss(Config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE,
                            weight_decay=Config.WEIGHT_DECAY)
    warmup_sched = WarmupCosineScheduler(optimizer, Config.WARMUP_EPOCHS,
                                         Config.NUM_EPOCHS, Config.LEARNING_RATE)
    plateau_sched = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=Config.PATIENCE // 3,
        factor=0.5, min_lr=1e-7)

    swt_proc = SWTProcessor(level=Config.SWT_LEVEL).to(device)

    # Training state
    best_psnr = -1
    best_epoch = 0
    no_improve = 0
    history = []

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        model.train()
        epoch_losses = {k: 0.0 for k in ['total', 'LL', 'LH', 'HL', 'HH',
                                           'edge', 'img_l1', 'img_ssim', 'img_lpips']}
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"  E{epoch}/{Config.NUM_EPOCHS}", leave=False)):
            inp, tgt = batch[0].to(device), batch[1].to(device)

            out, pred_bands, _ = model(inp, return_bands=True)
            tgt_bands, tgt_stds = swt_proc.forward_transform(tgt)
            loss, loss_dict = criterion(pred_bands, tgt_bands, out, tgt)
            loss = loss / Config.GRAD_ACCUM
            loss.backward()

            if (step + 1) % Config.GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_losses['total'] += loss.item() * Config.GRAD_ACCUM
            for k, v in loss_dict.items():
                if k in epoch_losses:
                    epoch_losses[k] += v.item() if torch.is_tensor(v) else v

        # Average losses
        n_batches = len(train_loader)
        avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}

        # Update band weights
        criterion.update_weights(avg_losses)

        # Warmup scheduler
        if epoch <= Config.WARMUP_EPOCHS:
            warmup_sched.step(epoch)

        # Validation
        model.eval()
        val_psnr = []
        with torch.no_grad():
            for batch in val_loader:
                inp, tgt = batch[0].to(device), batch[1].to(device)
                out = model(inp).clamp(0, 1)
                for b in range(inp.shape[0]):
                    mse = torch.mean((out[b] - tgt[b]) ** 2)
                    psnr = 10 * torch.log10(1.0 / (mse + 1e-10))
                    val_psnr.append(psnr.item())
        mean_psnr = np.mean(val_psnr)

        # Plateau scheduler (after warmup)
        if epoch > Config.WARMUP_EPOCHS:
            plateau_sched.step(mean_psnr)

        lr = optimizer.param_groups[0]['lr']
        print(f"  E {epoch:3d} | Loss={avg_losses['total']:.4f} | "
              f"Val PSNR={mean_psnr:.2f} | LR={lr:.2e}", flush=True)

        # Best checkpoint
        if mean_psnr > best_psnr:
            best_psnr = mean_psnr
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "best_psnr": best_psnr,
                "optimizer": optimizer.state_dict(),
                "band_weights": criterion.weights,
            }, os.path.join(CKPT_DIR, "best_model.pth"))
            print(f"    -> Best (PSNR={mean_psnr:.4f})", flush=True)
        else:
            no_improve += 1

        # Visualization
        if epoch % Config.VISUALIZATION_INTERVAL == 0:
            save_epoch_vis(model, val_loader, epoch, VIS_DIR)

        history.append({
            "epoch": epoch, "loss": avg_losses['total'],
            "val_psnr": mean_psnr, "lr": lr
        })

        # Early stopping
        if no_improve >= Config.PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (best={best_epoch})", flush=True)
            break

    # Save training history
    with open(os.path.join(Config.OUTPUT_ROOT, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  Training complete. Best epoch: {best_epoch} | PSNR: {best_psnr:.4f}")
    print(f"  Checkpoint: {os.path.join(CKPT_DIR, 'best_model.pth')}")
    print(f"{'=' * 70}", flush=True)


if __name__ == "__main__":
    main()
