# Restormer-SWT: Hierarchical MRI Denoising via Stationary Wavelet Transform and Restormer

A lightweight (0.83M parameters) MRI denoising model that combines **Stationary Wavelet Transform (SWT)** decomposition with a **Restormer**-based hierarchical architecture for effective noise removal in MRI images.

## Architecture

Restormer-SWT decomposes each input slice into four SWT sub-bands (LL, LH, HL, HH) and processes them with specialised sub-networks:

- **LL Encoder**: CNN + Sobel gradient features for low-frequency structural content
- **Mid-Frequency Transformer**: Cross-attention Restormer blocks for LH/HL detail bands, guided by LL features
- **HH Gate**: Multi-scale dilated convolutions with LL-guided gating for high-frequency noise suppression
- **Refinement CNN**: Final residual refinement in the image domain

The sub-band outputs are recombined via inverse SWT, followed by a lightweight refinement network.

## Project Structure

```
Restormer-SWT/
├── model/
│   ├── full_model.py        # HierarchicalRestormer (main model)
│   ├── ll_encoder.py        # LL sub-band encoder
│   ├── mid_transformer.py   # LH/HL cross-attention transformer
│   ├── restormer_block.py   # CrossRestormerBlock, CrossMDTA, GatedDconvFFN
│   └── hh_gate.py           # Edge-enhanced HH gating module
├── config.py                # Model & training configuration
├── dataset.py               # MRI paired dataset loader
├── dwt_utils.py             # Stationary Wavelet Transform (SWT) processor
├── losses.py                # Adaptive band loss + image-domain losses
├── edge_loss.py             # Sobel edge + FFT high-frequency consistency loss
├── train.py                 # Training script
├── evaluate.py              # Evaluation script
├── requirements.txt         # Dependencies
└── README.md
```

## Installation

```bash
git clone https://github.com/MRI-Brain-Synthesis/Restormer-SWT.git
cd Restormer-SWT
pip install -r requirements.txt
```

## Configuration

All hyperparameters are defined in `config.py`. Key settings:

| Parameter | Value | Description |
|---|---|---|
| `LEARNING_RATE` | 2e-4 | AdamW learning rate |
| `BATCH_SIZE` | 2 | Per-GPU batch size |
| `GRAD_ACCUM` | 4 | Gradient accumulation steps (effective batch = 8) |
| `NUM_EPOCHS` | 50 | Maximum training epochs |
| `PATIENCE` | 30 | Early stopping patience |
| `DROPOUT` | 0.1 | Dropout rate |
| `MID_DIM` | 96 | Transformer hidden dimension |
| `NUM_TRANSFORMER_BLOCKS` | 4 | Number of CrossRestormerBlocks |
| `SWT_LEVEL` | 1 | SWT decomposition level |

### Loss Weights

| Loss Component | Weight |
|---|---|
| Image L1 (Charbonnier) | 1.0 |
| Image SSIM | 0.4 |
| Image LPIPS | 0.05 |
| Edge Loss | 0.15 |
| Band Weights (LL/LH/HL/HH) | Adaptive (init 0.25 each) |

## Dataset

The model expects paired `.npz` files with keys `input` (noisy) and `target` (clean/averaged), organised as:

```
data_root/
├── train/
│   ├── 0001.npz
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

Each `.npz` file should contain 2D MRI slices (H×W). The dataset loader automatically normalises by the 99th percentile and applies random flips for augmentation.

## Training

```bash
# Set data and output paths
export SWT_DATA_ROOT=/path/to/your/data
export SWT_OUTPUT_ROOT=/path/to/output

# Train
python train.py
```

Training features:
- AdamW optimiser with cosine warmup (5 epochs)
- ReduceLROnPlateau scheduler (patience = NUM_EPOCHS//3)
- Gradient clipping (max_norm = 1.0)
- Gradient accumulation (effective batch size = 8)
- Early stopping on validation PSNR
- Periodic visualisation every 5 epochs
- Adaptive band weight rebalancing

## Evaluation

```bash
python evaluate.py
```

Computes PSNR, SSIM, LPIPS, and FID on the test set using the best checkpoint.
