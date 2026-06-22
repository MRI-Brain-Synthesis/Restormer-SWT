"""
Restormer-SWT Configuration
All hyperparameters used in the paper.
"""
import os


class Config:
    # ── Paths (override via environment variables) ────────────────────────────
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = os.environ.get("SWT_DATA_ROOT", os.path.join(BASE_DIR, "data"))
    OUTPUT_ROOT = os.environ.get("SWT_OUTPUT_ROOT", os.path.join(BASE_DIR, "outputs"))
    CHECKPOINT_DIR = os.path.join(OUTPUT_ROOT, "checkpoints")
    VISUALIZATION_DIR = os.path.join(OUTPUT_ROOT, "visualization")

    # ── Device ────────────────────────────────────────────────────────────────
    DEVICE = "cuda"

    # ── Training ──────────────────────────────────────────────────────────────
    BATCH_SIZE = 2
    GRAD_ACCUM = 4          # effective batch size = BATCH_SIZE * GRAD_ACCUM = 8
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 50
    PATIENCE = 30           # early stopping patience
    WARMUP_EPOCHS = 5
    SEED = 42

    # ── Model Architecture ────────────────────────────────────────────────────
    IN_CHANNELS = 1
    NUM_GPUS = 1
    SWT_LEVEL = 1
    NUM_TRANSFORMER_BLOCKS = 4
    MID_DIM = 96
    DROPOUT = 0.1

    # ── Loss Weights ──────────────────────────────────────────────────────────
    IMAGE_L1_WEIGHT = 1.0
    IMAGE_SSIM_WEIGHT = 0.4
    IMAGE_LPIPS_WEIGHT = 0.05
    EDGE_LOSS_WEIGHT = 0.15
    INITIAL_BAND_WEIGHTS = {"LL": 0.25, "LH": 0.25, "HL": 0.25, "HH": 0.25}
    EPSILON = 1e-6

    # ── Visualization ─────────────────────────────────────────────────────────
    VISUALIZATION_INTERVAL = 5
    NUM_VIS = 10
    NUM_VIS_EPOCH = 4
