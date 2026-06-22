import os
import glob
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from config import Config

class MRIDataset(Dataset):
    def __init__(self, split='train'):
        self.data_dir = os.path.join(Config.DATA_ROOT, split)
        self.files = sorted(glob.glob(os.path.join(self.data_dir, "*.npz")))
        self.augment = (split == 'train')
        if not self.files:
            print(f"⚠️ Warning: No .npz files found in {self.data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])


        input_img = data.get('input', data.get('low_nsa'))
        target_img = data.get('target', data.get('high_nsa'))

        input_img = input_img.astype(np.float32)
        target_img = target_img.astype(np.float32)


        scale = np.percentile(target_img, 99) + 1e-8
        input_norm = input_img / scale
        target_norm = target_img / scale


        if self.augment:
            if random.random() > 0.5:
                input_norm = np.fliplr(input_norm).copy()
                target_norm = np.fliplr(target_norm).copy()
            if random.random() > 0.5:
                input_norm = np.flipud(input_norm).copy()
                target_norm = np.flipud(target_norm).copy()


        if input_norm.ndim == 2:
            input_norm = input_norm[np.newaxis, ...]
            target_norm = target_norm[np.newaxis, ...]

        return torch.from_numpy(input_norm), torch.from_numpy(target_norm), torch.tensor(scale, dtype=torch.float32)
