#!/usr/bin/env python3
import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from skimage.io import imread
import numpy as np

class CoCrDataset(Dataset):
    """
    PyTorch Dataset for CoCr LPBF slices.
    Returns: (image_tensor, mask_tensor, params_tensor) for each sample.
    Images and masks are resized to a fixed target size.
    """
    def __init__(self,
                 root_dir: str,
                 csv_path: str,
                 transforms=None,
                 slice_index: int = 0,
                 target_size: tuple = (256, 256)):
        super().__init__()
        self.root_dir    = root_dir
        self.transforms  = transforms
        self.slice_index = slice_index
        self.target_size = target_size

        # Load and clean CSV
        self.params_df = pd.read_csv(csv_path, dtype={'sample_id': str})
        self.params_df['sample_id'] = self.params_df['sample_id'].str.strip()

        # List available image sample folders
        images_dir = os.path.join(root_dir, 'images')
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        available = set(os.listdir(images_dir))

        # Filter CSV to match available folders
        self.params_df = (
            self.params_df[self.params_df['sample_id'].isin(available)]
            .reset_index(drop=True)
        )
        if self.params_df.empty:
            raise ValueError("No matching samples found between CSV and image folders.")
        self.sample_ids = self.params_df['sample_id'].tolist()

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]

        # Paths for this sample
        img_folder  = os.path.join(self.root_dir, 'images', sample_id)
        mask_folder = os.path.join(self.root_dir, 'masks',  sample_id)
        if not (os.path.isdir(img_folder) and os.path.isdir(mask_folder)):
            raise FileNotFoundError(f"Missing folder for sample {sample_id}")

        # Get sorted TIFF filenames
        img_files  = sorted([f for f in os.listdir(img_folder)  if f.lower().endswith('.tif')])
        mask_files = sorted([f for f in os.listdir(mask_folder) if f.lower().endswith('.tif')])
        if not img_files or not mask_files:
            raise FileNotFoundError(f"No TIFF files found for sample {sample_id}")

        # ── BOUNDS CHECK FOR slice_index ─────────────────────────
        if not (0 <= self.slice_index < len(img_files)):
            raise IndexError(
                f"slice_index={self.slice_index} is out of range "
                f"(must be between 0 and {len(img_files)-1}) for sample {sample_id}"
            )
        idx_slice = self.slice_index

        # Build full paths
        img_path  = os.path.join(img_folder,  img_files[idx_slice])
        mask_path = os.path.join(mask_folder, mask_files[idx_slice])

        # Load images
        image = imread(img_path).astype(np.float32)
        mask  = imread(mask_path).astype(np.float32)

        # Normalize image to [0,1]
        if image.max() > image.min():
            image = (image - image.min()) / (image.max() - image.min())
        # Binarize mask to 0/1
        mask = (mask > 0).astype(np.float32)

        # Add channel dimension
        image = np.expand_dims(image, axis=0)
        mask  = np.expand_dims(mask,  axis=0)

        # To torch tensors
        image_tensor = torch.from_numpy(image)
        mask_tensor  = torch.from_numpy(mask)

        # Resize to target size
        image_tensor = F.interpolate(
            image_tensor.unsqueeze(0),
            size=self.target_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        mask_tensor = F.interpolate(
            mask_tensor.unsqueeze(0),
            size=self.target_size,
            mode='nearest'
        ).squeeze(0)

        # Retrieve parameters and compute energy density
        row    = self.params_df.loc[self.params_df['sample_id'] == sample_id].iloc[0]
        power  = float(row['power_W'])
        speed  = float(row['scan_speed_mm_s'])
        hatch  = float(row['hatch_spacing_mm'])
        thick  = float(row['layer_thick_mm'])
        energy = power / (speed * hatch * thick)

        params = torch.tensor([power, speed, hatch, thick, energy], dtype=torch.float32)

        # Apply transforms if any
        if self.transforms:
            image_tensor, mask_tensor = self.transforms(image_tensor, mask_tensor)

        return image_tensor, mask_tensor, params

if __name__ == '__main__':
    # Quick sanity test
    base_dir  = os.path.dirname(__file__)
    data_root = os.path.join(base_dir, 'CoCrOutput')
    csv_path  = os.path.join(data_root, 'build_logs.csv')

    dataset = CoCrDataset(
        root_dir    = data_root,
        csv_path    = csv_path,
        slice_index = 0,
        target_size = (256,256)
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)
    for images, masks, params in loader:
        print(f"Images: {images.shape}, Masks: {masks.shape}, Params: {params.shape}")
        break
