#!/usr/bin/env python3
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from dataset import CoCrDataset
from train import UNet

def compute_iou(pred, gt, eps=1e-6):
    """Intersection over Union for two boolean arrays."""
    intersection = (pred & gt).sum()
    union = (pred | gt).sum() + eps
    return intersection / union

def main():
    base_dir   = os.path.dirname(__file__)
    data_root  = os.path.join(base_dir, 'CoCrOutput')
    csv_path   = os.path.join(data_root, 'build_logs.csv')
    model_path = os.path.join(base_dir, 'best_model.pth')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = UNet(n_channels=1, n_classes=1, param_dim=5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Build validation loader (20% of data)
    dataset = CoCrDataset(data_root, csv_path, slice_index=0, target_size=(256,256))
    n_val   = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    _, val_ds = random_split(dataset, [n_train, n_val])
    loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    dices = []
    ious  = []

    with torch.no_grad():
        for image, gt_mask, params in loader:
            image  = image.to(device)
            params = params.to(device)
            logits = model(image, params)
            probs  = torch.sigmoid(logits)

            # Binarize at 0.5
            pred_mask = (probs > 0.5).cpu().numpy().astype(bool)[0,0]
            gt        = gt_mask.numpy().astype(bool)[0,0]

            # Dice score
            inter = 2 * (pred_mask & gt).sum()
            denom = pred_mask.sum() + gt.sum() + 1e-6
            dices.append(inter / denom)

            # IoU
            ious.append(compute_iou(pred_mask, gt))

    print(f"Mean Dice on val set: {np.mean(dices):.4f}")
    print(f"Mean IoU  on val set: {np.mean(ious):.4f}")

if __name__ == '__main__':
    main()
