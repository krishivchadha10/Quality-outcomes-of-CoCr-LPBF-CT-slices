#!/usr/bin/env python3
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from dataset import CoCrDataset
from train import UNet
import multiprocessing

# ─── Configuration ───────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
DATA_ROOT   = os.path.join(BASE_DIR, 'CoCrOutput')
CSV_PATH    = os.path.join(DATA_ROOT, 'build_logs.csv')
MODEL_PATH  = os.path.join(BASE_DIR, 'best_model.pth')
SLICE_INDEX = 0
TARGET_SIZE = (256, 256)
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # Load dataset and split
    dataset = CoCrDataset(DATA_ROOT, CSV_PATH,
                          slice_index=SLICE_INDEX,
                          target_size=TARGET_SIZE)
    n_val   = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    _, val_ds = random_split(dataset, [n_train, n_val])

    loader = DataLoader(val_ds,
                        batch_size=1,
                        shuffle=False,
                        num_workers=4,  # or 0 if you still have issues
                        pin_memory=True)

    # Load model
    model = UNet(n_channels=1, n_classes=1, param_dim=5).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Precompute probabilities and GT pore masks
    probs_list = []
    gt_pore_list = []

    with torch.no_grad():
        for img, solid_mask, params in loader:
            img    = img.to(DEVICE)
            params = params.to(DEVICE)
            logits = model(img, params)
            prob_solid = torch.sigmoid(logits)[0,0].cpu().numpy()

            gt_solid = solid_mask.numpy().astype(bool)[0,0]
            gt_pore  = ~gt_solid

            probs_list.append(prob_solid)
            gt_pore_list.append(gt_pore)

    # Sweep pore thresholds
    best = {'pore_thr': None, 'dice': -1.0}
    pore_thresholds = np.linspace(0.01, 0.5, 50)

    for pore_thr in pore_thresholds:
        dices = []
        for prob_solid, gt_pore in zip(probs_list, gt_pore_list):
            pore_prob = 1.0 - prob_solid
            pred_pore = pore_prob > pore_thr
            inter = 2 * (pred_pore & gt_pore).sum()
            denom = pred_pore.sum() + gt_pore.sum() + 1e-6
            dices.append(inter / denom)
        avg_dice = float(np.mean(dices))
        if avg_dice > best['dice']:
            best['dice'] = avg_dice
            best['pore_thr'] = float(pore_thr)

    print(f"Best pore threshold: {best['pore_thr']:.3f}, Dice: {best['dice']:.4f}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
