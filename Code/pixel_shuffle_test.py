#!/usr/bin/env python3
import os
import numpy as np
import torch
from dataset import CoCrDataset
from train import UNet

# ─── Configuration ───────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
DATA_ROOT   = os.path.join(BASE_DIR, 'CoCrOutput')
CSV_PATH    = os.path.join(DATA_ROOT, 'build_logs.csv')
MODEL_PATH  = os.path.join(BASE_DIR, 'best_model.pth')
TARGET_SIZE = (256, 256)
SLICE_INDEX = 0

# ─── Device ──────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── Load one real sample ────────────────────────────────────
ds = CoCrDataset(
    root_dir=DATA_ROOT,
    csv_path=CSV_PATH,
    slice_index=SLICE_INDEX,
    target_size=TARGET_SIZE
)
img_t, solid_mask_t, params_t = ds[0]  # img_t: (1,H,W), solid_mask_t: (1,H,W)

# ─── Scramble pixels ─────────────────────────────────────────
arr = img_t.squeeze(0).cpu().numpy()  # (H,W)
# only scramble inside the metal disk, leave background intact
mask = solid_mask_t.squeeze(0).cpu().numpy().astype(bool)
flat_vals = arr[mask]
np.random.shuffle(flat_vals)
scrambled = arr.copy()
scrambled[mask] = flat_vals
# convert back to tensor
scr_t = torch.from_numpy(scrambled).unsqueeze(0).unsqueeze(0).float().to(device)
params_batch = params_t.unsqueeze(0).to(device)

# ─── Load model ───────────────────────────────────────────────
model = UNet(n_channels=1, n_classes=1, param_dim=5).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ─── Inference on scrambled image ────────────────────────────
with torch.no_grad():
    logits = model(scr_t, params_batch)  # (1,1,H,W)
    probs  = torch.sigmoid(logits)[0,0]  # (H,W)

probs_np = probs.cpu().numpy()
print(f"Scrambled → P(solid) min/max: {probs_np.min():.4f} / {probs_np.max():.4f}")
