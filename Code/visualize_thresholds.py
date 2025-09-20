#!/usr/bin/env python3
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import CoCrDataset
from train import UNet
import random

# ─── CONFIG ─────────────────────────────────────────────────
base_dir    = os.path.dirname(__file__)
data_root   = os.path.join(base_dir, 'CoCrOutput')
csv_path    = os.path.join(data_root, 'build_logs.csv')
model_path  = os.path.join(base_dir, 'best_model.pth')
slice_index = 1
target_size = (256, 256)
device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── LOAD SAMPLE & MODEL ────────────────────────────────────
ds = CoCrDataset(
    root_dir   = data_root,
    csv_path   = csv_path,
    slice_index= slice_index,
    target_size= target_size
)
# pick a random sample index
idx = random.randint(0, len(ds) - 1)
img_t, gt_solid_t, params_t = ds[idx]

# convert tensors to numpy arrays
img_np       = img_t.squeeze(0).cpu().numpy()
gt_solid_np  = gt_solid_t.squeeze(0).cpu().numpy().astype(bool)
gt_pore_np   = ~gt_solid_np

# load the trained model
model = UNet(n_channels=1, n_classes=1, param_dim=5).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# run inference
with torch.no_grad():
    inp_img   = img_t.unsqueeze(0).to(device)     # (1,1,H,W)
    inp_param = params_t.unsqueeze(0).to(device)  # (1,5)
    logits    = model(inp_img, inp_param)         # (1,1,H,W)
    probs     = torch.sigmoid(logits)[0,0].cpu().numpy()

# ─── COMPUTE P(pore) & THRESHOLDS ───────────────────────────
pore_probs = 1.0 - probs
solid_thr  = 0.90
pore_thr   = 0.10

pred_solid = probs > solid_thr
pred_pore  = pore_probs > pore_thr

# ─── LOGGING ────────────────────────────────────────────────
print(f"Visualizing sample #{idx}, slice #{slice_index}")

# ─── PLOT EVERYTHING ────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Top row: input + GT masks
axes[0,0].imshow(img_np, cmap='gray')
axes[0,0].set_title('Input XCT Slice\n'
                    f'Sample {idx}, Slice {slice_index}')
axes[0,0].axis('off')

axes[0,1].imshow(gt_solid_np, cmap='gray')
axes[0,1].set_title('GT Solid Mask\n'
                    f'Sample {idx}, Slice {slice_index}')
axes[0,1].axis('off')

axes[0,2].imshow(gt_pore_np, cmap='gray')
axes[0,2].set_title('GT Pore Mask\n'
                    f'Sample {idx}, Slice {slice_index}')
axes[0,2].axis('off')

# Bottom row: probabilities + predicted masks
axes[1,0].imshow(probs, cmap='viridis')
axes[1,0].set_title('Predicted P(solid)\n'
                    f'Sample {idx}, Slice {slice_index}')
axes[1,0].axis('off')

axes[1,1].imshow(pred_solid, cmap='gray')
axes[1,1].set_title(f'Pred Solid @ {solid_thr:.2f}\n'
                    f'Sample {idx}, Slice {slice_index}')
axes[1,1].axis('off')

axes[1,2].imshow(pred_pore, cmap='gray')
axes[1,2].set_title(f'Pred Pore @ {pore_thr:.2f}\n'
                    f'Sample {idx}, Slice {slice_index}')
axes[1,2].axis('off')

# super-title for the entire figure
fig.suptitle(f"Sample {idx} — Slice {slice_index}", fontsize=16)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
