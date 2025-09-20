#!/usr/bin/env python3
import numpy as np
from skimage.filters import threshold_otsu
from dataset import CoCrDataset
from train import UNet
import torch
import matplotlib.pyplot as plt

# load one example
ds = CoCrDataset('CoCrOutput', 'CoCrOutput/build_logs.csv', slice_index=0, target_size=(256,256))
img_t, gt_mask_t, _ = ds[0]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4))

img = img_t.squeeze(0).numpy()
gt  = (~gt_mask_t.squeeze(0).numpy().astype(bool))  # pore mask

ax1.imshow(img, cmap='gray')
ax1.set_title('XCT Slice (img_t)')
ax1.axis('off')

ax2.imshow(gt, cmap='gray')
ax2.set_title('GT Mask (gt_mask_t)')
ax2.axis('off')

# 1) Otsu threshold
th = threshold_otsu(img)
baseline = img < th     # pores will be darker

ax3.imshow(baseline, cmap='gray')
ax3.set_title('GT Mask (gt_mask_t)')
ax3.axis('off')

plt.tight_layout()


# compute Dice
inter = 2 * (baseline & gt).sum()
denom = baseline.sum() + gt.sum() + 1e-6
print("Otsu baseline Dice:", inter/denom)

# 2) Your model’s Dice at the learned threshold
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = UNet(1,1,5).to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()
with torch.no_grad():
    logits = model(img_t.unsqueeze(0).to(device), torch.randn(1,5).to(device))
    probs  = torch.sigmoid(logits)[0,0].cpu().numpy()
pore_mask = (1 - probs) > 0.16
inter = 2 * (pore_mask & gt).sum()
denom = pore_mask.sum() + gt.sum() + 1e-6
print("Model Dice:", inter/denom)
plt.show()