#!/usr/bin/env python3
import os
import torch
import matplotlib.pyplot as plt
from dataset import CoCrDataset         # ✅ back to original name
from model import UNet                  # ✅ assuming UNet is in model.py

# --- Configuration ---
base_dir = os.path.dirname(__file__)
data_root = os.path.join(base_dir, 'CoCrOutput')
csv_path = os.path.join(data_root, 'build_logs.csv')
model_path = os.path.join(base_dir, 'best_model.pth')
SOLID_THRESHOLD = 0.84  # from threshold_sweep.py

# --- Device ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load dataset ---
dataset = CoCrDataset(
    root_dir=data_root,
    csv_path=csv_path,
    slice_index=0,
    target_size=(256, 256)
)

# --- Pick sample index ---
idx = 0
image, mask, params = dataset[idx]  # image, mask: (1, 256, 256), params: (5,)

# --- Load model ---
model = UNet(n_channels=1, n_classes=1, param_dim=5).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- Run inference ---
with torch.no_grad():
    img_batch = image.unsqueeze(0).to(device)      # (1,1,256,256)
    param_batch = params.unsqueeze(0).to(device)   # (1,5)
    pred = model(img_batch, param_batch)           # (1,1,256,256)

# --- Convert to NumPy ---
pred_np = pred.squeeze().cpu().numpy()     # (256,256)
mask_np = mask.squeeze().cpu().numpy()     # (256,256)
img_np  = image.squeeze().cpu().numpy()    # (256,256)

# --- Apply threshold ---
pred_mask = (pred_np > SOLID_THRESHOLD).astype(float)

# --- Optional: Plot prediction histogram ---
plt.figure(figsize=(6, 3))
plt.hist(pred_np.ravel(), bins=50, color='gray', edgecolor='black')
plt.title("Histogram of Predicted P(solid)")
plt.xlabel("Probability")
plt.ylabel("Pixel Count")
plt.tight_layout()
plt.show()

# --- Optional: Dice score ---
intersection = (pred_mask * mask_np).sum()
dice = 2 * intersection / (pred_mask.sum() + mask_np.sum() + 1e-6)
print(f"Dice score: {dice:.4f}")

# --- Plot side-by-side results ---
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(img_np, cmap='gray')
axes[0].set_title('Input XCT Slice')
axes[0].axis('off')

axes[1].imshow(mask_np, cmap='gray')
axes[1].set_title('Ground Truth Mask')
axes[1].axis('off')

axes[2].imshow(pred_mask, cmap='gray')
axes[2].set_title(f'Predicted Mask\n(Threshold = {SOLID_THRESHOLD})')
axes[2].axis('off')

plt.tight_layout()
plt.show()
