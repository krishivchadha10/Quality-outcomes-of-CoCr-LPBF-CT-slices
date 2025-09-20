# visualize_external_ct.py
#!/usr/bin/env python3
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from train import UNet

# --- CONFIGURATION ------------------------------------------------
# Path to your external XCT volume (.npy array)
image_path = r"C:\Users\krish\OneDrive\Desktop\ML Internship\XCTPore\XCTPore\MA002_train.npy"
# If you want a different slice, set SLICE_INDEX; default is center slice
SLICE_INDEX = None  # e.g. 10 or None for center

# Model and threshold
model_path = os.path.join(os.path.dirname(__file__), 'best_model.pth')
pore_thr   = 0.16

def load_slice(path, slice_index=None):
    """
    Load a 3D .npy volume and return a single 2D slice.
    If slice_index is None, use the middle slice.
    """
    vol = np.load(path)
    if vol.ndim == 3:
        # Determine slice index
        idx = slice_index if slice_index is not None else vol.shape[0] // 2
        slice2d = vol[idx]
    elif vol.ndim == 2:
        slice2d = vol
    else:
        raise ValueError(f"Unexpected array dim {vol.ndim}, expected 2 or 3.")
    return slice2d

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- LOAD & PREPROCESS SLICE --------------------------------------
arr = load_slice(image_path, SLICE_INDEX).astype(np.float32)
# Normalize to [0,1]
arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
# Convert to torch tensor (1,1,H,W) and resize to 256×256
t = torch.from_numpy(arr[None, None, ...])
t = F.interpolate(t, size=(256,256), mode='bilinear', align_corners=False)

# --- LOAD MODEL ---------------------------------------------------
model = UNet(n_channels=1, n_classes=1, param_dim=5).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Dummy process parameters
dummy_params = torch.zeros(1,5, device=device)

# --- INFERENCE ----------------------------------------------------
with torch.no_grad():
    logits = model(t.to(device), dummy_params)
    probs  = torch.sigmoid(logits)[0,0].cpu().numpy()
# Compute pore mask from P(pore) = 1 - P(solid)
pore_probs = 1.0 - probs
pore_mask  = pore_probs > pore_thr

# Resize mask back to original slice dimensions
orig_h, orig_w = arr.shape
mask_img = Image.fromarray((pore_mask.astype(np.uint8) * 255))
mask_img = mask_img.resize((orig_w, orig_h), Image.NEAREST)
mask_np  = np.array(mask_img) / 255

# --- PLOT RESULTS -------------------------------------------------
fig, axes = plt.subplots(1,2, figsize=(10,5))
axes[0].imshow(arr, cmap='gray')
axes[0].set_title('Input XCT Slice')
axes[0].axis('off')

axes[1].imshow(mask_np, cmap='gray')
axes[1].set_title(f'Predicted Pore Mask @ {pore_thr:.2f}')
axes[1].axis('off')

plt.tight_layout()
plt.show()
