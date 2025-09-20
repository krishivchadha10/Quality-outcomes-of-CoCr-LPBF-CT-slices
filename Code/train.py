#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from dataset import CoCrDataset
from tqdm import tqdm
from model import UNet  # ✅ Import UNet from model.py

# -----------------------
# 1. Focal Loss + Dice Loss
# -----------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.exp(-bce)
        loss = self.alpha * ((1 - p_t) ** self.gamma) * bce
        return loss.mean() if self.reduction == 'mean' else loss.sum()

def dice_loss_from_probs(probs, targets, eps=1e-6):
    num = 2 * (probs * targets).sum(dim=(2, 3))
    den = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + eps
    return 1 - (num / den).mean()

# -----------------------
# 2. Training & Validation
# -----------------------
def train_epoch(model, loader, opt, device, focal_loss_fn):
    model.train()
    total_loss = 0.0
    for imgs, masks, params in tqdm(loader, desc="Train"):
        imgs, masks, params = imgs.to(device), masks.to(device), params.to(device)
        opt.zero_grad()
        logits = model(imgs, params)
        fl = focal_loss_fn(logits, masks)
        probs = torch.sigmoid(logits)
        dl = dice_loss_from_probs(probs, masks)
        loss = fl + dl
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def val_epoch(model, loader, device, focal_loss_fn):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, masks, params in tqdm(loader, desc="Val  "):
            imgs, masks, params = imgs.to(device), masks.to(device), params.to(device)
            logits = model(imgs, params)
            fl = focal_loss_fn(logits, masks)
            probs = torch.sigmoid(logits)
            dl = dice_loss_from_probs(probs, masks)
            total_loss += (fl + dl).item()
    return total_loss / len(loader)

# -----------------------
# 3. Main Training Script
# -----------------------
def main():
    base_dir = os.path.dirname(__file__)
    data_root = os.path.join(base_dir, 'CoCrOutput')
    csv_path = os.path.join(data_root, 'build_logs.csv')

    # Hyperparameters
    epochs = 20
    batch_size = 2
    lr = 1e-3
    val_split = 0.2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset & DataLoaders
    dataset = CoCrDataset(data_root, csv_path, slice_index=0, target_size=(256, 256))
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # Model, optimizer, loss
    model = UNet(n_channels=1, n_classes=1, param_dim=5).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr)
    focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

    best_val = float('inf')
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, opt, device, focal_loss_fn)
        val_loss = val_epoch(model, val_loader, device, focal_loss_fn)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(base_dir, 'best_model.pth'))
            print(f"→ New best model (Val Loss {best_val:.4f})")

if __name__ == "__main__":
    main()
