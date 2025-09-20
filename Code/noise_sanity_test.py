import torch
from train import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = UNet(1,1,5).to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# create noise of the same shape as a real slice
dummy_img    = torch.randn(1,1,256,256, device=device)
dummy_params = torch.randn(1,5,      device=device)

with torch.no_grad():
    logits = model(dummy_img, dummy_params)
    probs  = torch.sigmoid(logits)[0,0].cpu().numpy()

print("Noise→P(solid) min/max:", probs.min(), probs.max())
# If the model learned structure, these will be scattered; 
# if it’s just copying the “disk” shape, you’d see near-1s everywhere.
