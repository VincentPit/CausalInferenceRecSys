from dataloader.coatLoader import load_coat_data, preprocess_coat
import os
import torch
from vaeModel import *

data_dir = "coat_sg"
raw_data = load_coat_data(data_dir)
data = preprocess_coat(raw_data)

A =  torch.tensor(data['exposure_matrix'])
train_data = data['train']

print(f"A SHAPE:{A.shape}")


os.makedirs("check_point_100k", exist_ok=True)

num_users = data['num_users']
num_items = data['num_items']


num_factors = 100

print("!!!dataset created")

print("Step 1: Train VPF")
pf_model = ExposureVAE( num_items, num_factors)

print("pf_model created")

pf_model = train_exposure_vae(pf_model, A, num_epochs=50)

# Step 1: Train Exposure Encoder (Neural Network)
exposure_encoder = ExposureVAE(num_items)
exposure_encoder = train_exposure_vae(exposure_encoder, A, num_epochs=200, lr = 0.001)

# Step 2: Get Estimated Exposure Probabilities
with torch.no_grad():
    recon, _, _ = exposure_encoder(A)
    exposures_hat = recon.detach()
# Step 3: Train Deconfounded MF (Deep MLP-based)
deep_mf_model = DeepDeconfoundedMF(num_users, num_items, num_factors)
deep_mf_model = train_deep_mf(deep_mf_model, train_data, exposures_hat, num_epochs=400)

# Step 4: Save model checkpoint
torch.save({
    'exposure_encoder_state_dict': exposure_encoder.state_dict(),
    'deep_mf_state_dict': deep_mf_model.state_dict(),
    'exposures_hat': exposures_hat,
}, "check_point_coat/VAEmodel.pt")

print("Neural model saved to check_point_coat/VAEmodel.pt")
