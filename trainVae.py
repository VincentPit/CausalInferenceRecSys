import os
import torch
from vaeModel import ExposureVAE, DeepDeconfoundedMF, train_exposure_vae, train_deep_mf
from dataloader.movieLoader import load_movielens_100k, preprocess_movielens

# Ensure checkpoint directory exists
os.makedirs("check_point_100k", exist_ok=True)

# Load and preprocess MovieLens 100k dataset
df = load_movielens_100k("ml-100k/u.data")
data = preprocess_movielens(df)
A = data['exposure_matrix']             # shape: (num_users, num_items)
train_data = data['train']              # tuple: (user_ids, item_ids, ratings)
num_users = data['num_users']
num_items = data['num_items']
num_factors = 100

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
}, "check_point_100k/VAEmodel.pt")

print("Neural model saved to check_point_100k/VAEmodel.pt")
