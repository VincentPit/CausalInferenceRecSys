from dataloader.coatLoader import load_coat_data, preprocess_coat
import os
import torch
from neuModel import *

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

# Step 1: Train Exposure Encoder (Neural Network)
exposure_encoder = ExposureEncoderNN(num_users, num_items)
exposure_encoder = train_exposure_encoder(exposure_encoder, A, num_epochs=100, lr = 0.001)

# Step 2: Get Estimated Exposure Probabilities
with torch.no_grad():
    exposures_hat = exposure_encoder(A).detach()  # shape: (num_users, num_items)

# Step 3: Train Deconfounded MF (Deep MLP-based)
deep_mf_model = DeepDeconfoundedMF(num_users, num_items, num_factors)
deep_mf_model = train_deep_mf(deep_mf_model, train_data, exposures_hat, num_epochs=200)

# Step 4: Save model checkpoint
torch.save({
    'exposure_encoder_state_dict': exposure_encoder.state_dict(),
    'deep_mf_state_dict': deep_mf_model.state_dict(),
    'exposures_hat': exposures_hat,
}, "check_point_coat/NNmodel.pt")

print("Neural model saved to check_point_coat/NNmodel.pt")