from dataloader.coatLoader import load_coat_data, preprocess_coat
import os
import torch
from ogModel import *

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
pf_model = VariationalPoissonFactorization(num_users, num_items, num_factors)

print("pf_model created")

pf_model = train_variational_poisson(pf_model, A, num_epochs=50)

print("Step 2: Reconstruct exposures")
exposures_hat = pf_model.reconstruct_exposures().detach()

print("Step 3: Train Deconfounded MF")
mf_model = DeconfoundedMatrixFactorization(num_users, num_items, num_factors)
mf_model = train_deconfounded_mf(mf_model, train_data, exposures_hat, num_epochs=100)

print("Step 4: Save model checkpoint")
torch.save({
    'user_embeddings': mf_model.user_embeddings.detach(),
    'item_embeddings': mf_model.item_embeddings.detach(),
    'gamma': mf_model.gamma.detach(),
    'bias': mf_model.bias.detach(),
    'exposures_hat': exposures_hat,
}, "check_point_coat/model.pt")

print("Model saved to check_point_coat/OGmodel.pt")