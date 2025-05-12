import os
import torch
from ogModel import *
from dataloader.movieLoader import *

os.makedirs("check_point_100k", exist_ok=True)

# Load and preprocess data
num_users, num_items, df = load_movielens_1m("ml-1m/ratings.dat")
data = preprocess_movielens(df)
A =  data['exposure_matrix']
train_data = data['train']

num_factors = 100

# Step 1: Train VPF
pf_model = VariationalPoissonFactorization(num_users, num_items, num_factors)
pf_model = train_variational_poisson(pf_model, A, num_epochs=150)

# Step 2: Reconstruct exposures
exposures_hat = pf_model.reconstruct_exposures().detach()

# Step 3: Train Deconfounded MF
mf_model = DeconfoundedMatrixFactorization(num_users, num_items, num_factors)
mf_model = train_deconfounded_mf(mf_model, train_data, exposures_hat, num_epochs=2000)

# Step 4: Save model checkpoint
torch.save({
    'user_embeddings': mf_model.user_embeddings.detach(),
    'item_embeddings': mf_model.item_embeddings.detach(),
    'gamma': mf_model.gamma.detach(),
    'bias': mf_model.bias.detach(),
    'exposures_hat': exposures_hat,
}, "check_point_1m/model.pt")

print("Model saved to check_point_1m/OGmodel.pt")