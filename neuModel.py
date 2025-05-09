import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# =========================
# Neural Exposure Encoder
# =========================
class ExposureEncoderNN(nn.Module):
    def __init__(self, num_users, num_items, hidden_dim=128, num_layers=3):
        super().__init__()
        layers = [nn.Linear(num_items, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, num_items), nn.Sigmoid()]
        self.encoder = nn.Sequential(*layers)

    def forward(self, exposure_data):
        return self.encoder(exposure_data)

# =========================
# Deep Deconfounded Matrix Factorization
# =========================
class DeepDeconfoundedMF(nn.Module):
    def __init__(self, num_users, num_items, num_factors):
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, num_factors)
        self.item_embeddings = nn.Embedding(num_items, num_factors)

        self.mlp = nn.Sequential(
            nn.Linear(num_factors * 2 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            
        )
        """ nn.ReLU(),
            nn.Linear(64, 1)
        )"""

    def forward(self, user_ids, item_ids, exposures_hat):
        theta_u = self.user_embeddings(user_ids)
        beta_i = self.item_embeddings(item_ids)
        x = torch.cat([theta_u, beta_i, exposures_hat.unsqueeze(1)], dim=1)
        #print("X shape:", x.shape)
        return self.mlp(x).squeeze()
"""class DeepDeconfoundedMF(nn.Module):
    def __init__(self, num_users, num_items, num_factors):
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, num_factors)
        self.item_embeddings = nn.Embedding(num_items, num_factors)

        self.mlp = nn.Sequential(
            nn.Linear(num_factors * 2 + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, user_ids, item_ids, exposures_hat):
        theta_u = self.user_embeddings(user_ids)
        beta_i = self.item_embeddings(item_ids)
        x = torch.cat([theta_u, beta_i, exposures_hat.unsqueeze(1)], dim=1)
        return self.mlp(x).squeeze()"""

# =========================
# Training Functions
# =========================
def train_exposure_encoder(encoder, exposure_data, num_epochs=50, lr=0.001):
    optimizer = optim.Adam(encoder.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        #print(f"Input shape to encoder: {exposure_data.shape}") 
        pred_exposures = encoder(exposure_data)
        loss = criterion(pred_exposures, exposure_data)
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            print(f"[Exposure Encoder] Epoch {epoch}, BCE Loss: {loss.item():.4f}")
    return encoder

def train_deep_mf(model, rating_data, exposures_hat, num_epochs=100, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        user_ids, item_ids, ratings = rating_data
        exposures_ui = exposures_hat[user_ids, item_ids]

        predictions = model(user_ids, item_ids, exposures_ui)
        loss = criterion(predictions, ratings)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"[Deep MF] Epoch {epoch}, MSE Loss: {loss.item():.4f}")
    return model

# =========================
# Main Training Script
# =========================
if __name__ == "__main__":
    torch.manual_seed(42)
    num_users = 1000
    num_items = 1000
    num_factors = 20

    # Generate synthetic exposure data
    exposure_data = (torch.rand(num_users, num_items) > 0.95).float()

    # Step 1: Train neural exposure encoder
    exposure_encoder = ExposureEncoderNN(num_users, num_items)
    exposure_encoder = train_exposure_encoder(exposure_encoder, exposure_data, num_epochs=50)

    with torch.no_grad():
        exposures_hat = exposure_encoder(exposure_data).detach()

    # Step 2: Simulate ratings
    observed_indices = exposure_data.nonzero(as_tuple=False)
    user_ids = observed_indices[:, 0]
    item_ids = observed_indices[:, 1]
    ratings = torch.randint(1, 6, (len(user_ids),), dtype=torch.float32)

    # Step 3: Train deep deconfounded MF
    deep_mf_model = DeepDeconfoundedMF(num_users, num_items, num_factors)
    deep_mf_model = train_deep_mf(deep_mf_model, (user_ids, item_ids, ratings), exposures_hat, num_epochs=100)

    # Step 4: Save model parameters
    torch.save({
        'exposure_encoder': exposure_encoder.state_dict(),
        'deep_mf_model': deep_mf_model.state_dict()
    }, "check_point_100k.pth")
    print("Model parameters saved to 'check_point_100k.pth'")

    # Step 5: Example inference
    user_id = 0
    item_ids = torch.arange(num_items)
    with torch.no_grad():
        exposures_user = exposures_hat[user_id, item_ids]
        predictions = deep_mf_model(torch.full_like(item_ids, user_id), item_ids, exposures_user)
    print(f"Predictions for User {user_id}:", predictions[:10])
