import torch
import torch.nn as nn
import torch.optim as optim

class VariationalPoissonFactorization:
    def __init__(self, num_users, num_items, num_factors, c1=0.3, c2=0.3, c3=0.3, c4=0.3):
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors

        # Variational parameters for users and items
        self.user_shape = torch.full((num_users, num_factors), c1)
        self.user_rate = torch.full((num_users, num_factors), c2)
        self.item_shape = torch.full((num_items, num_factors), c3)
        self.item_rate = torch.full((num_items, num_factors), c4)

    def e_step(self, exposure_data):
        # Update user parameters
        expected_lambda = self.item_shape / self.item_rate  # E[lambda]
        expected_pi = self.user_shape / self.user_rate      # E[pi]

        rate_ui = torch.matmul(expected_pi, expected_lambda.T) + 1e-8

        """for u in range(self.num_users):
            for k in range(self.num_factors):
                numer = (exposure_data[u] * expected_lambda[:, k] / rate_ui[u]).sum()
                self.user_shape[u, k] = 0.3 + numer
                self.user_rate[u, k] = 0.3 + expected_lambda[:, k].sum()"""
        
        # Step 1: Compute the "weights" for each user-item pair
        weights = exposure_data / rate_ui  # shape: (num_users, num_items)

        # Step 2: Update user_shape (num_users, num_factors)
        #         Multiply weights (U×I) with expected_lambda (I×K) → result is (U×K)
        self.user_shape = 0.3 + torch.matmul(weights, expected_lambda)

        # Step 3: Update user_rate (U×K)
        #         Each user gets the same expected_lambda.sum(0), shape (K,)
        self.user_rate = 0.3 + expected_lambda.sum(dim=0).unsqueeze(0).expand_as(self.user_shape)

        expected_lambda = self.item_shape / self.item_rate  # update again
        expected_pi = self.user_shape / self.user_rate
        rate_ui = torch.matmul(expected_pi, expected_lambda.T) + 1e-8

        # Update item parameters
        # Step 1: Compute weights across users for each item
        weights = exposure_data / rate_ui  # shape: (num_users, num_items)

        # Step 2: Transpose weights to (num_items, num_users)
        #         Then matmul with expected_pi: (num_items, U) @ (U, K) → (num_items, K)
        self.item_shape = 0.3 + torch.matmul(weights.T, expected_pi)

        # Step 3: Update item_rate (I×K)
        #         All items get the same sum over users per factor
        self.item_rate = 0.3 + expected_pi.sum(dim=0).unsqueeze(0).expand_as(self.item_shape)

    def reconstruct_exposures(self):
        expected_pi = self.user_shape / self.user_rate
        expected_lambda = self.item_shape / self.item_rate
        return torch.matmul(expected_pi, expected_lambda.T)

class DeconfoundedMatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, num_factors):
        super().__init__()
        self.user_embeddings = nn.Parameter(torch.randn(num_users, num_factors))
        self.item_embeddings = nn.Parameter(torch.randn(num_items, num_factors))
        self.gamma = nn.Parameter(torch.randn(num_users))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, user_ids, item_ids, exposures_hat):
        theta_u = self.user_embeddings[user_ids]
        beta_i = self.item_embeddings[item_ids]
        gamma_u = self.gamma[user_ids]

        dot = (theta_u * beta_i).sum(dim=1)
        prediction = dot + gamma_u * exposures_hat + self.bias
        return prediction

# === Helper Training Functions ===
def train_variational_poisson(pf_model, exposure_data, num_epochs=50):
    for epoch in range(num_epochs):
        pf_model.e_step(exposure_data)
        if epoch % 5 == 0:
            recon = pf_model.reconstruct_exposures()
            loss = ((recon - exposure_data) ** 2).mean()
            print(f"[V-PF Training] Epoch {epoch}, Loss: {loss.item():.4f}")
    return pf_model

def train_deconfounded_mf(mf_model, rating_data, exposures_hat, num_epochs=100, lr=0.001):
    optimizer = optim.Adam(mf_model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        user_ids, item_ids, ratings = rating_data
        exposures_ui = exposures_hat[user_ids, item_ids]

        predictions = mf_model(user_ids, item_ids, exposures_ui)
        loss = criterion(predictions, ratings)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"[Deconfounded MF Training] Epoch {epoch}, MSE Loss: {loss.item():.4f}")
    return mf_model

# === Example Usage ===
if __name__ == "__main__":
    num_users = 1000
    num_items = 1000
    num_factors = 20

    exposure_data = (torch.rand(num_users, num_items) > 0.95).float()

    # Step 1: Train Variational Poisson Factorization
    pf_model = VariationalPoissonFactorization(num_users, num_items, num_factors)
    pf_model = train_variational_poisson(pf_model, exposure_data, num_epochs=50)

    # Step 2: Substitute confounders
    exposures_hat = pf_model.reconstruct_exposures().detach()

    # Step 3: Observed ratings
    observed_indices = exposure_data.nonzero(as_tuple=False)
    user_ids = observed_indices[:, 0]
    item_ids = observed_indices[:, 1]
    ratings = torch.randint(1, 6, (len(user_ids),), dtype=torch.float32)

    # Step 4: Train Deconfounded MF
    mf_model = DeconfoundedMatrixFactorization(num_users, num_items, num_factors)
    mf_model = train_deconfounded_mf(mf_model, (user_ids, item_ids, ratings), exposures_hat, num_epochs=100)

    # Predictions
    user_id = 0
    item_ids = torch.arange(num_items)
    exposures_user = exposures_hat[user_id, item_ids]
    predictions = mf_model(torch.full_like(item_ids, user_id), item_ids, exposures_user)
    print(f"Predictions for User {user_id}:", predictions[:10].detach())
