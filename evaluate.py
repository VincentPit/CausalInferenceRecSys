import torch
import numpy as np
from sklearn.metrics import accuracy_score
import neuModel
import ogModel
def precision_at_k(predictions, ground_truth, k):
    """Compute Precision@K."""
    top_k_preds = predictions.argsort(descending=True)[:k]
    relevant_items = ground_truth.nonzero(as_tuple=False).squeeze()
    hits = len(set(top_k_preds.tolist()) & set(relevant_items.tolist()))
    return hits / k

def recall_at_k(predictions, ground_truth, k):
    """Compute Recall@K."""
    top_k_preds = predictions.argsort(descending=True)[:k]
    relevant_items = ground_truth.nonzero(as_tuple=False).squeeze()
    hits = len(set(top_k_preds.tolist()) & set(relevant_items.tolist()))
    return hits / len(relevant_items) if len(relevant_items) > 0 else 0

def ndcg_at_k(predictions, ground_truth, k):
    """Compute NDCG@K."""
    top_k_preds = predictions.argsort(descending=True)[:k]
    dcg = 0.0
    idcg = 0.0
    relevant_items = ground_truth.nonzero(as_tuple=False).squeeze()
    for i, pred in enumerate(top_k_preds):
        if pred in relevant_items:
            dcg += 1 / np.log2(i + 2)  # log2(i+2) because i is 0-indexed
    for i in range(min(len(relevant_items), k)):
        idcg += 1 / np.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0

def evaluate_model(model, test_data, exposures_hat, k=10):
    """Evaluate the model using Accuracy, Recall, Precision, and NDCG."""
    user_ids, item_ids, ground_truth_ratings = test_data
    predictions = []

    # Generate predictions for all user-item pairs in the test set
    for user_id, item_id in zip(user_ids, item_ids):
        exposures_ui = exposures_hat[user_id, item_id]
        prediction = model(torch.tensor([user_id]), torch.tensor([item_id]), torch.tensor([exposures_ui]))
        predictions.append(prediction.item())

    predictions = torch.tensor(predictions)
    ground_truth = (ground_truth_ratings > 3).float()  # Binary relevance: 1 if rating > 3, else 0

    # Compute metrics
    accuracy = accuracy_score(ground_truth.numpy(), (predictions > 3).numpy())
    precision = precision_at_k(predictions, ground_truth, k)
    recall = recall_at_k(predictions, ground_truth, k)
    ndcg = ndcg_at_k(predictions, ground_truth, k)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision@{k}: {precision:.4f}")
    print(f"Recall@{k}: {recall:.4f}")
    print(f"NDCG@{k}: {ndcg:.4f}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "ndcg": ndcg
    }

if __name__ == "__main__":
    # Load test data and model
    checkpoint = torch.load("check_point_100k/NNmodel.pt")
    exposure_encoder = neuModel.ExposureEncoderNN(num_users=943, num_items=1682)
    deep_mf_model = neuModel.DeepDeconfoundedMF(num_users=943, num_items=1682, num_factors=100)

    exposure_encoder.load_state_dict(checkpoint['exposure_encoder_state_dict'])
    deep_mf_model.load_state_dict(checkpoint['deep_mf_state_dict'])

    # Load test data
    from dataloader.movieLoader import load_movielens_100k, preprocess_movielens
    df = load_movielens_100k("./dataloader/ml-100k/u.data")
    data = preprocess_movielens(df)
    test_data = data['test']
    exposures_hat = checkpoint['exposures_hat']

    # Evaluate the model
    evaluate_model(deep_mf_model, test_data, exposures_hat, k=10)