import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix

def load_movielens_1m(path="ml-1m/ratings.dat"):
    """
    Load the MovieLens 1M dataset and return number of users and items.

    Parameters:
    - path (str): Path to the ratings.dat file.

    Returns:
    - num_users (int): Number of unique users.
    - num_items (int): Number of unique items.
    - df (pd.DataFrame): Loaded DataFrame.
    """
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
        encoding="ISO-8859-1"
    )

    df["user_id"] -= 1  # make zero-indexed if desired
    df["item_id"] -= 1

    num_users = df["user_id"].nunique()
    num_items = df["item_id"].nunique()
    return num_users, num_items, df

def load_movielens_100k(path="ml-100k/u.data"):
    # Columns: user_id, item_id, rating, timestamp
    df = pd.read_csv(path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    df['user_id'] -= 1  # make zero-indexed
    df['item_id'] -= 1
    return df

def preprocess_movielens(df, test_size=0.2, seed=42):
    # Reassign continuous IDs
    user_mapping = {user: idx for idx, user in enumerate(df['user_id'].unique())}
    item_mapping = {item: idx for idx, item in enumerate(df['item_id'].unique())}
    
    # Update user_id and item_id to be continuous
    df['user_id'] = df['user_id'].map(user_mapping)
    df['item_id'] = df['item_id'].map(item_mapping)

    num_users = len(user_mapping)
    num_items = len(item_mapping)

    # Exposure matrix A: 1 if rating exists, else 0
    A = torch.zeros((num_users, num_items), dtype=torch.float32)
    for row in df.itertuples():
        A[row.user_id, row.item_id] = 1.0

    # Rating matrix Y: only defined where exposure exists
    Y = torch.full((num_users, num_items), float('nan'))
    for row in df.itertuples():
        Y[row.user_id, row.item_id] = float(row.rating)

    # Create train/test split based on observed ratings
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)

    # Create training data tuples
    train_user_ids = torch.tensor(train_df['user_id'].values, dtype=torch.long)
    train_item_ids = torch.tensor(train_df['item_id'].values, dtype=torch.long)
    train_ratings = torch.tensor(train_df['rating'].values, dtype=torch.float32)

    # Create test data tuples
    test_user_ids = torch.tensor(test_df['user_id'].values, dtype=torch.long)
    test_item_ids = torch.tensor(test_df['item_id'].values, dtype=torch.long)
    test_ratings = torch.tensor(test_df['rating'].values, dtype=torch.float32)

    return {
        'exposure_matrix': A,
        'rating_matrix': Y,
        'train': (train_user_ids, train_item_ids, train_ratings),
        'test': (test_user_ids, test_item_ids, test_ratings),
        'num_users': num_users,
        'num_items': num_items
    }


if __name__ == "__main__":
    df = load_movielens_100k("../ml-100k/u.data")
    data = preprocess_movielens(df)

    A = data['exposure_matrix']       # binary matrix for PF
    Y = data['rating_matrix']         # actual ratings
    train_data = data['train']
    test_data = data['test']

    print("Exposure shape:", A.shape)
    print("Train samples:", len(train_data[0]))
    print("Test samples:", len(test_data[0]))
