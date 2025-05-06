import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from scipy.stats import invgamma

DATA_DIR = '../coat'
OUT_DATA_DIR = '../coat_sg'

# Load sparse data
raw_data = pd.read_csv(os.path.join(DATA_DIR, 'train.ascii'), sep=" ", header=None, engine="python")
test_data = pd.read_csv(os.path.join(DATA_DIR, 'test.ascii'), sep=" ", header=None, engine="python")

raw_data = pd.DataFrame({
    "userId": sparse.coo_matrix(raw_data).row,
    "songId": sparse.coo_matrix(raw_data).col,
    "rating": sparse.coo_matrix(raw_data).data
})

test_data = pd.DataFrame({
    "userId": sparse.coo_matrix(test_data).row,
    "songId": sparse.coo_matrix(test_data).col,
    "rating": sparse.coo_matrix(test_data).data
})


def split_train_test_proportion(data, uid, test_prop=0.5, random_seed=0, n_items_thresh=5):
    
    
    data_grouped_by_user = data.groupby(uid)
    
    print("length of data_grouped_by_user:", len(data_grouped_by_user))
    
    tr_list, te_list = [], []

    np.random.seed(random_seed)

    for u, (_, group) in enumerate(data_grouped_by_user):
        n_items_u = len(group)

        if n_items_u >= n_items_thresh:
            idx = np.zeros(n_items_u, dtype=bool)
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype(int)] = True
            tr_list.append(group[~idx])
            te_list.append(group[idx])
        else:
            tr_list.append(group)

        if u % 5000 == 0 and u !=0:
            print(f"{u} users sampled")
            sys.stdout.flush()

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te


def get_count(tp, id):
    return tp[[id]].groupby(id, as_index=False).size()


user_activity = get_count(raw_data, 'userId')
unique_uid = user_activity.index

np.random.seed(98765)
idx_perm = np.random.permutation(unique_uid.size)
unique_uid = unique_uid[idx_perm]

n_users = unique_uid.size

tr_users = unique_uid[:(n_users - int(0.4 * n_users))]
vd_users = unique_uid[(n_users - int(0.4 * n_users)): (n_users - int(0.2 * n_users))]
te_users = unique_uid[(n_users - int(0.2 * n_users)):]

train_plays = raw_data[raw_data['userId'].isin(tr_users)]
unique_sid = pd.unique(train_plays['songId'])

song2id = {sid: i for i, sid in enumerate(unique_sid)}
user2id = {uid: i for i, uid in enumerate(unique_uid)}

# Create output directory
os.makedirs(OUT_DATA_DIR, exist_ok=True)

with open(os.path.join(OUT_DATA_DIR, 'unique_uid.txt'), 'w') as f:
    for uid in unique_uid:
        f.write(f"{uid}\n")

with open(os.path.join(OUT_DATA_DIR, 'unique_sid.txt'), 'w') as f:
    for sid in unique_sid:
        f.write(f"{sid}\n")

# Validation data
vad_plays = raw_data[raw_data['userId'].isin(vd_users)]
vad_plays = vad_plays[vad_plays['songId'].isin(unique_sid)]
vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays, 'userId', test_prop=0.5, random_seed=13579)

# Test data (observational)
test_plays = raw_data[raw_data['userId'].isin(te_users)]
test_plays = test_plays[test_plays['songId'].isin(unique_sid)]
test_plays_tr, test_plays_te = split_train_test_proportion(test_plays, 'userId', test_prop=0.5, random_seed=13579)

# Test data (random)
rand_test_plays = test_data[test_data['userId'].isin(te_users)]
rand_test_plays = rand_test_plays[rand_test_plays['songId'].isin(unique_sid)]
rand_test_plays_tr, rand_test_plays_te = split_train_test_proportion(rand_test_plays, 'userId', test_prop=0.5, random_seed=13579)

# Output dataset sizes
print(len(train_plays), len(vad_plays), len(test_plays))
print(len(vad_plays_tr), len(vad_plays_te))
print(len(test_plays_tr), len(test_plays_te))
print(len(rand_test_plays_tr), len(rand_test_plays_te))


def numerize(tp):
    uid = list(map(lambda x: user2id[x], tp['userId']))
    sid = list(map(lambda x: song2id[x], tp['songId']))
    tp = tp.copy()
    tp.loc[:, 'uid'] = uid
    tp.loc[:, 'sid'] = sid
    return tp[['uid', 'sid', 'rating']]


# Save CSVs
numerize(train_plays).to_csv(os.path.join(OUT_DATA_DIR, 'train.csv'), index=False)
numerize(vad_plays_tr).to_csv(os.path.join(OUT_DATA_DIR, 'validation_tr.csv'), index=False)
numerize(vad_plays_te).to_csv(os.path.join(OUT_DATA_DIR, 'validation_te.csv'), index=False)
numerize(test_plays_tr).to_csv(os.path.join(OUT_DATA_DIR, 'obs_test_tr.csv'), index=False)
numerize(test_plays_te).to_csv(os.path.join(OUT_DATA_DIR, 'obs_test_te.csv'), index=False)
numerize(rand_test_plays_tr).to_csv(os.path.join(OUT_DATA_DIR, 'test_tr.csv'), index=False)
numerize(rand_test_plays_te).to_csv(os.path.join(OUT_DATA_DIR, 'test_te.csv'), index=False)
