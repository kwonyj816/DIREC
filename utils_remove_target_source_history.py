import os, random

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp
from collections import defaultdict
import pickle


def mk_folders(run_name):
    # Create model output directories.
    os.makedirs("models", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)


def set_seed(s, reproducible=False):
    # Seed Python/NumPy/PyTorch RNGs for reproducibility.
    try: torch.manual_seed(s)
    except NameError: pass
    try: torch.cuda.manual_seed_all(s)
    except NameError: pass
    try: np.random.seed(s%(2**32-1))
    except NameError: pass
    random.seed(s)
    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def collate_fn(batch):
    # Merge a list of samples into a single batch dict.
    batched = {}

    # 1) user–item pairs as a list of tuples
    batched['pair'] = [sample['pair'] for sample in batch]

    # 2) fixed-size embeddings → stack
    batched['target_review'] = torch.stack([s['target_review'] for s in batch], dim=0)
    batched['target_item_embedding'] = torch.stack([s['target_item_embedding'] for s in batch], dim=0)
    batched['target_rating'] = torch.stack([s['target_rating'] for s in batch], dim=0)
    
    batched['num_target_item_interact'] = torch.tensor([s['num_target_item_interact'] for s in batch], dtype=torch.long)

    # 3) variable-length per-user lists → keep as list-of-Tensors
    batched['source_reviews'] = [s['source_reviews'] for s in batch]
    batched['source_item_embeddings'] = [s['source_item_embeddings'] for s in batch]

    return batched


class EmbeddingDataset(Dataset):
    """
    Dataset for cross-domain recommendation using precomputed embeddings.

    Args:
        source_emb_dict (dict): mapping (user_id, item_id) -> np.ndarray
        target_emb_dict (dict): mapping (user_id, item_id) -> np.ndarray
        source_item_emb (np.ndarray): shape (n_source_items, emb_dim)
        target_item_emb (np.ndarray): shape (n_target_items, emb_dim)
        target_rating_mat (scipy.sparse.csr_matrix): user x item ratings
        pairs (list[(user, item) pairs])
    """
    def __init__(self,
                 args,
                 source_emb_dict,
                 target_emb_dict,
                 source_item_emb,
                 target_item_emb,
                 target_rating_mat: sp.csr_matrix,
                 pairs):
        # Initialize dataset with embedding dicts, items, and (user,item) pairs.
        
        self.source_emb_dict = source_emb_dict
        self.target_emb_dict = target_emb_dict
        # convert global item‐embedding arrays once
        self.source_item_emb = torch.from_numpy(source_item_emb).float()
        self.target_item_emb = torch.from_numpy(target_item_emb).float()
        
        # rating_mat: keep it in CSR for lookup
        self.rating_mat = target_rating_mat.tocsr()
        # self.source_time_mat = source_time_mat
        self.pairs = pairs
        
        self.history_len = args.history_len
        self.source_user_dict = defaultdict(list)
        for (u, i), emb_np in source_emb_dict.items():
            self.source_user_dict[u].append((i, torch.from_numpy(emb_np).float()))


        n_items = target_rating_mat.shape[1]
        self.num_target_item_interacts = [0] * n_items

        for _, item_id in self.pairs:
            self.num_target_item_interacts[item_id] += 1

        
    def __len__(self):
        # Return number of user-item pairs.
        return len(self.pairs)

    def __getitem__(self, idx):
        # Build one training sample from embeddings and user history.
        user_id, item_id = self.pairs[idx]

        target_review_np = self.target_emb_dict[(user_id, item_id)]
        target_review = torch.from_numpy(target_review_np).float()

        target_item_embedding = self.target_item_emb[item_id]

        source_items_and_embs = self.source_user_dict[user_id]
        source_item_indices, source_reviews = zip(*source_items_and_embs)

        n_item = len(source_item_indices)

        if n_item > self.history_len:
            selected_idx = torch.randperm(n_item)[:self.history_len]
            source_item_indices = [source_item_indices[i] for i in selected_idx]
            source_reviews = [source_reviews[i] for i in selected_idx]
        source_reviews = torch.stack(source_reviews, dim=0)  # (n_item or history_len, emb_dim)
        source_item_embeddings = self.source_item_emb[list(source_item_indices)]

        # — Target rating (scalar)
        rating = float(self.rating_mat[user_id, item_id])
        target_rating = torch.tensor(rating).float()

        num_target_item_interact = self.num_target_item_interacts[item_id]
        return {
            'pair': (user_id, item_id),
            'target_review': target_review,                       # [emb_dim]
            'source_reviews': source_reviews,                     # [n_src, emb_dim]
            'target_item_embedding': target_item_embedding,       # [emb_dim]
            'source_item_embeddings': source_item_embeddings,     # [n_src, emb_dim]
            'target_rating': target_rating,                       # []
            'num_target_item_interact': num_target_item_interact
        }


def get_data(args):
    # Load embeddings, split users, and build dataloaders.
    data_path = os.path.join(args.data_path, args.dataset)
    test_ratio = args.cold_start_ratio
    valid_ratio = args.valid_ratio

    source_embedding_dict = np.load(os.path.join(data_path, "source_embedding_dict.npy"), allow_pickle=True).item()
    target_embedding_dict = np.load(os.path.join(data_path, "target_embedding_dict.npy"), allow_pickle=True).item()
    
    source_item_embedding = np.load(os.path.join(data_path, "embedding_source_item_avg.npy"))

    target_rating_mat = sp.load_npz(os.path.join(data_path, "rating_matrix_target.npz"))
    target_rating_mat = sp.csr_matrix(target_rating_mat)

    
    n_user = target_rating_mat.shape[0]
    n_items = target_rating_mat.shape[1]

    with open(os.path.join(data_path, f"train_indices_{args.dataset}_{test_ratio}.pkl"), "rb") as f:
        train_indices = pickle.load(f)
    with open(os.path.join(data_path, f"val_indices_{args.dataset}_{test_ratio}.pkl"), "rb") as f:
        val_indices = pickle.load(f)
    with open(os.path.join(data_path, f"test_indices_{args.dataset}_{test_ratio}.pkl"), "rb") as f:
        test_indices = pickle.load(f)

    train_pairs = []
    val_pairs = []
    test_pairs = []
    

    
    for (user, item), emb in target_embedding_dict.items():
        if user in train_indices:
            train_pairs.append((user, item))
        elif user in val_indices:
            val_pairs.append((user, item))
        elif user in test_indices:
            test_pairs.append((user, item))

    num_target_item_interacts = [0] * n_items
    for _, item_id in train_pairs:
        num_target_item_interacts[item_id] += 1
    filtered_train_pairs = [
    (user, item_id)
    for (user, item_id) in train_pairs
    if num_target_item_interacts[item_id] > 1]

    train_pairs = filtered_train_pairs

    target_item_embedding = np.zeros((n_items, source_item_embedding.shape[1]), dtype=np.float32)
    item_to_embeddings = {}

    for (user_id, item_id) in train_pairs:
        if (user_id, item_id) in target_embedding_dict:
            emb = target_embedding_dict[(user_id, item_id)]
            if item_id not in item_to_embeddings:
                item_to_embeddings[item_id] = []
            item_to_embeddings[item_id].append(emb)

    for item_id, embeddings in item_to_embeddings.items():
        if len(embeddings) > 0:
            avg_emb = np.mean(embeddings, axis=0)
            target_item_embedding[item_id] = avg_emb

    train_dataset = EmbeddingDataset(args, source_embedding_dict, target_embedding_dict, source_item_embedding, target_item_embedding, target_rating_mat, train_pairs)
    val_dataset = EmbeddingDataset(args, source_embedding_dict, target_embedding_dict, source_item_embedding, target_item_embedding, target_rating_mat, val_pairs)
    test_dataset = EmbeddingDataset(args, source_embedding_dict, target_embedding_dict, source_item_embedding, target_item_embedding, target_rating_mat, test_pairs)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)


    return train_loader, val_loader, test_loader, n_user, n_items



def compute_rec(predicted_rating, target_rating):
    # Compute summed MAE and MSE for a batch.
    sum_abs_err = F.l1_loss(predicted_rating, target_rating, reduction='sum')
    sum_sqr_err = F.mse_loss(predicted_rating, target_rating, reduction='sum')
    return sum_abs_err.item(), sum_sqr_err.item()
