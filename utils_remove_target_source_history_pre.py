from ast import arg
import pickle
import os, random

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import scipy.sparse as sp
from collections import defaultdict


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
    """
    Collate a list of samples (dicts) into a single batched dict.
    Fixed-size fields are stacked into tensors; variable-length
    fields remain as lists of tensors.
    """
    batched = {}
    batched['nonoverlap_target_user_review'] = torch.stack([s['nonoverlap_target_user_review'] for s in batch], dim=0)
    batched['nonoverlap_target_user_rating'] = torch.stack([s['nonoverlap_target_user_rating'] for s in batch], dim=0)
    batched['nonoverlap_target_user_review_item_id'] = torch.stack([s['nonoverlap_target_user_review_item_id'] for s in batch], dim=0)

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
                 target_rating_mat: sp.csr_matrix,
                 nonoverlap_target_user_review,
                 nonoverlap_target_user_ratings,
                 nonoverlap_target_user_review_item_ids):
        # Initialize dataset for non-overlapping target domain user samples.
        
        # rating_mat: keep it in CSR for lookup
        self.rating_mat = target_rating_mat.tocsr()

        self.nonoverlap_target_user_review = nonoverlap_target_user_review
        self.nonoverlap_target_user_ratings = nonoverlap_target_user_ratings
        self.nonoverlap_target_user_review_item_ids = nonoverlap_target_user_review_item_ids
        self.n_nonoverlap_user = nonoverlap_target_user_review.shape[0]


        
    def __len__(self):
        # Return number of non-overlapping users.
        return self.n_nonoverlap_user

    def __getitem__(self, idx):
        # Return one non-overlapping user review, rating, and item id sample.

        nonoverlap_target_user_review = self.nonoverlap_target_user_review[idx]
        nonoverlap_target_user_rating = self.nonoverlap_target_user_ratings[idx]
        nonoverlap_target_user_review_item_id = self.nonoverlap_target_user_review_item_ids[idx]

        return {
            'nonoverlap_target_user_review': nonoverlap_target_user_review,
            'nonoverlap_target_user_rating': nonoverlap_target_user_rating,
            'nonoverlap_target_user_review_item_id': nonoverlap_target_user_review_item_id
        }


def get_data(args):
    # Load non-overlap user review, rating, and item id embeddings and build train/val dataloaders.
    data_path = os.path.join(args.data_path, args.dataset)
    valid_ratio = args.valid_ratio
    

    target_rating_mat = sp.load_npz(os.path.join(data_path, "rating_matrix_target.npz"))
    target_rating_mat = sp.csr_matrix(target_rating_mat)
    
    n_user = target_rating_mat.shape[0]
    n_items = target_rating_mat.shape[1]

    
    nonoverlap_target_user_review = np.load(os.path.join(data_path, "target_non_overlapping_user_review_embedding.npy"))
    nonoverlap_target_user_review = torch.from_numpy(nonoverlap_target_user_review).float()

    nonoverlap_target_user_ratings = np.load(os.path.join(data_path, "target_non_overlapping_user_ratings.npy"))
    nonoverlap_target_user_ratings = torch.from_numpy(nonoverlap_target_user_ratings).float()

    nonoverlap_target_user_review_item_ids = np.load(os.path.join(data_path, "target_non_overlapping_user_review_item_ids.npy"))
    nonoverlap_target_user_review_item_ids = torch.from_numpy(nonoverlap_target_user_review_item_ids).long()

    
    full_dataset = EmbeddingDataset(args, target_rating_mat, nonoverlap_target_user_review, nonoverlap_target_user_ratings, nonoverlap_target_user_review_item_ids)
    val_size = int(len(full_dataset) * valid_ratio)
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(args.seed) if hasattr(args, "seed") else None
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)


    return train_loader, val_loader, n_user, n_items



def compute_rec(predicted_rating, target_rating):
    # Compute summed MAE and MSE for a batch.
    sum_abs_err = F.l1_loss(predicted_rating, target_rating, reduction='sum')
    sum_sqr_err = F.mse_loss(predicted_rating, target_rating, reduction='sum')
    return sum_abs_err.item(), sum_sqr_err.item()
