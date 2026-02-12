import scipy.sparse as sp
import os
import numpy as np
import argparse
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="movie_to_music")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main():
    # Split users into train/val/test and save indices.
    args = parse_args()
    data_path = os.path.join("./dataset", args.dataset)

    target_rating_mat = sp.load_npz(os.path.join(data_path, "rating_matrix_target.npz"))
    target_rating_mat = sp.csr_matrix(target_rating_mat)

    n_user = target_rating_mat.shape[0]
    valid_ratio = 0.1
    user_list = list(range(n_user))
    np.random.seed(args.seed)
    np.random.shuffle(user_list)


    for test_ratio in [0.2, 0.5, 0.8]:
        n_test = int(n_user * test_ratio)
        n_val = int((n_user - n_test) * valid_ratio)
        n_train = n_user - n_test - n_val

        test_indices = user_list[:n_test]
        val_indices = user_list[n_test:n_test + n_val]
        train_indices = user_list[n_test + n_val:]
        
        with open(os.path.join(data_path, f"test_indices_{args.dataset}_{test_ratio}.pkl"), "wb") as f:
            pickle.dump(test_indices, f)
        with open(os.path.join(data_path, f"val_indices_{args.dataset}_{test_ratio}.pkl"), "wb") as f:
            pickle.dump(val_indices, f)
        with open(os.path.join(data_path, f"train_indices_{args.dataset}_{test_ratio}.pkl"), "wb") as f:
            pickle.dump(train_indices, f)


if __name__ == "__main__":
    main()
