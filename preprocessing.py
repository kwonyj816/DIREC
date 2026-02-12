import os
import json
import gzip
import numpy as np
from scipy.sparse import csr_matrix, save_npz
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import torch
from fastprogress import progress_bar

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_and_filter_reviews(file_path):
    filtered_reviews, reviewer_ids = [], set()
    open_fn = gzip.open if file_path.endswith('.gz') else open

    with open_fn(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            review = json.loads(line)
            filtered = {
                "reviewerID": review["reviewerID"],
                "asin": review["asin"],
                "reviewText": review["reviewText"],
                "overall": review["overall"]
            }
            filtered_reviews.append(filtered)
            reviewer_ids.add(review["reviewerID"])

    return filtered_reviews, reviewer_ids


def load_and_filter_reviews_20_core(file_path, min_reviews=20):

    all_reviews = []
    open_fn = gzip.open if file_path.endswith('.gz') else open

    with open_fn(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            review = json.loads(line)
            filtered = {
                "reviewerID": review["reviewerID"],
                "asin": review["asin"],
                "reviewText": review["reviewText"],
                "overall": review["overall"]
            }
            all_reviews.append(filtered)
    

    reviewer_counts = defaultdict(int)
    for review in all_reviews:
        reviewer_counts[review["reviewerID"]] += 1

    valid_reviewer_ids = {rid for rid, count in reviewer_counts.items() if count >= min_reviews}
    
    filtered_reviews = [r for r in all_reviews if r["reviewerID"] in valid_reviewer_ids]
    reviewer_ids = valid_reviewer_ids
    
    return filtered_reviews, reviewer_ids


def map_users_and_items(source_reviews, target_reviews, target_non_common, common_ids):
    id_mapping = {rid: idx for idx, rid in enumerate(sorted(common_ids))}
    for r in source_reviews:
        r['mappedUserID'] = id_mapping[r['reviewerID']]
    for r in target_reviews:
        r['mappedUserID'] = id_mapping[r['reviewerID']]

    source_asins = sorted({r['asin'] for r in source_reviews})
    target_asins = sorted({r['asin'] for r in target_reviews})
    source_item_mapping = {asin: idx for idx, asin in enumerate(source_asins)}
    target_item_mapping = {asin: idx for idx, asin in enumerate(target_asins)}

    for r in source_reviews:
        r['mappedItemID'] = source_item_mapping[r['asin']]
    for r in target_reviews:
        r['mappedItemID'] = target_item_mapping[r['asin']]
    for r in target_non_common:
        if r['asin'] in target_item_mapping:
            r['mappedItemID'] = target_item_mapping[r['asin']]
        else:
            r['mappedItemID'] = -1
    return source_reviews, target_reviews, target_non_common, source_item_mapping, target_item_mapping


def build_rating_matrix(reviews):
    user_ids, item_ids, ratings = [], [], []
    for entry in reviews:
        user_ids.append(entry['mappedUserID'])
        item_ids.append(entry['mappedItemID'])
        ratings.append(entry['overall'])

    num_users = max(user_ids) + 1
    num_items = max(item_ids) + 1

    return csr_matrix((ratings, (user_ids, item_ids)), shape=(num_users, num_items))


def build_item_text_dict(reviews):
    item_reviews = defaultdict(list)
    for entry in reviews:
        item_reviews[entry['mappedItemID']].append(entry['reviewText'])
    return {item: texts for item, texts in sorted(item_reviews.items())}


def mean_pool_embeddings(model, review_dict):
    res_embeddings = []
    pbar = progress_bar(sorted(review_dict.keys()))
    for item_id in pbar:
        pbar.comment = "Item embeddings"
        reviews = review_dict[item_id]
        if reviews:
            embeddings = model.encode(reviews, convert_to_numpy=True, show_progress_bar=False)
            avg_embedding = np.mean(embeddings, axis=0)
        else:
            avg_embedding = np.zeros(model.get_sentence_embedding_dimension())
        res_embeddings.append(avg_embedding)
    return np.vstack(res_embeddings)


def build_review_embedding_dict(model, reviews):
    review_dict = {(r['mappedUserID'], r['mappedItemID']): r['reviewText'] for r in reviews}
    embedding_dict = {}
    pbar = progress_bar(review_dict.items())
    for key, text in pbar:
        pbar.comment = "Embedding reviews"
        embedding_dict[key] = model.encode(text)
    return embedding_dict


def embed_target_non_common_reviews_ratings(model, target_non_common, common_ids, target_common_asins):
    target_non_common_reviews = [
        r for r in target_non_common 
        if r["reviewerID"] not in common_ids 
        and r["asin"] in target_common_asins
        and r.get("mappedItemID", -1) >= 0
    ]
    
    if not target_non_common_reviews:
        emb_size = model.get_sentence_embedding_dimension()
        return np.empty((0, emb_size)), np.empty((0,)), np.empty((0,), dtype=np.int64)
    
    review_texts = [r["reviewText"] for r in target_non_common_reviews]
    ratings = np.array([r["overall"] for r in target_non_common_reviews])
    item_ids = np.array([r["mappedItemID"] for r in target_non_common_reviews])
    
    embeddings = model.encode(
        review_texts, 
        convert_to_numpy=True, 
        show_progress_bar=True,
        batch_size=256
    )
    


    return embeddings, ratings, item_ids


def main():
    scenario = "book_to_movie"  # choose: "movie_to_music", "book_to_movie", "book_to_music"

    scenario_map = {
        "movie_to_music": ("reviews_Movies_and_TV_5.json.gz", "reviews_CDs_and_Vinyl_5.json.gz"),
        "book_to_movie": ("reviews_Books_5.json.gz", "reviews_Movies_and_TV_5.json.gz"),
        "book_to_music": ("reviews_Books_5.json.gz", "reviews_CDs_and_Vinyl_5.json.gz"),
    }
    if scenario not in scenario_map:
        raise ValueError(f"Unknown scenario: {scenario}")

    directory = f"./dataset/{scenario}/"
    os.makedirs(directory, exist_ok=True)

    source_file, target_file = scenario_map[scenario]
    source_reviews, source_ids = load_and_filter_reviews(os.path.join("./dataset/raw", source_file))
    target_reviews, target_ids = load_and_filter_reviews(os.path.join("./dataset/raw", target_file))


    common_ids = source_ids & target_ids
    source_common = [r for r in source_reviews if r["reviewerID"] in common_ids]
    target_common = [r for r in target_reviews if r["reviewerID"] in common_ids]
    target_non_common = [r for r in target_reviews if r["reviewerID"] not in common_ids]


    source_total_users = len(source_ids)
    target_total_users = len(target_ids)
    


    source_common, target_common, target_non_common, _, _ = map_users_and_items(source_common, target_common, target_non_common, common_ids)

    target_rating_matrix_sparse = build_rating_matrix(target_common)
    save_npz(os.path.join(directory, "rating_matrix_target.npz"), target_rating_matrix_sparse)

    
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)

    item_source_text = build_item_text_dict(source_common)
    item_source_embeddings = mean_pool_embeddings(model, item_source_text)

    source_embedding_dict = build_review_embedding_dict(model, source_common)
    target_embedding_dict = build_review_embedding_dict(model, target_common)

    target_common_asins = set(r['asin'] for r in target_common)
    target_non_common_embeddings, target_non_common_ratings, target_non_common_item_ids = embed_target_non_common_reviews_ratings(model, target_non_common, common_ids, target_common_asins)
    

    np.save(os.path.join(directory, "source_embedding_dict.npy"), source_embedding_dict)
    np.save(os.path.join(directory, "target_embedding_dict.npy"), target_embedding_dict)
    np.save(os.path.join(directory, "embedding_source_item_avg.npy"), item_source_embeddings)
    np.save(os.path.join(directory, "target_non_overlapping_user_review_embedding.npy"), target_non_common_embeddings)
    np.save(os.path.join(directory, "target_non_overlapping_user_ratings.npy"), target_non_common_ratings)
    np.save(os.path.join(directory, "target_non_overlapping_user_review_item_ids.npy"), target_non_common_item_ids)

    print(f"Source domain - total users: {source_total_users}")
    print(f"Target domain - total users: {target_total_users}")
    print(f"Overlapping users: {len(common_ids)}")
    print(f"Common reviewerIDs: {len(common_ids)}")
    print(f"Source domain - overlapping user reviews: {len(source_common)}")
    print(f"Target domain - overlapping user reviews: {len(target_common)}")
    print(f"Target domain - non-overlapping user reviews: {len(target_non_common)}")


if __name__ == "__main__":
    main()
