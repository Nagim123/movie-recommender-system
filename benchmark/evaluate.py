from torch_geometric.data import HeteroData
import torch
import os
import argparse
import pathlib
import os
import tqdm as tqdm
import numpy as np
import json

SCRIPT_PATH = pathlib.Path(__file__).parent.resolve()
INTERIM_PATH = os.path.join(SCRIPT_PATH, "../data/interim")
MODELS_PATH = os.path.join(SCRIPT_PATH, "../models")
FINAL_PREDICTION_PATH = os.path.join(INTERIM_PATH, "complete_prediction.pt")

def compute_RMSE(predicted_ratings: torch.Tensor, true_rating: torch.Tensor) -> float:
    """
    Compute RMSE metric.

    Parameters:
        predicted_ratings (Tensor): Ratings predicted by model.
        true_ratings (Tensor): Ground truth ratings.

    Returns:
        float: RMSE value.
    """
    return float(np.sqrt(np.mean(np.square((predicted_ratings - true_rating).numpy()))))

@torch.no_grad()
def evaluate_precision_and_recall(recommendation: torch.Tensor, test_graph: HeteroData, user_id: int, k: int = 5) -> tuple[float, float]:
    """
    Evaluate precision and recall value for specified user.
    
    Parameters:
        recommendation (Tensor): Recommendation generated by model.
        test_graph (HeteroData): Birpartite graph with movies and users.
        user_id (int): User for whom to recommend movies.
        k (int): The number of recommendations for evaluation.
    Returns
        tuple[float, float]: Precision@k, Recall@k
    """
    TruePositive, FalsePositive, FalseNegative = 0, 0, 0
    
    if len(recommendation) == 0:
        return (1, 1)
    
    movie_ratings = extract_movies_with_ratings_by_user(test_graph, user_id)
    for movie in movie_ratings:
        if movie[1] >= 4:
            if movie[0] in recommendation:
                TruePositive += 1
            else:
                FalseNegative += 1
        else:
            if movie[0] in recommendation:
                FalsePositive += 1
    return (TruePositive / (TruePositive + FalsePositive),
            TruePositive / (TruePositive + FalseNegative)) if TruePositive > 0 else (0, 0)

def compute_DCG(recommendation: torch.Tensor, movie_ratings: torch.Tensor, user_id: int) -> float:
    """
    Evaluate precision and recall value for specified user.
    
    Parameters:
        recommendation (Tensor): Recommendation generated by model.
        movie_ratings (Tensor): Real movie ratings.
        user_id (int): User for whom to recommend movies.
        k (int): The number of recommendations for evaluation.
    Returns
        tuple[float, float]: Precision@k, Recall@k
    """
    DCG = 0
    for i, movie in enumerate(recommendation):
        if not movie in movie_ratings[:, 0]:
            score = 0
        else:
            indx = (movie_ratings[:, 0] == movie).nonzero().item()
            score = (2**(movie_ratings[indx][1]) - 1).item()
        discount = np.log2(i+2)
        DCG += score / discount

    return DCG
@torch.no_grad()
def evaluate_NDCG(recommendation: torch.Tensor, test_graph: HeteroData, user_id: int):
    """
    Evaluate NDCG value for specified user.
    
    Parameters:
        recommendation (Tensor): Recommendation generated by model.
        test_graph (HeteroData): Birpartite graph with movies and users.
        user_id (int): User for whom to recommend movies.
        k (int): The number of recommendations for evaluation.
    Returns
        tuple[float, float]: NDCG@k
    """
    movie_ratings = extract_movies_with_ratings_by_user(test_graph, user_id)
    DCG = compute_DCG(recommendation, movie_ratings, user_id)
    best_recommendation = movie_ratings[movie_ratings[:, 1].sort(descending=True)[1]][:, 0]
    IDCG = compute_DCG(best_recommendation, movie_ratings, user_id)

    return DCG/IDCG

def get_user_links_ids(graph: HeteroData, user_id: int) -> torch.Tensor:
    """
    Get indecies of links of specified user.
    
    Parameters:
        graph (HeteroData): Birpartite graph with movies and users.
        user_id (int): User from whom to get links indecies.
    Returns
        Tensor: Link indecies.
    """
    indices = (user_id == graph["user", "rates", "movie"].edge_index[0]).nonzero()
    return indices

def extract_movies_with_ratings_by_user(graph: HeteroData, user_id: int) -> torch.Tensor:
    """
    Extract movies with ratings from specified user.
    
    Parameters:
        graph (HeteroData): Birpartite graph with movies and users.
        user_id (int): User from whom to collect movies and ratings.
    Returns
        Tensor: Movies with ratings
    """
    # Extract edges
    edge_index = graph["user", "rates", "movie"].edge_index
    edge_label = graph["user", "rates", "movie"].edge_label
    # Get indecies
    indices = get_user_links_ids(graph, user_id)
    # Concatenate ratings with movie IDs
    movie_ratings = torch.cat((edge_index[1][indices], edge_label[indices]), dim=1).long()
    return movie_ratings

@torch.no_grad()
def get_model_recommendations(predictions: torch.Tensor, edge_list: torch.Tensor, user_id: int, k: int = 5) -> torch.Tensor:
    """
    Get model's movie recommendation
    
    Parameters:
        predictions (Tensor): Predicted rating for each movie.
        test_graph (HeteroData): Birpartite graph with movies and users.
        user_id (int): User for whom to recommend movies.
        k (int): The number of recommendations to return.
    Returns
        (Tensor) Recommended movies.
    """
    user_links = (user_id == edge_list[0]).nonzero()
    model_movie_rating =  torch.cat((edge_list[1][user_links], predictions[user_links]), dim=1)
    model_movie_rating = model_movie_rating[model_movie_rating[:, 1].sort(descending=True)[1]]
    
    return model_movie_rating[:k, 0].long()

if __name__ == "__main__":
    # Command line argument parsing
    parser = argparse.ArgumentParser(description="GNN prediction script. Specify part of dataset to make a prediction.")
    parser.add_argument("part", choices=['1', '2', '3', '4', '5', 'a', 'b'])
    args = parser.parse_args()


    if not os.path.exists(FINAL_PREDICTION_PATH):
        raise Exception("Final prediction was not created. Please run predict.py file!")

    test_data = HeteroData(_mapping=torch.load(os.path.join(INTERIM_PATH, f"data{args.part}_test.pt")))
    train_data = HeteroData(_mapping=torch.load(os.path.join(INTERIM_PATH, f"data{args.part}_train.pt")))
    
    user_number = test_data["user"].x.shape[0]
    movie_number = test_data["movie"].x.shape[0]

    
    test_edges = test_data["user", "rates", "movie"].edge_index
    train_edges = train_data["user", "rates", "movie"].edge_index
    full_edges = torch.zeros((2, user_number*movie_number))
    final_prediction = torch.load(FINAL_PREDICTION_PATH)
    # Prediction for only test edges
    masked_prediction = torch.zeros(test_edges.shape[1])
    # Prediction for a whole graph without training data
    real_prediction = final_prediction.detach().clone()

    for user in range(user_number):
        for movie in range(movie_number):
            full_edges[0][(user+1)*(movie+1)-1] = user
            full_edges[1][(user+1)*(movie+1)-1] = movie
    
    for i in range(test_edges.shape[1]):
        user, movie = test_edges[0][i], test_edges[1][i]
        index = (user+1)*(movie+1)-1
        masked_prediction[i] = final_prediction[index]
    
    for i in range(train_edges.shape[1]):
        user, movie = train_edges[0][i], train_edges[1][i]
        index = (user+1)*(movie+1)-1
        real_prediction[index] = 0


    K = 20
    progress = tqdm.tqdm(range(user_number))
    test_users = set(test_edges[0].tolist())

    precisions, recalls, NDCGs = [], [], []
    for user in progress:
        if not user in test_users:
            # Only evaluate for users that are in test set
            continue
        recommendation = get_model_recommendations(masked_prediction, test_edges, user, k=K)
        precision, recall = evaluate_precision_and_recall(recommendation, test_data, user, k=K)
        NDCG = evaluate_NDCG(recommendation, test_data, user)
        precisions.append(precision)
        recalls.append(recall)
        NDCGs.append(NDCG)
    # Display metrics
    print(f"Precision@{K} = {np.mean(precisions)}")
    print(f"Recall@{K} = {np.mean(recalls)}")
    print(f"NDCG@{K} = {np.mean(NDCGs)}")
    print(f"RMSE = {compute_RMSE(masked_prediction, test_data['user', 'rates', 'movie'].edge_label)}")

    # Save calculated metrics to metric file
    with open(os.path.join(SCRIPT_PATH, f"metric_file_{args.part}.json"), "w") as metric_file:
        metrics = {
            "part": args.part,
            "K": K,
            "Precision": np.mean(precisions),
            "Recall": np.mean(recalls),
            "NDCG": np.mean(NDCGs),
            "RMSE": compute_RMSE(masked_prediction, test_data["user", "rates", "movie"].edge_label)
        }
        metric_file.write(json.dumps(metrics))

