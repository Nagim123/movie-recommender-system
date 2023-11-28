from torch_geometric.data import HeteroData
from constants import FINAL_PREDICTION_PATH
import torch
import os
import argparse

def get_user_links_ids(graph: HeteroData, user_id: int) -> torch.Tensor:
    """
    Get indecies of links of specified user.
    
    Parameters:
        graph (HeteroData): Birpartite graph with movies and users.
        user_id (int): User from whom to get links indecies.
    Returns
        Tensor: Link indecies.
    """
    indices = (user_id == graph["user", "rates", "movie"].edge_label_index[0]).nonzero()
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
    edge_index = graph["user", "rates", "movie"].edge_label_index
    edge_label = graph["user", "rates", "movie"].edge_label
    # Get indecies
    indices = get_user_links_ids(graph, user_id)
    # Concatenate ratings with movie IDs
    movie_ratings = torch.cat((edge_index[1][indices], edge_label[indices]), dim=1).long()
    return movie_ratings

@torch.no_grad()
def get_model_recommendations(predictions: torch.Tensor, graph: HeteroData, user_id: int, k: int = 5) -> torch.Tensor:
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
    
    user_links_ids = get_user_links_ids(graph, user_id)
    movie_ratings = extract_movies_with_ratings_by_user(graph, user_id)
    model_movie_rating =  torch.cat((movie_ratings[:, :1], predictions[user_links_ids]), dim=1)
    model_movie_rating = model_movie_rating[model_movie_rating[:, 1].sort(descending=True)[1]]
    
    return model_movie_rating[:k, 0].long()
if __name__ == "__main__":
    # Command line argument parsing
    parser = argparse.ArgumentParser(description="GNN prediction script. Specify part of dataset to make a prediction.")
    parser.add_argument("part", choices=['1', '2', '3', '4', '5', 'a', 'b'])
    args = parser.parse_args()
    

    if not os.path.exists(FINAL_PREDICTION_PATH):
        raise Exception("Final prediction was not created. Please run predict.py file!")

    final_prediction = torch.load(FINAL_PREDICTION_PATH)
