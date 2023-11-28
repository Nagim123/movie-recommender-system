from models.GNN_RecommenderModel import RecommendationModel
from constants import BEST_MODEL_PATH, INTERIM_PATH
from torch_geometric.data import HeteroData
import torch
import os
import argparse

if __name__ == "__main__":
    # Command line argument parsing
    parser = argparse.ArgumentParser(description="GNN prediction script. Specify part of dataset to make a prediction.")
    parser.add_argument("part", choices=['1', '2', '3', '4', '5', 'a', 'b'])
    args = parser.parse_args()

    if not os.path.exists(BEST_MODEL_PATH):
        raise Exception("Best model was not found! (train or download one)")

    # Loading model
    model = RecommendationModel(hidden_channels=128)
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()

    # Predict ratings
    with torch.no_grad():
        # Load train data
        data = HeteroData(_mapping=torch.load(os.path.join(INTERIM_PATH, f"data{args.part}_train.pt")))
        user_number = data["user"].x.shape[0]
        movie_number = data["movie"].x.shape[0]
        
        embeddings = model.encode_graph(data)
        # Predict for each user rating for each movie
        full_predictions = torch.matmul(embeddings["user"], embeddings["movie"].T)
        torch.save(full_predictions, os.path.join(INTERIM_PATH, "complete_prediction.pt"))