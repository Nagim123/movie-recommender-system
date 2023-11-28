from models.GNN_RecommenderModel import RecommendationModel
from constants import BEST_MODEL_PATH, FINAL_PREDICTION_PATH, INTERIM_PATH
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
        
        full_edges = torch.zeros((2, user_number*movie_number))
        for user in range(user_number):
            for movie in range(movie_number):
                full_edges[0][(user+1)*(movie+1)-1] = user
                full_edges[1][(user+1)*(movie+1)-1] = movie
        # Predict for each user rating for each movie
        embeds = model.encode_graph(data)
        full_predictions = model.predict_ratings(embeds["user"], embeds["movie"], full_edges.long())
        torch.save(full_predictions, FINAL_PREDICTION_PATH)