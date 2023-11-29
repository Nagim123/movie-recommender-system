import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, HeteroConv, JumpingKnowledge
from torch_geometric.typing import Tensor
from torch_geometric.data import HeteroData

class GNN(nn.Module):
    """
    Graph neural network based on SAGEConv's to produce node embeddings.
    """
    def __init__(self, hidden_channels: int):
        super().__init__()
        # We will use 2 layer architecture
        self.conv1 = HeteroConv({ # Set up different layers for different edge types
            ("user", "rates", "movie"): SAGEConv((24, 18), hidden_channels),
            ("movie", "rated_by", "user"): SAGEConv((18, 24), hidden_channels),
        }, aggr="sum")

        self.conv2 = HeteroConv({
            ("user", "rates", "movie"): SAGEConv(hidden_channels, hidden_channels),
            ("movie", "rated_by", "user"): SAGEConv(hidden_channels, hidden_channels),
        }, aggr="sum")

    def forward(self, x_dict: Tensor, edge_index: Tensor):
        x_dict1 = self.conv1(x_dict, edge_index)
        x_dict2 = self.conv2(x_dict1, edge_index)

        return x_dict2

class RecommendationModel(nn.Module):
    """
    Recommendation model to predict ratings between user and movie.
    """
    def __init__(self, hidden_channels):
        super().__init__()
        # Instantiate GNN model
        self.gnn = GNN(hidden_channels)
        # Use cosine simularity to predict rating
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def encode_graph(self, data: HeteroData) -> dict:
        """
        Produce embeddings for each node type.
        
        Returns:
            dict: {"user": Tensor, "movie": Tensor}
        """
        x_dict = self.gnn(data.x_dict, data.edge_index_dict)
        return x_dict

    def predict_ratings(self, user_embeds: Tensor, movie_embeds: Tensor, edge_label_index: Tensor) -> Tensor:
        """
        Predict ratings for each input edge.
        
        Parameters:
            user_embeds (Tensor): User embeddings
            movie_embeds (Tensor): Movie embeddings
            edge_label_index (Tensor): Edges
        
        Returns:
            Tensor: For each edge in edge_label_index returns ratings.
        """
        edge_feat_user = user_embeds[edge_label_index[0]]
        edge_feat_movie = movie_embeds[edge_label_index[1]]

        return (self.classifier(torch.cat([edge_feat_user, edge_feat_movie], dim=1))*5).squeeze(1)

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = self.encode_graph(data)
        x = self.predict_ratings(x_dict["user"], 
                                 x_dict["movie"], 
                                 data["user", "rates", "movie"].edge_label_index)
        return x