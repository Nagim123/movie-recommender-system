from models.GNN_RecommenderModel import RecommendationModel
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.data import HeteroData
from constants import INTERIM_PATH, BEST_MODEL_PATH
import torch
import torch.nn as nn
import tqdm
import os
import argparse

def create_dataloader(data: HeteroData, batch_size: int) -> LinkNeighborLoader:
    """
    Create dataloader from graph.

    Paramaters:
        data (HeteroData): graph.
        batch_size (int): Size of a batch.
    
    Returns:
        LinkNeighborLoader: data loader.
    """
    return LinkNeighborLoader(
        data,
        # Sample 128 neighbors for each node for 2 iterations
        num_neighbors=[128] * 2,
        # Use a batch size of 256 for sampling training nodes
        batch_size=batch_size,
        edge_label_index=(("user", "rates", "movie"), data["user", "rates", "movie"].edge_label_index),
        edge_label=data["user", "rates", "movie"].edge_label
    )

def train_one_epoch(model: RecommendationModel, train_loader: LinkNeighborLoader, optimizer: torch.optim.Optimizer, loss_fn, device) -> None:
    """
    Train model for one epoch.

    Parameters:
        model (RecommendationModel): Recommendation model.
        train_loader (LinkNeighborLoader): Train data loader.
        optimizer (Optimizer): Optimizer.
        loss_fn (_Loss): Loss function.
        device (str): Device
    """
    model.train()
    total_loss = 0
    for sampled_data in tqdm.tqdm(train_loader):
        optimizer.zero_grad()
        # Get predictions from model and true labels
        pred = model(sampled_data)
        ground_truth = sampled_data["user", "rates", "movie"].edge_label
        # Calculate loss and update gradients
        loss = loss_fn(pred, ground_truth.float())
        loss.backward()
        optimizer.step()
        # Keep track of total loss
        total_loss += loss.item()
    # Get average loss per epoch
    total_loss = total_loss / len(train_loader)
    return total_loss


def val_one_epoch(model: RecommendationModel, train_data: HeteroData, val_data: HeteroData, loss_fn, device) -> None:
    """
    Validate model for one epoch.

    Parameters:
        model (RecommendationModel): Recommendation model.
        train_data (HeteroData): Train graph.
        val_data (HeteroData): Validation graph.
        loss_fn (_Loss): Loss function.
        device (str): Device
    """
    model.eval()
    with torch.no_grad():
        # Get predictions from model and true labels
        embeds = model.encode_graph(train_data)
        edges = val_data["user", "rates", "movie"].edge_label_index
        pred = model.predict_ratings(embeds["user"], embeds["movie"], edges)
        ground_truth = val_data["user", "rates", "movie"].edge_label
        # Calculate loss and total loss
        loss = loss_fn(pred, ground_truth.float())
    return loss.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN training script. Specify part of dataset to start training.")
    parser.add_argument("part", choices=['1', '2', '3', '4', '5', 'a', 'b'])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()


    train_data = HeteroData(_mapping=torch.load(os.path.join(INTERIM_PATH, f"data{args.part}_train.pt")))
    val_data = HeteroData(_mapping=torch.load(os.path.join(INTERIM_PATH, f"data{args.part}_val.pt")))
    train_loader = create_dataloader(train_data, args.batch_size)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create model
    model = RecommendationModel(hidden_channels=128).to(device)
    # Use Adam optmizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    epochs = args.epochs
    best_loss = 1e9

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = val_one_epoch(model, train_data, val_data, loss_fn, device)
        if val_loss < best_loss: # Saving best model
            best_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"Epoch: {epoch}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")
