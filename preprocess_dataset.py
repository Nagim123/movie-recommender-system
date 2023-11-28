from constants import RAW_DATASET_PATH, INTERIM_PATH, MOVIE_HEADERS, USER_HEADERS, RATING_HEADERS
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit

import zipfile
import pandas as pd
import os
import torch
import shutil

def create_bipartite_graph_from_dataset(rating_set_id: str) -> tuple[HeteroData, HeteroData]:
    """
    Create train and test graphs from MovieLens dataset.

    Paramaters:
        rating_set_id (str): 1, 2, 3, 4, 5, a, b sets of MovieLens dataset.

    Returns:
        tuple[HeteroData, HeteroData]: Train graph and test graph respectively.
    """
    ml100k_path = os.path.join(INTERIM_PATH, "ml-100k")

    users_data_path = os.path.join(ml100k_path, "u.user")
    movies_data_path = os.path.join(ml100k_path, "u.item")
    rating_train_data_path = os.path.join(ml100k_path, f"u{rating_set_id}.base")
    rating_test_data_path = os.path.join(ml100k_path, f"u{rating_set_id}.test")

    # Process movie data:
    movie_df = pd.read_csv(movies_data_path, sep='|', header=None, names=MOVIE_HEADERS, index_col='movieId', encoding='ISO-8859-1')
    movie_mapping = {idx: i for i, idx in enumerate(movie_df.index)}
    genre_data = movie_df[MOVIE_HEADERS[6:]].values

    # Process user data:
    user_df = pd.read_csv(users_data_path, sep='|', header=None, names=USER_HEADERS, index_col='userId', encoding='ISO-8859-1')
    user_mapping = {idx: i for i, idx in enumerate(user_df.index)}

    # Normalize age
    age = user_df['age'].values / user_df['age'].values.max()
    age = torch.from_numpy(age).to(torch.float).view(-1, 1)

    # Encode gender
    gender = user_df['gender'].str.get_dummies().values
    gender = torch.from_numpy(gender).to(torch.float)

    # Encode occupation
    occupation = user_df['occupation'].str.get_dummies().values
    occupation = torch.from_numpy(occupation).to(torch.float)

    result = []
    for rating_path in [rating_train_data_path, rating_test_data_path]:
        data = HeteroData()
        data['movie'].x = torch.from_numpy(genre_data).to(torch.float)
        data['user'].x = torch.cat([age, gender, occupation], dim=-1)

        # Process rating data for training:
        df = pd.read_csv(rating_path, sep='\t', header=None, names=RATING_HEADERS,)

        src = [user_mapping[idx] for idx in df['userId']]
        dst = [movie_mapping[idx] for idx in df['movieId']]
        
        edge_index = torch.tensor([src, dst])
        data['user', 'rates', 'movie'].edge_index = edge_index

        rating = torch.from_numpy(df['rating'].values).to(torch.long)
        data['user', 'rates', 'movie'].rating = rating

        data['movie', 'rated_by', 'user'].edge_index = edge_index.flip([0])
        data['movie', 'rated_by', 'user'].rating = rating
        result.append(data)

    return result

def train_val_split(data: HeteroData) -> tuple[HeteroData, HeteroData]:
    """
    Split graph into train and validation sets.

    Paramaters:
        data (HeteroData): Heterogenous graph.

    Returns:
        tuple[HeteroData, HeteroData]: Train graph and validation graph respectively.
    """
    transform = RandomLinkSplit(
        num_val = 0.12, # Fraction of edges for validation set
        num_test = 0.0, # Fraction of edges for test set 
        # (since we already have separated test set, we set this value to 0)
        
        add_negative_train_samples=False, # We do not need negative edges
        neg_sampling_ratio=0.0,
        
        edge_types=("user", "rates", "movie"), # Set all edge types
        rev_edge_types=("movie", "rated_by", "user"), # Reverse edges for message passing
    )
    train_data, val_data, _ = transform(data)
    return (train_data, val_data)

if __name__ == "__main__":

    # Unzip dataset file
    with zipfile.ZipFile(RAW_DATASET_PATH) as zip_file:
        zip_file.extractall(INTERIM_PATH)
    
    # Transform each data split from dataset
    for part in ["1", "2", "3", "4", "5", "a", "b"]:
        data, test_data = create_bipartite_graph_from_dataset(part)
        train_data, val_data = train_val_split(data)

        torch.save(train_data.to_dict(), os.path.join(INTERIM_PATH, f"data{part}_train.pt"))
        torch.save(test_data.to_dict(), os.path.join(INTERIM_PATH, f"data{part}_test.pt"))
        torch.save(val_data.to_dict(), os.path.join(INTERIM_PATH, f"data{part}_val.pt"))
        print(f"u{part}.base and u{part}.test are prepocessed!")
    shutil.rmtree(os.path.join(INTERIM_PATH, "ml-100k"))
