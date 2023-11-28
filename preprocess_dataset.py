from constants import RAW_DATASET_PATH, INTERIM_PATH, MOVIE_HEADERS, USER_HEADERS, RATING_HEADERS
from torch_geometric.data import HeteroData

import zipfile
import pandas as pd
import os
import torch
import shutil

def create_bipartite_graph_from_dataset(rating_set_id: str):
    ml100k_path = os.path.join(INTERIM_PATH, "ml-100k")

    users_data_path = os.path.join(ml100k_path, "u.user")
    movies_data_path = os.path.join(ml100k_path, "u.item")
    rating_train_data_path = os.path.join(ml100k_path, f"u{rating_set_id}.base")
    rating_test_data_path = os.path.join(ml100k_path, f"u{rating_set_id}.base")

    data = HeteroData()

    # Process movie data:
    df = pd.read_csv(movies_data_path, sep='|', header=None, names=MOVIE_HEADERS, index_col='movieId', encoding='ISO-8859-1')
    movie_mapping = {idx: i for i, idx in enumerate(df.index)}

    x = df[MOVIE_HEADERS[6:]].values
    data['movie'].x = torch.from_numpy(x).to(torch.float)

    # Process user data:
    df = pd.read_csv(users_data_path, sep='|', header=None, names=USER_HEADERS, index_col='userId', encoding='ISO-8859-1')
    user_mapping = {idx: i for i, idx in enumerate(df.index)}

    age = df['age'].values / df['age'].values.max()
    age = torch.from_numpy(age).to(torch.float).view(-1, 1)

    gender = df['gender'].str.get_dummies().values
    gender = torch.from_numpy(gender).to(torch.float)

    occupation = df['occupation'].str.get_dummies().values
    occupation = torch.from_numpy(occupation).to(torch.float)

    data['user'].x = torch.cat([age, gender, occupation], dim=-1)

    # Process rating data for training:
    df = pd.read_csv(rating_train_data_path, sep='\t', header=None, names=RATING_HEADERS,)

    src = [user_mapping[idx] for idx in df['userId']]
    dst = [movie_mapping[idx] for idx in df['movieId']]
    edge_index = torch.tensor([src, dst])
    data['user', 'rates', 'movie'].edge_index = edge_index

    rating = torch.from_numpy(df['rating'].values).to(torch.long)
    data['user', 'rates', 'movie'].rating = rating

    time = torch.from_numpy(df['timestamp'].values)
    data['user', 'rates', 'movie'].time = time

    data['movie', 'rated_by', 'user'].edge_index = edge_index.flip([0])
    data['movie', 'rated_by', 'user'].rating = rating
    data['movie', 'rated_by', 'user'].time = time

    # Process rating data for testing:
    df = pd.read_csv(rating_test_data_path, sep='\t', header=None, names=RATING_HEADERS)

    src = [user_mapping[idx] for idx in df['userId']]
    dst = [movie_mapping[idx] for idx in df['movieId']]
    edge_label_index = torch.tensor([src, dst])
    data['user', 'rates', 'movie'].edge_label_index = edge_label_index

    edge_label = torch.from_numpy(df['rating'].values).to(torch.float)
    data['user', 'rates', 'movie'].edge_label = edge_label

    torch.save(data.to_dict(), os.path.join(INTERIM_PATH, f"data{rating_set_id}.pt"))


if __name__ == "__main__":

    # Unzip dataset file
    with zipfile.ZipFile(RAW_DATASET_PATH) as zip_file:
        zip_file.extractall(INTERIM_PATH)
    
    create_bipartite_graph_from_dataset("1")
    shutil.rmtree(os.path.join(INTERIM_PATH, "ml-100k"))
