"""
All file paths and other constants.
"""

import pathlib
import os

SCRIPT_PATH = pathlib.Path(__file__).parent.resolve()
RAW_DATASET_PATH = os.path.join(SCRIPT_PATH, "data/raw/ml-100k.zip")
INTERIM_PATH = os.path.join(SCRIPT_PATH, "data/interim")
MODELS_PATH = os.path.join(SCRIPT_PATH, "models")
BEST_MODEL_PATH = os.path.join(MODELS_PATH, "best_model.pt")
FINAL_PREDICTION_PATH = os.path.join(INTERIM_PATH, "complete_prediction.pt")
BENCHMARK_DIR_PATH = os.path.join(SCRIPT_PATH, "benchmark")

MOVIE_HEADERS = [
    "movieId", "title", "releaseDate", "videoReleaseDate", "IMDb URL",
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]
USER_HEADERS = ["userId", "age", "gender", "occupation", "zipCode"]
RATING_HEADERS = ["userId", "movieId", "rating", "timestamp"]