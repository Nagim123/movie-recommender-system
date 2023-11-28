from constants import FINAL_PREDICTION_PATH
import torch
import os
import argparse

#def get_recommendation_for_user()


if __name__ == "__main__":
    # Command line argument parsing
    parser = argparse.ArgumentParser(description="GNN prediction script. Specify part of dataset to make a prediction.")
    parser.add_argument("part", choices=['1', '2', '3', '4', '5', 'a', 'b'])
    args = parser.parse_args()

    if not os.path.exists(FINAL_PREDICTION_PATH):
        raise Exception("Final prediction was not created. Please run predict.py file!")

    final_prediction = torch.load(FINAL_PREDICTION_PATH)
