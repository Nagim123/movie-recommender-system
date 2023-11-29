import matplotlib.pyplot as plt
import json
import pathlib
import os
SCRIPT_PATH = pathlib.Path(__file__).parent.resolve()
FIGURE_FOLDER_PATH = os.path.join(SCRIPT_PATH, "../reports/figures")

def search_for_metric_files() -> None:
    """
    Search for all metric files in benchmark directory.
    """
    metric_files = []
    for (_, _, filenames) in os.walk(SCRIPT_PATH):
        for filename in filenames:
            if filename.split('.')[1] == "json":
                metric_files.append(os.path.join(SCRIPT_PATH, filename))
        break
    return metric_files

def visualize_metric_file_comparison(filepaths: list[str]) -> None:
    """
    Visualize the comparison of metrics.

    Parameters:
        filepaths (list[str]): Paths to metric files.
    """
    metrics = {"Precision": [], "Recall": [], "NDCG": [], "RMSE": []}
    parts = []
    for filepath in filepaths:
        with open(filepath, "r") as metric_file:
            json_data = json.load(metric_file)
            for key in metrics:
                metrics[key].append(json_data[key])
                parts.append(f"u{json_data['part']}.test, K={json_data['K']}")
    
    for i, metric in enumerate(metrics):
        plt.figure(i)
        plt.bar(parts, metrics[metric])
        plt.savefig(os.path.join(FIGURE_FOLDER_PATH, f"{metric}_plot.png"))
        

def visualize_training_loss(filepath: str):
    pass

if __name__ == "__main__":
    metric_files = search_for_metric_files()
    visualize_metric_file_comparison(metric_files)