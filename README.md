# Movie Recommender System Assignment
## Student
**Name**: Nagim Isyanbaev
<br/>
**Email**: n.isyanbaev@innopolis.university
<br/>
**Group numer**: B21-DS-02
# Installation
1. Clone repository
```console
git clone https://github.com/Nagim123/movie-recommender-system.git
```
```console
cd movie-recommender-system
```
2. Create virtual environment
```console
python -m venv .venv
```
3. Download dependencies
```console
pip install -r requirements.txt
```
If you are on linux also download
```console
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.0.1+cpu.html
```
# Training
To start training run the following command
```console
python train_model.py 1|2|3|4|5|a|b --epochs <EPOCH_NUM> --batch_size <SIZE_OF_BATCH>
```
Example:
```console
python train_model.py 1 --epochs 5 --batch_size 256
```
Data about losses will be saved to *benchmark/test_1.txt* and best weights will be saved to *models/best_model.pt*

# Prediction
To predict full user/movie rating matrix run the following command. You must have best weights for the appropriate subset of MovieLens100k (If you predict for **ua.test**, you should train model on **ua.base** before)
```console
python predict.py 1|2|3|4|5|a|b
```
Example:
```console
python predict.py 1
```
The full rating matrix will be saved into *data/interum/complete_prediction.pt*
# Evaluation
You must have full rating matrix *complete_predictio.pt* to run evaluation.
```console
python benchmark/evaluate.py 1|2|3|4|5|a|b
```
Example
```console
python benchmark/evaluate.py 1
```
Data about metrics will be saved into *benchmark/metric_1.json* file.