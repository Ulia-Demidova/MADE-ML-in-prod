import click
import numpy as np
import pandas as pd
from src.enities import read_predict_params, PredictParams
from src.data import read_data
from src.features import make_features
from src.models import load_model, predict


def save_data(path: str, data: np.array):
    pd.DataFrame({"target": data}).to_csv(path, index=False)


def predict_pipeline(params: PredictParams):
    data = read_data(params.input_data_path)
    transformer = load_model(params.transformer_path)
    features = make_features(transformer, data)
    model = load_model(params.model_path)
    predicts = predict(model, features)
    save_data(params.preds_path, predicts)


@click.command(name="predict")
@click.argument("config_path")
def predict_command(config_path: str):
    params = read_predict_params(config_path)
    predict_pipeline(params)


if __name__ == "__main__":
    predict_command()
