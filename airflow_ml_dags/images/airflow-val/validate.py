import os
import click
import pickle
from typing import Dict
import json

import pandas as pd
from sklearn.metrics import accuracy_score


@click.command("validate")
@click.option("--data-dir")
@click.option("--model-dir")
def validate(data_dir: str, model_dir: str):
    data = pd.read_csv(os.path.join(data_dir, "val_data.csv"), index_col=False)
    target = pd.read_csv(os.path.join(data_dir, "val_target.csv"), index_col=False)

    with open(os.path.join(model_dir, "model.pkl"), 'rb') as f:
        model = pickle.loads(f.read())

    predicts = model.predict(data)
    metrics = {"accuracy": accuracy_score(target, predicts)}
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    validate()

