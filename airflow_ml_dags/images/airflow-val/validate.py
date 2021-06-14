import os
import click
import pickle
from typing import Dict
import json

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


THRESHOLD = 0.5


def evaluate_model(probabilities: np.ndarray, target: pd.Series) -> Dict[str, float]:
    predicts = (probabilities > THRESHOLD).astype(int)
    return {
        "roc_auc_score": roc_auc_score(target, probabilities),
        "f1_score": f1_score(target, predicts),
        "accuracy": accuracy_score(target, predicts),
    }


@click.command("validate")
@click.option("--data-dir")
@click.option("--model-dir")
def validate(data_dir: str, model_dir: str):
    data = pd.read_csv(os.path.join(data_dir, "val_data.csv"), index_col=False)
    target = pd.read_csv(os.path.join(data_dir, "val_target.csv"), index_col=False)

    with open(os.path.join(model_dir, "model.pkl"), 'rb') as f:
        model = pickle.loads(f.read())

    probabilities = model.predict_proba(data)[:, 1]
    metrics = evaluate_model(probabilities, target)
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    validate()

