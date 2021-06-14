import os
import click
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression


@click.command("train")
@click.option("--data-dir")
@click.option("--model-dir")
def train(data_dir: str, model_dir: str):
    data = pd.read_csv(os.path.join(data_dir, "train_data.csv"), index_col=False)
    target = pd.read_csv(os.path.join(data_dir, "train_target.csv"), index_col=False)

    model = LogisticRegression()
    model.fit(data, target)

    with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    train()

