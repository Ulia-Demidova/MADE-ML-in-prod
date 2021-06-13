import os

import click
import pandas as pd
from sklearn.model_selection import train_test_split


@click.command("preprocess")
@click.option("--input-dir")
@click.option("--output-dir")
def split(input_dir: str, output_dir):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"), index_col=False)
    target = pd.read_csv(os.path.join(input_dir, "target.csv"), index_col=False)
    X_train, X_val, y_train, y_val = train_test_split(data, target, test_size=0.2)

    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "train_data.csv"), index=False)
    X_val.to_csv(os.path.join(output_dir, "val_data.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "train_target.csv"), index=False)
    y_val.to_csv(os.path.join(output_dir, "val_target.csv"), index=False)


if __name__ == "__main__":
    split()
