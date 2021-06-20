import os
import pickle
import click

import pandas as pd
import numpy as np


@click.command("predict")
@click.option("--input-dir")
@click.option("--model-dir")
@click.option("--output-dir")
def predict(input_dir: str, model_dir: str, output_dir: str):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"), index_col=False)
    with open(os.path.join(model_dir, "model.pkl"), 'rb') as f:
        model = pickle.loads(f.read())

    preds = model.predict(data)
    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(os.path.join(output_dir, "predictions.csv"), preds, delimiter=",")


if __name__ == '__main__':
    predict()