import pickle
from typing import Union

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

SklearnClassificationModel = Union[RandomForestClassifier, LogisticRegression]


def save_model(model: SklearnClassificationModel, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output


def load_model(path: str) -> SklearnClassificationModel:
    with open(path, 'rb') as f:
        model = pickle.loads(f.read())
    return model
