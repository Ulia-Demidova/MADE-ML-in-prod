import pickle
from typing import Union
from sklearn.compose import ColumnTransformer
from src.models import SklearnClassificationModel


Artifact = Union[SklearnClassificationModel, ColumnTransformer]


def save_model(model: Artifact, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output


def load_model(path: str) -> Artifact:
    with open(path, 'rb') as f:
        model = pickle.loads(f.read())
    return model
