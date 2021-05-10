import pickle
from src.models import SklearnClassificationModel


def save_model(model: SklearnClassificationModel, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output


def load_model(path: str) -> SklearnClassificationModel:
    with open(path, 'rb') as f:
        model = pickle.loads(f.read())
    return model
