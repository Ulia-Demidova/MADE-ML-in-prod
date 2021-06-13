import logging
import os
import pickle
from typing import List, Union, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, conlist
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


DEFAULT_HOST = '0.0.0.0'
DEFAULT_PORT = 8000


def load_object(path: str) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


class HeartDiseaseModel(BaseModel):
    data: List[conlist(Union[float, None], min_items=10, max_items=15)]
    features: List[str]


class HeartDiseaseResponse(BaseModel):
    disease: int


model: Optional[Pipeline] = None


def make_predict(
    data: List, features: List[str], model: Pipeline,
) -> List[HeartDiseaseResponse]:
    data = pd.DataFrame(data, columns=features)
    predicts = model.predict(data)

    return [HeartDiseaseResponse(disease=p) for p in predicts]


app = FastAPI()


@app.get("/")
def main():
    return "it is entry point of our predictor"


@app.on_event("startup")
def load_model():
    global model
    model_path = os.getenv("PATH_TO_MODEL")
    if model_path is None:
        err = f"PATH_TO_MODEL {model_path} is None"
        logger.error(err)
        raise RuntimeError(err)

    model = load_object(model_path)


@app.get("/predict/", response_model=List[HeartDiseaseResponse])
def predict(request: HeartDiseaseModel):
    return make_predict(request.data, request.features, model)


if __name__ == '__main__':
    uvicorn.run('app:app', host=DEFAULT_HOST, port=os.getenv('PORT', DEFAULT_PORT))