from typing import Union, Dict

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from src.enities.train_params import TrainParams


THRESHOLD = 0.5
SklearnClassificationModel = Union[RandomForestClassifier, LogisticRegression]


def train_model(
    features: pd.DataFrame, target: pd.Series, train_params: TrainParams
) -> SklearnClassificationModel:
    if train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(n_estimators=100)
    elif train_params.model_type == "LogisticRegression":
        model = LogisticRegression()
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model


def create_inference_pipeline(
        model: SklearnClassificationModel, transformer: ColumnTransformer
) -> Pipeline:
    return Pipeline([("transformer", transformer), ("model", model)])


def predict(pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
    return pipeline.predict(features)


def predict_probabilities(pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
    return pipeline.predict_proba(features)[:, 1]


def evaluate_model(probabilities: np.ndarray, target: pd.Series) -> Dict[str, float]:
    predicts = (probabilities > THRESHOLD).astype(int)
    return {
        "roc_auc_score": roc_auc_score(target, probabilities),
        "f1_score": f1_score(target, predicts),
        "accuracy": accuracy_score(target, predicts),
    }
