from typing import Union

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.enities.train_params import TrainParams


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
