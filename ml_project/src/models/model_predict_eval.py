from typing import Dict, Union

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


THRESHOLD = 0.5
SklearnClassificationModel = Union[RandomForestClassifier, LogisticRegression]


def predict_probabilities(model: SklearnClassificationModel, features: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(features)[:, 1]


def predict(model: SklearnClassificationModel, features: pd.DataFrame) -> np.ndarray:
    return model.predict(features)


def evaluate_model(probabilities: np.ndarray, target: pd.Series) -> Dict[str, float]:
    predicts = (probabilities > THRESHOLD).astype(int)
    return {
        "roc_auc_score": roc_auc_score(target, probabilities),
        "f1_score": f1_score(target, predicts),
        "accuracy": accuracy_score(target, predicts),
    }
