import pandas as pd
import numpy as np
from typing import List
from src.enities.feature_params import FeatureParams

import pytest


@pytest.fixture()
def sample_data():
    size = 20
    np.random.seed(123)
    df = pd.DataFrame({
        'age': np.random.randint(29, 77, size=size),
        'cp': np.random.randint(4, size=size),
        'sex': np.random.randint(2, size=size),
        'trestbps': np.random.randint(94, 200, size=size),
        'chol': np.random.randint(126, 564, size=size),
        'fbs': np.random.randint(2, size=size),
        'restecg': np.random.randint(2, size=size),
        'thalach': np.random.randint(71, 202, size=size),
        'exang': np.random.randint(2, size=size),
        'oldpeak': 4 * np.random.rand(),
        'slope': np.random.randint(3, size=size),
        'ca': np.random.randint(5, size=size),
        'thal': np.random.randint(4, size=size),
        'target': np.random.randint(2, size=size),
    })
    return df


@pytest.fixture()
def categorical_features() -> List[str]:
    return ["cp", "restecg", "slope", "ca", "thal"]


@pytest.fixture()
def binary_features() -> List[str]:
    return ["sex", "fbs", "exang"]


@pytest.fixture()
def numeric_features() -> List[str]:
    return ["age", "trestbps", "chol", "thalach", "oldpeak"]


@pytest.fixture()
def target_col() -> str:
    return "target"


@pytest.fixture()
def feature_params(categorical_features,
                   binary_features,
                   numeric_features,
                   target_col) -> FeatureParams:
    return FeatureParams(
        categorical_features,
        numeric_features,
        binary_features,
        target_col
    )
