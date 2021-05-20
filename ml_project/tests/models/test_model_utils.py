from src.models import save_model, load_model, train_model
import pytest
import pandas as pd
from typing import Tuple
from src.enities import TrainParams
from src.features import build_transformer, make_features, fit_transformer, extract_target
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted
import os


@pytest.fixture
def features_and_target(sample_data: pd.DataFrame, feature_params) -> Tuple[pd.DataFrame, pd.Series]:
    transformer = build_transformer(feature_params)
    transformer, data = fit_transformer(transformer, sample_data)
    features = make_features(transformer, data)
    target = extract_target(data, feature_params)
    return features, target


def test_fit_model(features_and_target: Tuple[pd.DataFrame, pd.Series]):
    features, target = features_and_target
    model = train_model(features, target, TrainParams())
    assert isinstance(model, LogisticRegression)
    check_is_fitted(model, attributes=["coef_", "intercept_"])


def test_save_load_model(tmpdir):
    model = RandomForestClassifier()
    expected_output = tmpdir.join("model.pkl")
    output = save_model(model, expected_output)
    assert output == expected_output
    assert os.path.exists(expected_output)
    model = load_model(output)
    assert isinstance(model, RandomForestClassifier)

