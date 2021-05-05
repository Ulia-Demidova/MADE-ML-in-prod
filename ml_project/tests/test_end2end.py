from train import train_pipeline
from src.enities import TrainPipelineParams, TrainParams
from src.enities.splitting_params import SplittingParams
from src.enities.feature_params import FeatureParams
import os
import pandas as pd

from py._path.local import LocalPath


def test_e2e(tmpdir: LocalPath,
             sample_data: pd.DataFrame,
             feature_params: FeatureParams):
    data_path = tmpdir.join('sample.csv')
    sample_data.to_csv(data_path)

    expected_model_path = tmpdir.join("model.pkl")
    expected_metric_path = tmpdir.join("metrics.json")
    path_to_model, metrics = train_pipeline(TrainPipelineParams(input_data_path=data_path,
                                                                output_model_path=expected_model_path,
                                                                metric_path=expected_metric_path,
                                                                splitting_params=SplittingParams(0.2, 123),
                                                                train_params=TrainParams("LogisticRegression"),
                                                                feature_params=feature_params))
    assert "roc_auc_score" in metrics
    assert "f1_score" in metrics
    assert "accuracy" in metrics
    assert metrics["accuracy"] > 0
    assert path_to_model == expected_model_path
    assert os.path.exists(path_to_model)
