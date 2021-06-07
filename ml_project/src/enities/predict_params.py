from dataclasses import dataclass
from .feature_params import FeatureParams
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class PredictParams:
    input_data_path: str
    model_path: str
    transformer_path: str
    preds_path: str
    feature_params: FeatureParams


PredictParamsSchema = class_schema(PredictParams)


def read_predict_params(config_path: str) -> PredictParams:
    with open(config_path, 'r') as config:
        schema = PredictParamsSchema()
        return schema.load(yaml.safe_load(config))
