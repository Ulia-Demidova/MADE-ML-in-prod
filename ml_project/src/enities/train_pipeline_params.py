from dataclasses import dataclass
from .train_params import TrainParams
from .feature_params import FeatureParams
from .splitting_params import SplittingParams
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class TrainPipelineParams:
    input_data_path: str
    output_model_path: str
    output_transformer_path: str
    metric_path: str
    splitting_params: SplittingParams
    train_params: TrainParams
    feature_params: FeatureParams


TrainPipelineParamsSchema = class_schema(TrainPipelineParams)


def read_train_pipeline_params(config_path: str) -> TrainPipelineParams:
    with open(config_path, 'r') as config:
        schema = TrainPipelineParamsSchema()
        return schema.load(yaml.safe_load(config))
