import click
from src.enities import read_train_pipeline_params, TrainPipelineParams
import logging
from src.data import read_data, split_train_val
import json
from src.features import (build_transformer,
                          make_features,
                          extract_target,
                          fit_transformer)
from src.models import (train_model,
                        save_model,
                        predict_probabilities,
                        evaluate_model)

handler = logging.FileHandler("../logs/train.log")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


def train_pipeline(params: TrainPipelineParams):
    logger.info(f"Start training pipeline with params {params}")
    logger.debug("Read data...")
    data = read_data(params.input_data_path)
    logger.debug("Split data...")
    train_df, val_df = split_train_val(data, params.splitting_params)

    logger.info(f"train_df.shape is {train_df.shape}")
    logger.info(f"val_df.shape is {val_df.shape}")

    transformer = build_transformer(params.feature_params)
    transformer, train = fit_transformer(transformer, train_df)
    train_features = make_features(transformer, train)
    train_target = extract_target(train, params.feature_params)
    logger.debug("Split data...")

    logger.info(f"train_features.shape is {train_features.shape}")

    model = train_model(
        train_features, train_target, params.train_params
    )

    val_features = make_features(transformer, val_df)
    val_target = extract_target(val_df, params.feature_params)

    logger.info(f"val_features.shape is {val_features.shape}")
    probabilities = predict_probabilities(
        model,
        val_features
    )

    metrics = evaluate_model(
        probabilities,
        val_target,
    )

    with open(params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"metrics is {metrics}")

    path_to_model = save_model(model, params.output_model_path)

    return path_to_model, metrics


@click.command(name="train")
@click.argument("config_path")
def train_command(config_path: str):
    params = read_train_pipeline_params(config_path)
    train_pipeline(params)


if __name__ == "__main__":
    train_command()
