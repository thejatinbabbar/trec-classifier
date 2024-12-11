import logging
import os
from argparse import ArgumentParser

import boto3
import mlflow
import onnx
import yaml

from data.data import TRECDataModule, download_from_s3
from pipeline.model import Classifier

if __name__ == "__main__":

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    logging.basicConfig(level=logging.INFO)

    arg_parser = ArgumentParser()
    arg_parser.add_argument("--config", default="config/config.yml")
    arg_parser.add_argument("--experiment_name", default="trec-classification")
    arg_parser.add_argument("--endpoint_url", default=None)

    args = arg_parser.parse_args()

    config = yaml.safe_load(open(args.config))

    step_to_execute = config["step_to_execute"]

    # Set up MLflow tracking URI
    mlflow.set_tracking_uri(config["mlflow"]["mlflow_uri"])
    mlflow.set_experiment(experiment_name=f"{args.experiment_name}-{step_to_execute}")

    s3_client = boto3.client(
        "s3",
        endpoint_url=args.endpoint_url,
        aws_access_key_id="localstack",
        aws_secret_access_key="localstack",
        region_name="us-east-1",
    )
    download_from_s3(s3_client, config["data"]["s3_bucket"], config["data"]["local_uri"], config["data"]["s3_prefix"])

    if step_to_execute == "train":

        config_model = config["model"]
        config_experiment = config["experiment"]

        for param, value in config["model"].items():
            mlflow.log_param(param, value)
        for param, value in config["experiment"].items():
            mlflow.log_param(param, value)

        # Initialize data module
        trec_data_module = TRECDataModule(
            data_uri=config["data"]["local_uri"],
            tokenizer=config_model["pretrained_model_name"],
            batch_size=config_model["batch_size"],
            max_length=config_model["max_length"],
            n_workers=config_experiment["n_workers"],
            test_size=config_experiment["test_size"],
            seed=config_experiment["seed"],
        )

        # Initialize model
        model = Classifier(
            n_classes=config_model["n_classes"],
            learning_rate=config_model["learning_rate"],
            max_epochs=config_model["max_epochs"],
            pretrained_model_name=config_model["pretrained_model_name"],
        )

        # Train model
        model.train_model(trec_data_module)

        # Run on test data
        model.evaluate_model(trec_data_module)

        # Export model
        pytorch_file_path = config_model["output_model_pytorch"]
        onnx_file_path = config_model["output_model_onnx"]

        model.save_model(pytorch_file_path, onnx_file_path)
        # mlflow.log_artifact(pytorch_file_path)
        # mlflow.log_artifact(onnx_file_path)
        # mlflow.pytorch.log_model(model, "pytorch_model")

        onnx_model = onnx.load(onnx_file_path)
        mlflow.onnx.log_model(onnx_model, "model")

    elif step_to_execute == "evaluate":
        evaluation_config = config["training"]

        # Initialize data module
        trec_data_module = TRECDataModule(
            data_uri=config["data"]["local_folder_path"],
            tokenizer=evaluation_config["pretrained_model_name"],
            batch_size=evaluation_config["batch_size"],
            max_length=evaluation_config["max_length"],
            n_workers=evaluation_config["n_workers"],
            test_size=evaluation_config["test_size"],
            seed=evaluation_config["seed"],
        )

        # Initialize model and load checkpoint
        model_uri = mlflow.artifacts.download_artifacts(config["inference"]["model_uri"])
        mlflow.log_param("model_uri", model_uri)

        model = Classifier.load_from_checkpoint(
            model_uri,
            n_classes=evaluation_config["n_classes"],
            learning_rate=evaluation_config["learning_rate"],
            max_epochs=evaluation_config["max_epochs"],
            pretrained_model_name=evaluation_config["pretrained_model_name"],
        )

        # Evaluate model
        model.evaluate_model(trec_data_module)
    else:
        raise ValueError(f"Invalid step_to_execute: {step_to_execute}")
