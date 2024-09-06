import logging
import os
from argparse import ArgumentParser

import mlflow
import yaml

from classifier.data import TRECDataModule
from classifier.model import Classifier

if __name__ == "__main__":

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    logging.basicConfig(level=logging.INFO)

    arg_parser = ArgumentParser()
    arg_parser.add_argument("--config", default="config/config.yml")
    arg_parser.add_argument("--experiment_name", default="trec-classification-training")

    args = arg_parser.parse_args()

    config = yaml.safe_load(open(args.config))

    # Set up MLflow tracking URI
    mlflow.set_tracking_uri(config["mlflow"]["mlflow_uri"])
    mlflow.set_experiment(experiment_name=args.experiment_name)

    training_config = config["training"]
    for param, value in training_config.items():
        mlflow.log_param(param, value)

    # Initialize data module
    trec_data_module = TRECDataModule(
        tokenizer=training_config["pretrained_model_name"],
        batch_size=training_config["batch_size"],
        max_length=training_config["max_length"],
        n_workers=training_config["n_workers"],
        test_size=training_config["test_size"],
        seed=training_config["seed"],
    )

    # Initialize model
    model = Classifier(
        n_classes=training_config["n_classes"],
        learning_rate=training_config["learning_rate"],
        max_epochs=training_config["max_epochs"],
        pretrained_model_name=training_config["pretrained_model_name"],
    )

    # Train model
    model.train_model(trec_data_module)

    # Run on test data
    model.evaluate_model(trec_data_module)

    # Export model
    pytorch_file_path = training_config["output_model_pytorch"]
    onnx_file_path = training_config["output_model_onnx"]

    model.save_model(pytorch_file_path, onnx_file_path)
    mlflow.log_artifact(pytorch_file_path)
    mlflow.log_artifact(onnx_file_path)
