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
    arg_parser.add_argument("--experiment_name", default="trec-classification-evaluation")

    args = arg_parser.parse_args()

    config = yaml.safe_load(open(args.config))

    # Set up MLflow tracking URI
    mlflow.set_tracking_uri(config["mlflow"]["mlflow_uri"])
    mlflow.set_experiment(experiment_name=args.experiment_name)

    training_config = config["training"]

    # Initialize data module
    trec_data_module = TRECDataModule(
        tokenizer=training_config["pretrained_model_name"],
        batch_size=training_config["batch_size"],
        max_length=training_config["max_length"],
        n_workers=training_config["n_workers"],
        test_size=training_config["test_size"],
        seed=training_config["seed"],
    )

    # Initialize model and load checkpoint
    model_uri = mlflow.artifacts.download_artifacts(config["inference"]["model_uri"])
    mlflow.log_param("model_uri", model_uri)

    model = Classifier.load_from_checkpoint(
        model_uri,
        n_classes=training_config["n_classes"],
        learning_rate=training_config["learning_rate"],
        max_epochs=training_config["max_epochs"],
        pretrained_model_name=training_config["pretrained_model_name"],
    )

    # Evaluate model
    model.evaluate_model(trec_data_module)
