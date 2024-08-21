import logging
import os
from argparse import ArgumentParser
from datetime import datetime

import mlflow

from classifier.data import TRECDataModule
from classifier.model import Classifier

if __name__ == "__main__":

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    logging.basicConfig(level=logging.INFO)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    arg_parser = ArgumentParser()
    arg_parser.add_argument("--output_dir", default="experiments", help="path to output directory")
    arg_parser.add_argument("--experiment_name", default=now, help="name of experiment")

    args = arg_parser.parse_args()

    experiment_dir = os.path.join(args.output_dir, args.experiment_name)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(experiment_dir)

    # Set up MLflow tracking URI
    mlflow_uri = "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(args.experiment_name)

    # Initialize data module
    trec_data_module = TRECDataModule(batch_size=16, max_length=128)
    trec_data_module.setup()

    with mlflow.start_run():
        # Initialize model
        model = Classifier(n_classes=6, learning_rate=1e-5, max_epochs=2, model_dir=experiment_dir)
        model.initialize_model(pretrained_model_name="prajjwal1/bert-tiny")

        # Train model
        model.train_model(trec_data_module)

        # Run on test data
        model.evaluate(trec_data_module)

        # Export model
        model.save_model(experiment_dir)

    _ = None
