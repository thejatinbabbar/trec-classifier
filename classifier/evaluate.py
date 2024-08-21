import logging
import os
from argparse import ArgumentParser
from datetime import datetime

from classifier.data import TRECDataModule
from classifier.model import Classifier

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    arg_parser = ArgumentParser()
    arg_parser.add_argument("--output_dir", default="experiments", help="path to output directory")
    arg_parser.add_argument("--experiment_name", default="20240821", help="name of experiment")

    args = arg_parser.parse_args()

    experiment_dir = os.path.join(args.output_dir, args.experiment_name)
    model_path = os.path.join(str(experiment_dir), "model.pth")

    # Initialize data module
    trec_data_module = TRECDataModule(batch_size=16)
    trec_data_module.setup()

    # Initialize model and load checkpoint
    model = Classifier.load_from_checkpoint(model_path)

    # Evaluate model
    model.evaluate_model(trec_data_module)
