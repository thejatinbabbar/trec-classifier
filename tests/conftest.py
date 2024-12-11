import pytest
import yaml

from data.data import TRECDataModule
from pipeline.model import Classifier


@pytest.fixture
def data_module(config):

    return TRECDataModule(
        data_uri=config["data"]["local_uri"],
        tokenizer=config["model"]["pretrained_model_name"],
        batch_size=config["model"]["batch_size"],
        max_length=config["model"]["max_length"],
        n_workers=config["experiment"]["n_workers"],
        test_size=config["experiment"]["test_size"],
        seed=config["experiment"]["seed"],
    )


@pytest.fixture
def config():
    config_yml = yaml.safe_load(open("config/config.yml"))
    config_yml["model"]["output_model_onnx"] = "model.onnx"
    config_yml["data"]["local_uri"] = "tests/test_data"
    return config_yml


@pytest.fixture
def model(config):
    config_model = config["model"]

    return Classifier(
        n_classes=config_model["n_classes"],
        learning_rate=config_model["learning_rate"],
        max_epochs=config_model["max_epochs"],
        pretrained_model_name=config_model["pretrained_model_name"],
    )
