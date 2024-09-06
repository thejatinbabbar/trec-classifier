import pytest
import yaml

from classifier.data import TRECDataModule
from classifier.model import Classifier


@pytest.fixture
def data_module(config):
    training_config = config["training"]

    return TRECDataModule(
        tokenizer=training_config["pretrained_model_name"],
        batch_size=training_config["batch_size"],
        max_length=training_config["max_length"],
        n_workers=training_config["n_workers"],
        test_size=training_config["test_size"],
        seed=training_config["seed"],
    )


@pytest.fixture
def config():
    return yaml.safe_load(open("config/config.yml"))


@pytest.fixture
def model(config):
    training_config = config["training"]

    return Classifier(
        n_classes=training_config["n_classes"],
        learning_rate=training_config["learning_rate"],
        max_epochs=training_config["max_epochs"],
        pretrained_model_name=training_config["pretrained_model_name"],
    )
