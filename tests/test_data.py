import pytest


def test_data_loaders(data_module):

    assert len(data_module.train_dataloader()) > 0, "Training dataloader should not be empty"
    assert len(data_module.val_dataloader()) > 0, "Validation dataloader should not be empty"
    assert len(data_module.test_dataloader()) > 0, "Test dataloader should not be empty"


def test_batch_shape(data_module, config):

    training_config = config["training"]

    batch = next(iter(data_module.train_dataloader()))

    assert batch['input_ids'].shape == (training_config["batch_size"], training_config["max_length"]), "Batch input_ids shape mismatch"
    assert batch['attention_mask'].shape == (training_config["batch_size"], training_config["max_length"]), "Batch attention_mask shape mismatch"
    assert batch['labels'].shape == (training_config["batch_size"],), "Batch labels shape mismatch"
