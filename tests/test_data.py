import pytest  # noqa


def test_data_split(data_module):

    assert len(data_module.train_dataset) == 16
    assert len(data_module.val_dataset) == 4
    assert len(data_module.test_dataset) == 20


def test_batch_shape(data_module, config):

    config_model = config["model"]

    batch = next(iter(data_module.train_dataloader()))

    assert batch["input_ids"].shape == (
        config_model["batch_size"],
        config_model["max_length"],
    ), "Batch input_ids shape mismatch"
    assert batch["attention_mask"].shape == (
        config_model["batch_size"],
        config_model["max_length"],
    ), "Batch attention_mask shape mismatch"
    assert batch["labels"].shape == (config_model["batch_size"],), "Batch labels shape mismatch"
