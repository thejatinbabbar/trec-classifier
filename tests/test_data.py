import pytest
from classifier.data import TRECDataModule


@pytest.fixture
def data_module():
    return TRECDataModule(batch_size=2, max_length=64)


def test_data_loaders(data_module):
    data_module.setup()
    assert len(data_module.train_dataloader()) > 0, "Training dataloader should not be empty"
    assert len(data_module.val_dataloader()) > 0, "Validation dataloader should not be empty"
    assert len(data_module.test_dataloader()) > 0, "Test dataloader should not be empty"


def test_batch_shape(data_module):
    data_module.setup()
    batch = next(iter(data_module.train_dataloader()))
    assert batch['input_ids'].shape == (2, 64), "Batch input_ids shape mismatch"
    assert batch['attention_mask'].shape == (2, 64), "Batch attention_mask shape mismatch"
    assert batch['labels'].shape == (2,), "Batch labels shape mismatch"
