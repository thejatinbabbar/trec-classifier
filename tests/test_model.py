import pytest
import torch


def test_model_forward_pass(model):
    input_ids = torch.randint(0, 1000, (2, 64))
    attention_mask = torch.ones((2, 64))
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    assert output.logits.shape == (2, 6), "Model output shape mismatch"
