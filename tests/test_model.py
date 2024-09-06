import os.path

import torch
import pytest
from classifier.model import Classifier


# def test_save_and_load_model(model, tmpdir, data_module):
#
#     pytorch_file_path = os.path.join(tmpdir, "model.pth")
#     onnx_file_path = os.path.join(tmpdir, "model.onnx")
#
#     model.evaluate_model(data_module)
#     model.save_model(tmpdir)
#
#     assert os.path.exists(pytorch_file_path)
#     assert os.path.exists(onnx_file_path)
#
#     _ = Classifier.load_from_checkpoint(pytorch_file_path)
#     assert True, "Error while loading model"


def test_model_forward_pass(model):
    input_ids = torch.randint(0, 1000, (2, 64))
    attention_mask = torch.ones((2, 64))
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    assert output.logits.shape == (2, 6), "Model output shape mismatch"
