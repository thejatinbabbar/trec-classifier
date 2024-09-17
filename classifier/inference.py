import mlflow
import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoTokenizer

from classifier.data import generate_encodings


class InferencePipeline:
    def __init__(self, config):

        self.config = config

        mlflow.set_tracking_uri(self.config["mlflow"]["mlflow_uri"])
        model_uri = mlflow.artifacts.download_artifacts(config["inference"]["onnx_model_uri"])
        self.session = ort.InferenceSession(f"{model_uri}/model.onnx")

        self.tokenizer = AutoTokenizer.from_pretrained(config["training"]["pretrained_model_name"])

    def run_onnx_session(self, text):
        inputs = generate_encodings(self.tokenizer, [text], self.config["training"]["max_length"])
        inputs = {
            "input_ids": np.array(inputs["input_ids"]),
            "attention_mask": np.array(inputs["attention_mask"]),
        }
        outputs = self.session.run(output_names=["output"], input_feed=inputs)
        logits = torch.tensor(outputs[0])
        predicted_class_id = torch.argmax(logits, dim=-1).item()
        return predicted_class_id
