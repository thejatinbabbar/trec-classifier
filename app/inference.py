import os

import mlflow
import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoTokenizer

from data.data import generate_encodings


class InferencePipeline:
    def __init__(self, config):

        self.config = config

        model_file = os.path.join(self.config["inference"]["artifacts_uri"], self.config["model"]["output_model_onnx"])

        if os.path.exists(model_file):
            self.session = ort.InferenceSession(model_file)
        else:
            mlflow.set_tracking_uri(self.config["mlflow"]["mlflow_uri"])
            model_uri = mlflow.artifacts.download_artifacts(config["inference"]["onnx_model_uri"])
            self.session = ort.InferenceSession(f"{model_uri}/{model_file}")

        self.tokenizer = AutoTokenizer.from_pretrained(config["model"]["pretrained_model_name"])

    def run_onnx_session(self, text):
        inputs = generate_encodings(self.tokenizer, [text], self.config["model"]["max_length"])
        inputs = {
            "input_ids": np.array(inputs["input_ids"]),
            "attention_mask": np.array(inputs["attention_mask"]),
        }
        outputs = self.session.run(output_names=["output"], input_feed=inputs)
        logits = torch.tensor(outputs[0])
        predicted_class_id = torch.argmax(logits, dim=-1).item()
        confidence_score = torch.softmax(logits[0], dim=-1)[predicted_class_id].item()
        return predicted_class_id, confidence_score
