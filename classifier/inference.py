import torch
from transformers import AutoTokenizer
import onnxruntime as ort

from classifier.data import TRECDataModule
from classifier.model import Classifier


class InferencePipeline:
    def __init__(self, model_type="pytorch", checkpoint_path=None, onnx_model_path=None):
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.onnx_model_path = onnx_model_path
        self.data_module = TRECDataModule()

        self.model = Classifier.load_from_checkpoint(checkpoint_path)
        # if self.model_type in ["pytorch", "onnx"]:
        #     self.model = Classifier.load_model_from_checkpoint(checkpoint_path)
        # else:
        #     raise ValueError("Unsupported model type: choose 'pytorch' or 'onnx'")

    def predict(self, text):

        inputs = self.data_module.generate_encodings([text])

        if self.model_type == "pytorch":
            return self.model.predict_with_pytorch(inputs)
        elif self.model_type == "onnx":
            session = ort.InferenceSession(self.onnx_model_path)

            outputs = session.run(None, inputs)
            logits = torch.tensor(outputs[0])
            predicted_class_id = torch.argmax(logits, dim=-1).item()

            return predicted_class_id
        else:
            raise ValueError("Unsupported model type: choose 'pytorch' or 'onnx'")


if __name__ == "__main__":
    # Example usage
    pipeline = InferencePipeline(model_type="pytorch", checkpoint_path="experiments/20240821/model.pth")
    # Or use the ONNX model
    # pipeline = InferencePipeline(model_type="onnx", onnx_model_path="model.onnx")

    text = "What is the capital of France?"
    predicted_class = pipeline.predict(text)
    print(f"Predicted class: {predicted_class}")
