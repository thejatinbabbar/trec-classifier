import os

import mlflow
import pytorch_lightning as pl
import torch
from transformers import AutoModelForSequenceClassification, DistilBertForSequenceClassification


class Classifier(pl.LightningModule):
    def __init__(
        self,
        n_classes=6,
        learning_rate=1e-5,
        max_epochs=2,
        model_dir=".",
        pretrained_model_name="prajjwal1/bert-tiny",
    ):
        super().__init__()

        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model_dir = model_dir if model_dir is not None else "checkpoint"
        self.pretrained_model_name = pretrained_model_name

        self.trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            logger=False,
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_model_name,
            num_labels=self.n_classes,
        )

        # Freeze all parameters except the classifier head
        for param in self.model.bert.parameters():
            param.requires_grad = False

        self.log_params_to_mlflow()

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def log_params_to_mlflow(self):
        mlflow.log_param("learning_rate", self.learning_rate)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = self(input_ids, attention_mask, labels)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels)

        # Log loss to MLflow
        mlflow.log_metric("train_loss", loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = self(input_ids, attention_mask, labels)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels)

        # Log loss to MLflow
        mlflow.log_metric("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = self(input_ids, attention_mask, labels)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels)

        # Log loss to MLflow
        mlflow.log_metric("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def train_model(self, data_loader):

        if self.model is None:
            raise ValueError("Model must be initialized before training.")

        self.trainer.fit(self, data_loader)

    def evaluate_model(self, data_loader):

        if self.model is None:
            raise ValueError("Model must be initialized before training.")

        results = self.trainer.test(self, data_loader)
        return results

    def save_model(self, output_dir):

        # Save the PyTorch model
        pytorch_file_path = os.path.join(output_dir, "model.pth")
        self.trainer.save_checkpoint(pytorch_file_path)
        print(f"PyTorch model saved to {pytorch_file_path}")

        # Save the ONNX model
        onnx_file_path = os.path.join(output_dir, "model.onnx")
        dummy_input = torch.ones(1, 64, dtype=torch.long)
        torch.onnx.export(
            self,
            (dummy_input, dummy_input),
            onnx_file_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["output"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "seq_len"},
                "attention_mask": {0: "batch_size", 1: "seq_len"},
                "output": {0: "batch_size"},
            },
        )
        print(f"ONNX model saved to {onnx_file_path}")

    def predict_with_pytorch(self, inputs):

        with torch.no_grad():
            outputs = self(inputs["input_ids"], inputs["attention_mask"])

        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=-1).item()

        return predicted_class_id
