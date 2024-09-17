import mlflow
import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy, Precision, Recall
from transformers import AutoModelForSequenceClassification


class Classifier(pl.LightningModule):
    def __init__(
        self,
        n_classes,
        learning_rate,
        max_epochs,
        pretrained_model_name,
    ):
        super().__init__()

        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
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

        self.accuracy = Accuracy(task="multiclass", num_classes=self.n_classes)
        self.precision = Precision(task="multiclass", num_classes=self.n_classes, average="macro")
        self.recall = Recall(task="multiclass", num_classes=self.n_classes, average="macro")

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = self(input_ids, attention_mask, labels)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels)

        # Get predictions
        preds = torch.argmax(outputs.logits, dim=1)

        # Compute metrics
        acc = self.accuracy(preds, labels)
        prec = self.precision(preds, labels)
        rec = self.recall(preds, labels)

        # Log metrics to MLflow
        mlflow.log_metric("train_loss", loss.item())
        mlflow.log_metric("train_accuracy", acc.item())
        mlflow.log_metric("train_precision", prec.item())
        mlflow.log_metric("train_recall", rec.item())

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = self(input_ids, attention_mask, labels)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels)

        # Get predictions
        preds = torch.argmax(outputs.logits, dim=1)

        # Compute metrics
        acc = self.accuracy(preds, labels)
        prec = self.precision(preds, labels)
        rec = self.recall(preds, labels)

        # Log loss to MLflow
        mlflow.log_metric("val_loss", loss)
        mlflow.log_metric("val_accuracy", acc.item())
        mlflow.log_metric("val_precision", prec.item())
        mlflow.log_metric("val_recall", rec.item())

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

        self.trainer.fit(self, train_dataloaders=data_loader)

    def evaluate_model(self, data_loader):

        if self.model is None:
            raise ValueError("Model must be initialized before training.")

        results = self.trainer.test(self, data_loader)
        return results

    def save_model(self, pytorch_file_path, onnx_file_path):

        # Save the PyTorch model
        self.trainer.save_checkpoint(pytorch_file_path)
        print(f"PyTorch model saved to {pytorch_file_path}")

        # Save the ONNX model
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
