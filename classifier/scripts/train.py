import mlflow
import mlflow.pytorch
import pytorch_lightning as pl
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, default_data_collator


class Classifier(pl.LightningModule):
    def __init__(self, n_classes, lr):
        super().__init__()
        self.lr = lr
        self.model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=n_classes
        )

        # Freeze all parameters except the classifier head
        for param in self.model.distilbert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["coarse_label"],
        }
        outputs = self.model(**inputs)
        loss = outputs.loss
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


def preprocess_data(examples):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)


def main():
    # Load and preprocess data
    dataset = load_dataset("trec", trust_remote_code=True)
    encoded_dataset = dataset.map(preprocess_data, batched=True)
    train_dataset = encoded_dataset["train"].shuffle(seed=42)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=default_data_collator)

    # Initialize MLflow
    mlflow.start_run()

    # Log parameters (example)
    mlflow.log_param("learning_rate", 1e-5)
    mlflow.log_param("batch_size", 16)
    mlflow.log_param("epochs", 3)

    # Initialize model and trainer
    model = Classifier(n_classes=6, lr=1e-5)
    trainer = pl.Trainer(max_epochs=3)
    trainer.fit(model, train_dataloader)

    # End the MLflow run
    mlflow.end_run()


if __name__ == "__main__":
    main()
