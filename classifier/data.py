import logging

import pytorch_lightning as pl
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class TrecDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class TRECDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=16, max_length=512, dataset_name="trec", tokenizer_name="distilbert-base-uncased"):
        super().__init__()
        self.batch_size = batch_size
        self.max_length = max_length
        self.dataset_name = dataset_name
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        self.allow_zero_length_dataloader_with_multiple_devices = True

    def setup(self, stage=None):
        # Load the dataset
        dataset = load_dataset(self.dataset_name)

        # Tokenize the dataset
        self.train_dataset = self.tokenize_dataset(dataset["train"][:100])
        self.test_dataset = self.tokenize_dataset(dataset["test"][:50])
        self.train_dataset, self.val_dataset = train_test_split(
            self.train_dataset, test_size=0.2, shuffle=True, random_state=42
        )

        # Print out some details
        logging.info(f"Training DataLoader size: {len(self.train_dataloader())}")
        logging.info(f"Validation DataLoader size: {len(self.val_dataloader())}")
        logging.info(f"Test DataLoader size: {len(self.test_dataloader())}")

    def tokenize_dataset(self, dataset):
        encodings = self.tokenizer(dataset["text"], truncation=True, padding="max_length", max_length=self.max_length)
        labels = dataset["coarse_label"]
        return TrecDataset(encodings, labels)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=10, persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=10, persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=10, persistent_workers=True
        )
