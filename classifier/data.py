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
    def __init__(self, tokenizer, batch_size, max_length, n_workers, test_size, seed):
        super().__init__()

        dataset = load_dataset("trec", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        self.batch_size = batch_size
        self.max_length = max_length
        self.n_workers = n_workers
        self.test_size = test_size
        self.seed = seed

        self.allow_zero_length_dataloader_with_multiple_devices = True

        # Tokenize the dataset
        self.train_dataset = self.tokenize_dataset(dataset["train"], max_length)
        self.test_dataset = self.tokenize_dataset(dataset["test"], max_length)
        self.train_dataset, self.val_dataset = train_test_split(
            self.train_dataset, test_size=test_size, shuffle=True, random_state=seed
        )

    def tokenize_dataset(self, dataset, max_length):
        encodings = generate_encodings(self.tokenizer, dataset["text"], max_length)
        labels = dataset["coarse_label"]
        return TrecDataset(encodings, labels)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            persistent_workers=True,
        )


def generate_encodings(tokenizer, texts, max_length):
    encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    return encodings
