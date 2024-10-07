import os

import pytorch_lightning as pl
import torch
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


def download_from_s3(s3_client, bucket_name, local_dir, s3_prefix=None):
    try:
        # Create local directory if it doesn't exist
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        # List all objects in the S3 bucket with the specified prefix
        paginator = s3_client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)

        for page in page_iterator:
            if "Contents" in page:
                for obj in page["Contents"]:
                    # Get the object key (file path in the bucket)
                    s3_key = obj["Key"]

                    # Define the local file path
                    local_file_path = os.path.join(local_dir, os.path.relpath(s3_key, s3_prefix or ""))

                    # Ensure the local subdirectory exists
                    local_subdir = os.path.dirname(local_file_path)
                    if not os.path.exists(local_subdir):
                        os.makedirs(local_subdir)

                    # Download the file if it doesn't exist or if it's different
                    if not os.path.exists(local_file_path) or obj["Size"] != os.path.getsize(local_file_path):
                        print(f"Downloading {s3_key} to {local_file_path}")
                        s3_client.download_file(bucket_name, s3_key, local_file_path)
                    else:
                        print(f"{s3_key} is already up to date.")

    except Exception as e:
        raise e


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
    def __init__(self, data_uri, tokenizer, batch_size, max_length, n_workers, test_size, seed):
        super().__init__()

        dataset = load_from_disk(data_uri)
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
