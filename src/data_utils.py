"""Utilities for data handling and preprocessing"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from config import config

class TextDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.texts = [item["sms"] for item in dataset]
        self.labels = [item["label"] for item in dataset]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def prepare_dataloaders(train_dataset, val_dataset, config):
    """Prepare training and validation dataloaders"""
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)

    # Configure tokenizer for padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Create datasets
    train_dataset = TextDataset(train_dataset, tokenizer, config.max_length)
    val_dataset = TextDataset(val_dataset, tokenizer, config.max_length)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0  # Avoid multiprocessing issues
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, tokenizer