"""Utilities for data handling and preprocessing"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from config import config

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Add special tokens and proper formatting for the model
        text = f"{text}</s>"

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze()
        }

def prepare_dataloaders(texts, config):
    """Prepare training and validation dataloaders"""
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)

    # Configure tokenizer for DeepSeek model
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Split data into train/val
    train_size = int(0.9 * len(texts))
    train_texts = texts[:train_size]
    val_texts = texts[train_size:]

    # Create datasets
    train_dataset = TextDataset(train_texts, tokenizer, config.max_length)
    val_dataset = TextDataset(val_texts, tokenizer, config.max_length)

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