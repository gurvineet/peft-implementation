"""Main training script for PEFT fine-tuning"""

import gc
import logging
import os
import psutil
import time
import torch
import numpy as np
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset, Dataset
from config import config

def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format=config.logging_format
    )
    logger = logging.getLogger(__name__)
    return logger

def log_memory_usage(logger):
    """Log current memory usage"""
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / 1024 / 1024
    gpu_mem = f", GPU memory: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB" if torch.cuda.is_available() else ""
    logger.info(f"Memory usage: {mem_usage:.2f} MB{gpu_mem}")
    return mem_usage

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # Calculate accuracy and other metrics
    accuracy = (predictions == labels).mean()

    # Calculate MAE for ratings (add 1 since we're using 0-based indices)
    mae = np.abs((predictions + 1) - (labels + 1)).mean()

    # Calculate per-class accuracy
    class_accuracies = {}
    for i in range(config.num_labels):
        mask = labels == i
        if mask.sum() > 0:
            class_accuracies[f"class_{i+1}_accuracy"] = (predictions[mask] == labels[mask]).mean()

    metrics = {
        "accuracy": accuracy,
        "mae": mae,
        **class_accuracies
    }

    return metrics

def validate_data(examples, logger):
    """Validate the input data"""
    valid_ratings = [rating for rating in examples["rating"] if 1 <= rating <= 5]
    if len(valid_ratings) != len(examples["rating"]):
        logger.warning(f"Found {len(examples['rating']) - len(valid_ratings)} invalid ratings")

    valid_reviews = [review for review in examples["review"] if isinstance(review, str) and len(review.strip()) > 0]
    if len(valid_reviews) != len(examples["review"]):
        logger.warning(f"Found {len(examples['review']) - len(valid_reviews)} invalid reviews")

    return len(valid_ratings) > 0 and len(valid_reviews) > 0

def preprocess_function(examples, tokenizer):
    """Preprocess the examples efficiently"""
    # Validate data
    if not validate_data(examples, logging.getLogger(__name__)):
        raise ValueError("Invalid data format detected")

    # Tokenize the reviews
    tokenized = tokenizer(
        examples["review"],
        truncation=True,
        max_length=config.max_length,
        padding="max_length",
    )

    # Convert ratings to 0-based index for the model (e.g., rating 1 -> class 0)
    tokenized["labels"] = [rating - 1 for rating in examples["rating"]]

    # Validate labels
    if not all(0 <= label < config.num_labels for label in tokenized["labels"]):
        raise ValueError("Invalid label values after conversion")

    return tokenized

def load_dataset_with_retry(logger, max_retries=3):
    """Load dataset with retry mechanism"""
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} to load dataset...")
            log_memory_usage(logger)

            # Load the CSV file using pandas first to check its content
            df = pd.read_csv(config.dataset_name)
            logger.info(f"Raw CSV data loaded: {len(df)} rows")

            # Convert pandas DataFrame to Hugging Face Dataset
            dataset = Dataset.from_pandas(df)
            logger.info(f"Converted to HF Dataset: {len(dataset)} examples")

            # Filter out any invalid ratings
            dataset = dataset.filter(lambda x: 1 <= x["rating"] <= 5)
            logger.info(f"Dataset after filtering: {len(dataset)} examples")

            # Update config with actual dataset size
            if not config.num_samples:
                config.num_samples = len(dataset)
            elif config.num_samples > len(dataset):
                logger.warning(f"Requested {config.num_samples} samples but only {len(dataset)} available")
                config.num_samples = len(dataset)

            # Recalculate steps
            train_samples = int(config.num_samples * config.train_split)
            steps_per_epoch = max(1, int(train_samples / (config.batch_size * config.gradient_accumulation_steps)))
            config.total_steps = steps_per_epoch * config.num_epochs
            config.warmup_steps = int(config.total_steps * config.warmup_ratio)

            # Take a subset if specified
            if config.num_samples < len(dataset):
                dataset = dataset.shuffle(seed=config.random_seed).select(range(config.num_samples))
                logger.info(f"Using {config.num_samples} examples from the dataset")

            logger.info(f"Final dataset size: {len(dataset)} examples")
            logger.info(f"Training samples: {train_samples}")
            logger.info(f"Steps per epoch: {steps_per_epoch}")

            return dataset

        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                raise Exception(f"Failed to load dataset after {max_retries} attempts")

def setup_peft_model(logger):
    """Initialize the PEFT model with LoRA configuration"""
    logger.info(f"Initializing base model: {config.base_model_name}")

    # Initialize base model
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_name,
        num_labels=config.num_labels,
        id2label=config.id2label,
        label2id=config.label2id
    )

    # Configure tokenizer
    logger.info("Configuring tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name,
        model_max_length=config.model_max_length,
        padding_side=config.padding_side
    )

    # Set padding token for GPT-2
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # Setup LoRA configuration
    logger.info("Creating LoRA configuration...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
    )

    # Convert to PEFT model
    logger.info("Converting to PEFT model...")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer

def main():
    logger = setup_logging()
    logger.info("=== Starting PEFT Fine-tuning Process ===")

    try:
        # Load and prepare dataset
        dataset = load_dataset_with_retry(logger)

        # Split dataset
        dataset = dataset.train_test_split(
            train_size=config.train_split,
            shuffle=True,
            seed=config.random_seed
        )

        # Setup model and tokenizer
        model, tokenizer = setup_peft_model(logger)

        # Preprocess datasets
        logger.info("Preprocessing datasets...")
        tokenized_train = dataset["train"].map(
            lambda x: preprocess_function(x, tokenizer),
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Preprocessing train dataset"
        )
        tokenized_test = dataset["test"].map(
            lambda x: preprocess_function(x, tokenizer),
            batched=True,
            remove_columns=dataset["test"].column_names,
            desc="Preprocessing test dataset"
        )

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            num_train_epochs=config.num_epochs,
            weight_decay=config.weight_decay,
            evaluation_strategy="steps",
            eval_steps=config.eval_steps,
            save_strategy="steps",
            save_steps=config.save_steps,
            load_best_model_at_end=True,
            save_total_limit=config.save_total_limit,
            logging_steps=config.logging_steps,
            remove_unused_columns=False,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_grad_norm=config.max_grad_norm,
            warmup_steps=config.warmup_steps,
            fp16=config.fp16,
            disable_tqdm=config.disable_tqdm,
        )

        # Initialize trainer
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        # Initial evaluation
        logger.info("Running initial evaluation...")
        initial_metrics = trainer.evaluate()
        logger.info(f"Initial metrics: {initial_metrics}")

        # Training
        logger.info("Starting training...")
        trainer.train()

        # Final evaluation
        logger.info("Running final evaluation...")
        final_metrics = trainer.evaluate()
        logger.info(f"Final metrics: {final_metrics}")

        # Save the final model
        logger.info("Saving final model...")
        trainer.save_model(config.output_dir)
        logger.info("Training completed successfully!")

        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Error during training process: {str(e)}")
        log_memory_usage(logger)
        raise e

if __name__ == "__main__":
    main()