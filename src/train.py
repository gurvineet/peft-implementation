"""Main training script for PEFT fine-tuning"""

import gc
import logging
import os
import psutil
import time
import torch
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
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
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

def preprocess_function(examples, tokenizer):
    """Preprocess the examples efficiently"""
    # Tokenize the reviews
    tokenized = tokenizer(
        examples["review"],
        truncation=True,
        max_length=config.max_length,
        padding="max_length",
    )

    # Convert star ratings to labels (0-5)
    tokenized["labels"] = [
        min(5, max(0, star)) for star in examples["star"]
    ]

    return tokenized

def load_dataset_with_retry(logger, max_retries=3):
    """Load dataset with retry mechanism and progress tracking"""
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} to load dataset...")
            log_memory_usage(logger)

            logger.info("Starting dataset download...")
            dataset = load_dataset(
                config.dataset_name,
                split=f"train[:{config.num_samples}]",
                cache_dir=config.cache_dir
            )

            logger.info("Dataset download complete")
            logger.info(f"Dataset loaded with {len(dataset)} examples")
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

        # Setup training arguments with optimized values
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            num_train_epochs=config.num_epochs,
            weight_decay=config.weight_decay,
            eval_strategy="steps",
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