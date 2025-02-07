"""Main training script for PEFT fine-tuning"""

import torch
import numpy as np
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from config import config
from data_utils import prepare_dataloaders
import time
import psutil
import os

def log_memory_usage():
    """Log current memory usage"""
    process = psutil.Process(os.getpid())
    print(f"\nCurrent memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": (predictions == labels).mean(),
    }

def load_dataset_with_retry(max_retries=3, timeout=15):
    """Load dataset with retry mechanism and detailed progress tracking"""
    for attempt in range(max_retries):
        try:
            print(f"\nAttempt {attempt + 1}/{max_retries} to load dataset...")
            print("Initializing dataset load with minimal size...")
            log_memory_usage()

            print("Starting dataset download...")
            dataset = load_dataset(
                "sealuzh/app_reviews",
                split=f"train[:2000]",
                cache_dir=".cache"
            )

            print("Dataset download complete")
            log_memory_usage()

            print(f"Dataset loaded successfully with {len(dataset)} examples!")
            print("Dataset features:", dataset.features)
            print("\nSample entry:", dataset[0])

            return dataset

        except Exception as e:
            print(f"\nAttempt {attempt + 1} failed")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {str(e)}")
            log_memory_usage()

            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                print("All retry attempts exhausted.")
                print("Final memory usage:")
                log_memory_usage()
                raise Exception(f"Failed to load dataset after {max_retries} attempts. Last error: {str(e)}")

def setup_peft_model():
    """Initialize the PEFT model with LoRA configuration"""
    print("\nInitializing base model...")
    print(f"Using model: {config.base_model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_name,
        num_labels=config.num_labels,
        id2label=config.id2label,
        label2id=config.label2id
    )

    # Configure padding token for GPT-2
    print("Configuring tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
        print("Set padding token to EOS token")

    print("\nCreating LoRA configuration...")
    print(f"LoRA params - r: {config.lora_r}, alpha: {config.lora_alpha}, dropout: {config.lora_dropout}")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=["c_attn", "c_proj"]  # GPT-2 specific target modules
    )

    print("\nConverting to PEFT model...")
    model = get_peft_model(model, peft_config)
    print("\nTrainable parameters:")
    model.print_trainable_parameters()

    return model, tokenizer

def main():
    print("\n=== Starting PEFT Fine-tuning Process ===\n")

    try:
        # Load and prepare dataset
        full_dataset = load_dataset_with_retry()
        print(f"Successfully loaded dataset with {len(full_dataset)} examples")

        # Split dataset
        dataset = full_dataset.train_test_split(
            train_size=1600,  # 80% of 2000
            test_size=400,    # 20% of 2000
            shuffle=True,
            seed=42
        )

        # Setup model and tokenizer
        model, tokenizer = setup_peft_model()

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            num_train_epochs=config.num_epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=2,
            logging_steps=10,
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
        )

        # Initial evaluation
        print("\nInitial evaluation:")
        initial_metrics = trainer.evaluate()
        print(f"Initial metrics: {initial_metrics}")

        # Training
        print("\nStarting training...")
        trainer.train()

        # Final evaluation
        print("\nFinal evaluation:")
        final_metrics = trainer.evaluate()
        print(f"Final metrics: {final_metrics}")

        # Save the final model
        print("\nSaving final model...")
        trainer.save_model(config.output_dir)
        print("Training completed!")

    except Exception as e:
        print(f"\nError during training process: {str(e)}")
        log_memory_usage()
        raise e

if __name__ == "__main__":
    main()