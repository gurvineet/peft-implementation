"""Main training script for PEFT fine-tuning"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
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

def load_dataset_with_retry(max_retries=3, timeout=15):  # Further reduced timeout
    """Load dataset with retry mechanism and detailed progress tracking"""
    for attempt in range(max_retries):
        try:
            print(f"\nAttempt {attempt + 1}/{max_retries} to load dataset...")
            print("Initializing dataset load with minimal size...")
            log_memory_usage()

            print("Starting dataset download...")
            dataset = load_dataset(
                "sealuzh/app_reviews",
                split=f"train[:50]",  # Keeping minimal size
                cache_dir=".cache"  # Explicit cache directory
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
                wait_time = 2 ** attempt  # Exponential backoff
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

    return model

def train_epoch(model, train_loader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    print(f"\nStarting training epoch {epoch+1}")
    print(f"Number of batches: {len(train_loader)}")

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=True)

    for batch_idx, batch in enumerate(progress_bar):
        try:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            current_accuracy = correct / total

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            progress_bar.set_postfix({
                "batch": f"{batch_idx+1}/{len(train_loader)}",
                "loss": f"{loss.item():.4f}",
                "accuracy": f"{current_accuracy:.4f}"
            })

        except Exception as e:
            print(f"\nError in batch {batch_idx}: {str(e)}")
            continue

    epoch_loss = total_loss / len(train_loader)
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy

def evaluate(model, val_loader, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    print("\nStarting evaluation...")
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Evaluating", leave=True)
        for batch in progress_bar:
            try:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss.item()
                total_loss += loss

                predictions = torch.argmax(outputs.logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                current_accuracy = correct / total if total > 0 else 0
                progress_bar.set_postfix({
                    "loss": f"{loss:.4f}",
                    "accuracy": f"{current_accuracy:.4f}"
                })

            except Exception as e:
                print(f"\nError during evaluation: {str(e)}")
                continue

    accuracy = correct / total if total > 0 else 0
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
    return avg_loss, accuracy


def main():
    print("\n=== Starting PEFT Fine-tuning Process ===\n")

    # Load app reviews dataset
    print("Loading app reviews dataset...")
    try:
        full_dataset = load_dataset_with_retry()
        print(f"Successfully loaded dataset with {len(full_dataset)} examples")

        # Take a subset for faster training during development
        print("Shuffling and splitting dataset...")
        full_dataset = full_dataset.shuffle(seed=42)

        # Calculate split sizes
        train_size = 40     # Using 40 samples for training
        test_size = 10      # Using 10 samples for testing
        print(f"Using {train_size} examples for training and {test_size} for testing")

        # Split dataset
        dataset = full_dataset.train_test_split(
            train_size=train_size,
            test_size=test_size,
            shuffle=True,
            seed=42
        )

        train_dataset = dataset['train']
        val_dataset = dataset['test']
        print(f"Split complete. Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

        # Prepare data
        print("\nPreparing dataloaders...")
        train_loader, val_loader, _ = prepare_dataloaders(train_dataset, val_dataset, config)
        print("Dataloaders prepared successfully")

        # Setup model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {device}")
        model = setup_peft_model()
        model.to(device)

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )

        print(f"\nTraining for {config.num_epochs} epochs...")
        best_accuracy = 0.0

        # Create checkpoint directory if it doesn't exist
        os.makedirs(config.output_dir, exist_ok=True)

        for epoch in range(config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
            print("-" * 50)
            log_memory_usage()  # Track memory usage before each epoch

            train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, device, epoch)
            val_loss, val_accuracy = evaluate(model, val_loader, device)

            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"Train Loss: {train_loss:.4f} | Accuracy: {train_accuracy:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Accuracy: {val_accuracy:.4f}")

            # Save checkpoint after each epoch
            checkpoint_path = os.path.join(config.output_dir, f"checkpoint_epoch_{epoch+1}")
            model.save_pretrained(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                print(f"\nNew best accuracy: {best_accuracy:.4f}")
                print("Saving best model...")
                model.save_pretrained(f"{config.output_dir}/best")

            log_memory_usage()  # Track memory usage after each epoch

    except Exception as e:
        print(f"\nError during training process: {str(e)}")
        log_memory_usage()  # Log memory usage when error occurs
        raise e

    print("\nSaving final model...")
    model.save_pretrained(config.output_dir)
    print("Training completed!")
    log_memory_usage()  # Final memory usage log

if __name__ == "__main__":
    main()