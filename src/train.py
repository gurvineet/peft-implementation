"""Main training script for PEFT fine-tuning"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
from datasets import load_dataset
from config import config
from data_utils import prepare_dataloaders

def setup_peft_model():
    """Initialize the PEFT model with LoRA configuration"""
    print("Loading base model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_name,
        num_labels=config.num_labels,
        id2label=config.id2label,
        label2id=config.label2id,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float32,  # Use float32 for CPU training
    )

    # Configure padding token
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    print("Creating LoRA configuration...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # Changed to sequence classification
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=["c_proj", "c_attn", "q_proj", "v_proj"]  # Updated target modules for GPT-2
    )

    print("Converting to PEFT model...")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model

def train_epoch(model, train_loader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_batches = len(train_loader)
    correct = 0
    total = 0

    progress_bar = tqdm(enumerate(train_loader), 
                       total=total_batches,
                       desc=f"Epoch {epoch+1}", 
                       leave=True)

    for batch_idx, batch in progress_bar:
        try:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            total_loss += loss.item()

            # Calculate training accuracy
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            current_accuracy = correct / total

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{avg_loss:.4f}",
                "accuracy": f"{current_accuracy:.4f}",
                "batch": f"{batch_idx+1}/{total_batches}"
            })

            if (batch_idx + 1) % config.logging_steps == 0:
                print(f"\nStep {batch_idx+1}/{total_batches}")
                print(f"Average Loss: {avg_loss:.4f}")
                print(f"Training Accuracy: {current_accuracy:.4f}")

        except RuntimeError as e:
            print(f"Error in batch {batch_idx}: {str(e)}")
            continue

    epoch_accuracy = correct / total
    print(f"\nEpoch {epoch+1} Training Accuracy: {epoch_accuracy:.4f}")
    return total_loss / total_batches, epoch_accuracy

def evaluate(model, val_loader, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    progress_bar = tqdm(val_loader, desc="Evaluating", leave=True)

    with torch.no_grad():
        for batch in progress_bar:
            try:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = outputs.loss.item()
                total_loss += loss

                # Calculate accuracy
                predictions = torch.argmax(outputs.logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                accuracy = correct / total
                progress_bar.set_postfix({
                    "loss": f"{loss:.4f}",
                    "accuracy": f"{accuracy:.4f}"
                })

            except RuntimeError as e:
                print(f"Error during evaluation: {str(e)}")
                continue

    return total_loss / len(val_loader), accuracy

def main():
    print("\n=== Starting PEFT Fine-tuning Process ===\n")

    # Load the SMS spam dataset
    print("Loading SMS spam dataset...")
    dataset = load_dataset("sms_spam", split="train").train_test_split(
        test_size=0.2, shuffle=True, seed=42
    )

    # Take a subset of the data for faster training
    MAX_SAMPLES = 100  # Reduced for initial testing
    print(f"\nUsing {MAX_SAMPLES} samples for training...")

    train_dataset = dataset["train"].shuffle(seed=42).select(range(min(MAX_SAMPLES, len(dataset["train"]))))
    val_dataset = dataset["test"].shuffle(seed=42).select(range(min(MAX_SAMPLES//5, len(dataset["test"]))))

    # Prepare data
    print("\nPreparing dataloaders...")
    train_loader, val_loader, tokenizer = prepare_dataloaders(train_dataset, val_dataset, config)

    # Setup model
    print("\nSetting up model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = setup_peft_model()
    model.to(device)

    # Setup optimizer
    print("\nSetting up optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01,
        eps=1e-7
    )

    # Training loop
    print(f"\nTraining for {config.num_epochs} epochs...")
    best_val_loss = float('inf')
    best_accuracy = 0.0

    try:
        for epoch in range(config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
            print("-" * 50)

            # Train
            train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, device, epoch)

            # Evaluate
            val_loss, val_accuracy = evaluate(model, val_loader, device)

            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train Accuracy: {train_accuracy:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}")

            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                checkpoint_path = f"{config.output_dir}/checkpoint-best"
                print(f"\nSaving best model checkpoint to {checkpoint_path}")
                print(f"New best accuracy: {best_accuracy:.4f}")
                model.save_pretrained(checkpoint_path)

            # Save regular checkpoint
            if (epoch + 1) % 1 == 0:
                checkpoint_path = f"{config.output_dir}/checkpoint-{epoch}"
                print(f"Saving checkpoint to {checkpoint_path}")
                model.save_pretrained(checkpoint_path)

    except Exception as e:
        print(f"Training interrupted: {str(e)}")
        # Save the model even if training was interrupted
        model.save_pretrained(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)
        raise e

    # Save final model
    print("\nSaving final model...")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()