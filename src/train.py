"""Main training script for PEFT fine-tuning"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
from datasets import load_dataset
from config import config
from data_utils import prepare_dataloaders

def setup_peft_model():
    """Initialize the PEFT model with LoRA configuration"""
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        trust_remote_code=True,
        device_map="auto"
    )

    print("Creating LoRA configuration...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        # DeepSeek specific attention modules
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
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

    progress_bar = tqdm(enumerate(train_loader), 
                       total=total_batches,
                       desc=f"Epoch {epoch+1}", 
                       leave=True)

    for batch_idx, batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )

        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        # Update progress bar
        avg_loss = total_loss / (batch_idx + 1)
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "avg_loss": f"{avg_loss:.4f}",
            "processed": f"{(batch_idx+1)*config.batch_size}/{total_batches*config.batch_size}"
        })

        # Log every N steps
        if (batch_idx + 1) % config.logging_steps == 0:
            print(f"\nStep {batch_idx+1}/{total_batches}")
            print(f"Average Loss: {avg_loss:.4f}")

    return total_loss / total_batches

def evaluate(model, val_loader, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    progress_bar = tqdm(val_loader, desc="Evaluating", leave=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )

            loss = outputs.loss.item()
            total_loss += loss

            # Update progress
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({
                "loss": f"{loss:.4f}",
                "avg_loss": f"{avg_loss:.4f}"
            })

    return total_loss / len(val_loader)

def main():
    print("\n=== Starting PEFT Fine-tuning Process ===\n")

    # Load the SMS spam dataset
    print("Loading SMS spam dataset...")
    dataset = load_dataset("sms_spam", split="train").train_test_split(
        test_size=0.2, shuffle=True, seed=42
    )

    # Take a subset of the data for faster training
    MAX_SAMPLES = 200  # Reduced sample size for faster training
    print(f"\nUsing {MAX_SAMPLES} samples for training...")

    train_dataset = dataset["train"].shuffle(seed=42).select(range(min(MAX_SAMPLES, len(dataset["train"]))))
    val_dataset = dataset["test"].shuffle(seed=42).select(range(min(MAX_SAMPLES//5, len(dataset["test"]))))

    # Convert dataset to list of texts
    texts = [example["sms"] for example in train_dataset]
    print(f"Number of training examples: {len(texts)}")

    # Prepare data
    print("\nPreparing dataloaders...")
    train_loader, val_loader, tokenizer = prepare_dataloaders(texts, config)

    # Setup model
    print("\nSetting up model...")
    device = torch.device(config.device)
    model = setup_peft_model()
    model.to(device)

    # Setup optimizer with weight decay
    print("\nSetting up optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=0.01
    )

    # Training loop
    print(f"\nTraining for {config.num_epochs} epochs...")
    best_val_loss = float('inf')

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        print("-" * 50)

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)

        # Evaluate
        val_loss = evaluate(model, val_loader, device)

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = f"{config.output_dir}/checkpoint-best"
            print(f"\nSaving best model checkpoint to {checkpoint_path}")
            model.save_pretrained(checkpoint_path)

        # Save regular checkpoint
        if (epoch + 1) % 1 == 0:  # Save every epoch
            checkpoint_path = f"{config.output_dir}/checkpoint-{epoch}"
            print(f"Saving checkpoint to {checkpoint_path}")
            model.save_pretrained(checkpoint_path)

    # Save final model
    print("\nSaving final model...")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()