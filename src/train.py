"""Main training script for PEFT fine-tuning"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
from config import config
from data_utils import prepare_dataloaders

def setup_peft_model():
    """Initialize the PEFT model with LoRA configuration"""
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name)

    print("Creating LoRA configuration...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=["c_attn", "c_proj"]  # GPT-2 specific attention modules
    )

    print("Converting to PEFT model...")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model

def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="Training", leave=True)

    for batch_idx, batch in enumerate(progress_bar):
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
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "avg_loss": f"{total_loss / (batch_idx + 1):.4f}"
        })

    return total_loss / len(train_loader)

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
            progress_bar.set_postfix({
                "loss": f"{loss:.4f}",
                "avg_loss": f"{total_loss / (batch_idx + 1):.4f}"
            })

    return total_loss / len(val_loader)

def main():
    print("\n=== Starting PEFT Fine-tuning Process ===\n")

    # Extended sample texts for demonstration
    texts = [
        "Hello, my name is Sarah and I'm a software engineer.",
        "The weather today is sunny with clear blue skies.",
        "I love to program in Python because it's versatile and readable.",
        "The future of artificial intelligence looks promising and exciting.",
        "Machine learning models are becoming more efficient over time.",
        "Data science combines statistics, programming, and domain expertise.",
        "Neural networks can learn complex patterns in data.",
        "Deep learning has revolutionized natural language processing.",
        "Transfer learning allows us to leverage pre-trained models.",
        "Parameter efficient fine-tuning reduces computational costs.",
        "The key to good model performance is quality training data.",
        "GPT models have shown impressive capabilities in text generation.",
        "Responsible AI development considers ethical implications.",
        "Model optimization techniques help improve inference speed.",
        "The intersection of AI and healthcare shows great potential."
    ]

    print(f"Number of training examples: {len(texts)}")

    # Prepare data
    print("\nPreparing dataloaders...")
    train_loader, val_loader, tokenizer = prepare_dataloaders(texts, config)

    # Setup model
    print("\nSetting up model...")
    device = torch.device(config.device)
    model = setup_peft_model()
    model.to(device)

    # Setup optimizer
    print("\nSetting up optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Training loop
    print(f"\nTraining for {config.num_epochs} epochs...")
    best_val_loss = float('inf')

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        print("-" * 50)

        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)

        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        # Save checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = f"{config.output_dir}/checkpoint-best"
            print(f"\nSaving best model checkpoint to {checkpoint_path}")
            model.save_pretrained(checkpoint_path)

        # Save regular checkpoint
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