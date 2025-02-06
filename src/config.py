"""Configuration settings for PEFT fine-tuning"""

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Model settings
    base_model_name: str = "gpt2"
    max_length: int = 64  # Reduced for CPU training

    # LoRA hyperparameters
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # Training hyperparameters
    batch_size: int = 4  # Reduced for CPU training
    learning_rate: float = 1e-4  # Reduced for stability
    num_epochs: int = 3
    warmup_steps: int = 50
    max_grad_norm: float = 1.0  # Added gradient clipping

    # Output settings
    output_dir: str = "peft_model"
    evaluation_strategy: str = "epoch"

    # Device settings
    device: str = "cpu"

    # Evaluation settings
    eval_steps: int = 200
    save_steps: int = 200
    logging_steps: int = 50

config = TrainingConfig()