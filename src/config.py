"""Configuration settings for PEFT fine-tuning"""

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Model settings
    base_model_name: str = "deepseek-ai/deepseek-coder-1.3b-base"  # Using a smaller base model
    max_length: int = 128  # Keep this length for good context

    # LoRA hyperparameters
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # Training hyperparameters
    batch_size: int = 8  # Increased for better throughput
    learning_rate: float = 2e-4  # Slightly increased for faster convergence
    num_epochs: int = 2
    warmup_steps: int = 20
    max_grad_norm: float = 0.5  # Keep this for stable training

    # Output settings
    output_dir: str = "peft_model"
    evaluation_strategy: str = "steps"

    # Device settings
    device: str = "cpu"  # Will be auto-detected in training

    # Evaluation settings
    eval_steps: int = 20
    save_steps: int = 20
    logging_steps: int = 5
    save_total_limit: int = 2

config = TrainingConfig()