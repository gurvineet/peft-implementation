"""Configuration settings for PEFT fine-tuning"""

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Model settings
    base_model_name: str = "deepseek-ai/deepseek-coder-350m-base"
    max_length: int = 32  # Further reduced for faster processing

    # LoRA hyperparameters
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # Training hyperparameters
    batch_size: int = 8  # Increased batch size for smaller model
    learning_rate: float = 1e-4  # Adjusted for DeepSeek model
    num_epochs: int = 2
    warmup_steps: int = 20
    max_grad_norm: float = 1.0

    # Output settings
    output_dir: str = "peft_model"
    evaluation_strategy: str = "steps"

    # Device settings
    device: str = "cpu"

    # Evaluation settings
    eval_steps: int = 20
    save_steps: int = 20
    logging_steps: int = 5
    save_total_limit: int = 2

config = TrainingConfig()