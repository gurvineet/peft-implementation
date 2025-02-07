"""Configuration settings for PEFT fine-tuning"""

from dataclasses import dataclass, field
from typing import Dict

@dataclass
class TrainingConfig:
    # Model settings
    base_model_name: str = "gpt2"  # Using GPT-2 as base model
    max_length: int = 128  # Keep this length for good context

    # LoRA hyperparameters
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # Training hyperparameters
    batch_size: int = 8
    learning_rate: float = 2e-4
    num_epochs: int = 2
    warmup_steps: int = 20
    max_grad_norm: float = 0.5

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

    # Classification settings
    num_labels: int = 2  # Binary classification for spam detection
    id2label: Dict[str, str] = field(default_factory=lambda: {"0": "NOT_SPAM", "1": "SPAM"})
    label2id: Dict[str, int] = field(default_factory=lambda: {"NOT_SPAM": 0, "SPAM": 1})

config = TrainingConfig()