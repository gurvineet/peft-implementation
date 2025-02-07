"""Configuration settings for PEFT fine-tuning"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import torch

@dataclass
class TrainingConfig:
    # Model settings
    base_model_name: str = "gpt2"
    max_length: int = 128
    model_max_length: int = 128
    padding_side: str = "right"

    # Dataset settings
    dataset_name: str = "sealuzh/app_reviews"
    num_samples: int = 2000
    train_split: float = 0.8
    random_seed: int = 42
    cache_dir: str = ".cache"

    # LoRA hyperparameters
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: list = field(default_factory=lambda: ["c_attn", "c_proj"])

    # Training hyperparameters - Optimized values
    batch_size: int = 8  # Reduced from 16 to improve stability
    learning_rate: float = 1e-4  # Reduced from 2e-4 for better stability
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0  # Increased for better gradient stability
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 2  # Added to maintain effective batch size

    # Output settings
    output_dir: str = "peft_model"
    best_model_dir: str = "peft_model/best"
    checkpoint_dir: str = "peft_model/checkpoints"

    # Training settings
    eval_steps: int = 50
    save_steps: int = 50
    logging_steps: int = 10
    save_total_limit: int = 2

    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = False

    # Classification settings
    num_labels: int = 6  # Ratings from 0 to 5
    id2label: Dict[int, str] = field(
        default_factory=lambda: {
            i: f"RATING_{i}"
            for i in range(6)
        }
    )
    label2id: Dict[str, int] = field(
        default_factory=lambda: {
            f"RATING_{i}": i
            for i in range(6)
        }
    )

    # Logging settings
    log_level: str = "info"
    disable_tqdm: bool = False

    def __post_init__(self):
        """Validate and set derived configurations"""
        # Ensure output directories are properly set
        self.best_model_dir = f"{self.output_dir}/best"
        self.checkpoint_dir = f"{self.output_dir}/checkpoints"

        # Calculate steps based on dataset size and batch size
        train_samples = int(self.num_samples * self.train_split)
        steps_per_epoch = int(train_samples / (self.batch_size * self.gradient_accumulation_steps))
        self.total_steps = steps_per_epoch * self.num_epochs
        self.warmup_steps = int(self.total_steps * self.warmup_ratio)

        # Set logging format
        self.logging_format = (
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        )

config = TrainingConfig()