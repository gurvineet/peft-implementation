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
    dataset_name: str = "app_reviews.csv"
    num_samples: Optional[int] = None  # Use full dataset
    train_split: float = 0.8
    random_seed: int = 42
    cache_dir: str = ".cache"

    # LoRA hyperparameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: list = field(default_factory=lambda: ["c_attn", "c_proj"])

    # Training hyperparameters - Optimized for better convergence
    batch_size: int = 4  # Reduced to create more steps per epoch
    learning_rate: float = 2e-4  # Increased for faster initial learning
    num_epochs: int = 10  # Increased for better convergence
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0  # Increased for more stable updates
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 2  # Reduced to update more frequently

    # Output settings
    output_dir: str = "peft_model"
    best_model_dir: str = "peft_model/best"
    checkpoint_dir: str = "peft_model/checkpoints"

    # Training settings
    eval_steps: int = 20  # More frequent evaluation
    save_steps: int = 20  # More frequent saving
    logging_steps: int = 10  # More frequent logging
    save_total_limit: int = 3

    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = False

    # Classification settings
    num_labels: int = 5  # Ratings from 1 to 5
    id2label: Dict[int, str] = field(
        default_factory=lambda: {
            i + 1: f"RATING_{i + 1}"
            for i in range(5)
        }
    )
    label2id: Dict[str, int] = field(
        default_factory=lambda: {
            f"RATING_{i + 1}": i + 1
            for i in range(5)
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
        if self.num_samples:
            train_samples = int(self.num_samples * self.train_split)
        else:
            # Estimate based on file size or use a default
            train_samples = 1000  # Will be updated after dataset load

        steps_per_epoch = max(1, int(train_samples / (self.batch_size * self.gradient_accumulation_steps)))
        self.total_steps = steps_per_epoch * self.num_epochs
        self.warmup_steps = int(self.total_steps * self.warmup_ratio)

        # Set logging format
        self.logging_format = (
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        )

config = TrainingConfig()