"""Configuration settings for PEFT fine-tuning"""

from dataclasses import dataclass, field
from typing import Dict

@dataclass
class TrainingConfig:
    # Model settings
    base_model_name: str = "gpt2"  # Using GPT-2 as base model
    max_length: int = 128  # App reviews are typically shorter than movie reviews

    # LoRA hyperparameters
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # Training hyperparameters
    batch_size: int = 8  # Can use larger batch size due to shorter sequences
    learning_rate: float = 2e-4
    num_epochs: int = 3
    warmup_steps: int = 50
    max_grad_norm: float = 0.5

    # Output settings
    output_dir: str = "peft_model"
    evaluation_strategy: str = "steps"

    # Device settings
    device: str = "cpu"  # Will be auto-detected in training

    # Evaluation settings
    eval_steps: int = 50
    save_steps: int = 50
    logging_steps: int = 10
    save_total_limit: int = 2

    # Classification settings
    num_labels: int = 6  # Ratings from 0 to 5
    id2label: Dict[int, str] = field(
        default_factory=lambda: {
            0: "RATING_0",
            1: "RATING_1", 
            2: "RATING_2",
            3: "RATING_3",
            4: "RATING_4",
            5: "RATING_5"
        }
    )
    label2id: Dict[str, int] = field(
        default_factory=lambda: {
            "RATING_0": 0,
            "RATING_1": 1,
            "RATING_2": 2,
            "RATING_3": 3,
            "RATING_4": 4,
            "RATING_5": 5
        }
    )

config = TrainingConfig()