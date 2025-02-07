# Parameter-Efficient Fine-Tuning Implementation

This repository contains an implementation of parameter-efficient fine-tuning (PEFT) using the Hugging Face PEFT library for adapting the GPT-2 model. The implementation focuses on fine-tuning GPT-2 for app review rating prediction using the LoRA (Low-Rank Adaptation) technique.

## Features

- Parameter-efficient fine-tuning using LoRA
- Integration with Hugging Face's PEFT library
- App review rating prediction (1-5 stars)
- Model comparison utilities (base GPT-2 vs PEFT-tuned)
- Memory-efficient training process
- Progress tracking and checkpointing

## Requirements

- Python 3.x
- PyTorch
- Transformers
- PEFT
- Datasets
- Tqdm

## Usage

1. Check the dataset:
```bash
python src/check_dataset.py
```

2. Train the model:
```bash
python src/train.py
```

3. Run inference:
```bash
python src/inference.py
```

4. Compare models:
```bash
python src/compare_models.py
```

## Project Structure

- `src/`
  - `train.py`: Main training script
  - `inference.py`: Inference utilities
  - `compare_models.py`: Model comparison tools
  - `data_utils.py`: Data handling utilities
  - `check_dataset.py`: Dataset inspection script
  - `config.py`: Configuration settings

## Model Details

The implementation uses the following PEFT configuration:
- Base model: GPT-2
- PEFT method: LoRA (Low-Rank Adaptation)
- Task: Sequence Classification (Rating Prediction)
- Dataset: App Reviews

## License

MIT License
