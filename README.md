1. Clone the repo
```
git clone https://github.com/gurvineet/peft-implementation.git
cd peft-implementation
```

2. Install dependencies:
```bash
pip install torch transformers peft datasets tqdm psutil
```

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

```
.
├── src/
│   ├── train.py          # Main training script
│   ├── inference.py      # Inference utilities
│   ├── compare_models.py # Model comparison tools
│   ├── data_utils.py     # Data handling utilities
│   ├── check_dataset.py  # Dataset inspection script
│   └── config.py         # Configuration settings
├── peft_model/          # Model checkpoints and configs
└── README.md
```

## Model Details

The implementation uses the following PEFT configuration:
- Base model: GPT-2
- PEFT method: LoRA (Low-Rank Adaptation)
- Task: Sequence Classification (Rating Prediction)
- Dataset: App Reviews Dataset (sealuzh/app_reviews)
- Training Parameters:
  - LoRA rank (r): 8
  - LoRA alpha: 32
  - Dropout: 0.1
  - Learning rate: Configured in config.py
  - Target modules: c_attn, c_proj (GPT-2 specific)

## Development Process

- Dataset preprocessing with robust error handling
- Memory-efficient training implementation
- Checkpoint saving after each epoch
- Performance tracking with loss and accuracy metrics
- Model evaluation on validation set

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{peft_implementation,
  title={Parameter-Efficient Fine-Tuning Implementation},
  year={2025},
  publisher={GitHub},
  url={https://github.com/gurvineet/peft-implementation}
}