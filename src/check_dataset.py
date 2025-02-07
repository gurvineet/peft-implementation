from datasets import load_dataset

# Load a small sample of the dataset
dataset = load_dataset("sealuzh/app_reviews", split="train[:10]")

# Print the first example to see the structure
print("\nDataset features:", dataset.features)
print("\nFirst example:")
for key, value in dataset[0].items():
    print(f"{key}: {value}")
