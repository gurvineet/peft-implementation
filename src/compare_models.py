"""Script to compare base GPT-2 and PEFT-tuned model performance"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig
from datasets import load_dataset
from config import config
from tqdm import tqdm

def rating_to_stars(rating):
    """Convert numeric rating to star symbols"""
    return "★" * rating + "☆" * (5 - rating)

class ModelEvaluator:
    def __init__(self, model, tokenizer, device, model_name):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_name = model_name
        self.model.to(device)
        self.model.eval()

    def predict(self, texts):
        """Predict ratings for a list of texts"""
        predictions = []
        for text in tqdm(texts, desc=f"{self.model_name} predictions"):
            try:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=config.max_length,
                    padding="max_length"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    pred = torch.argmax(outputs.logits, dim=1).item()
                    predictions.append(pred)
            except Exception as e:
                print(f"Error predicting text: {str(e)}")
                predictions.append(None)

        return predictions

    def cleanup(self):
        """Clean up GPU memory"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

def setup_base_model():
    """Setup base GPT-2 model"""
    try:
        print("Loading base model weights...")
        model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model_name,
            num_labels=config.num_labels,
            id2label=config.id2label,
            label2id=config.label2id
        )
        print("Loading base model tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id

        return model, tokenizer
    except Exception as e:
        print(f"Error setting up base model: {str(e)}")
        raise

def setup_peft_model():
    """Setup PEFT-tuned model"""
    try:
        peft_model_path = "peft_model/best"
        print("Loading PEFT config...")
        peft_config = PeftConfig.from_pretrained(peft_model_path)

        print("Loading PEFT model weights...")
        model = AutoModelForSequenceClassification.from_pretrained(
            peft_config.base_model_name_or_path,
            num_labels=config.num_labels,
            id2label=config.id2label,
            label2id=config.label2id
        )
        model = PeftModel.from_pretrained(model, peft_model_path)

        print("Loading PEFT model tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id

        return model, tokenizer
    except Exception as e:
        print(f"Error setting up PEFT model: {str(e)}")
        raise

def main():
    print("\n=== Model Comparison: Base GPT-2 vs PEFT-tuned ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Load models
        base_model, base_tokenizer = setup_base_model()
        base_evaluator = ModelEvaluator(base_model, base_tokenizer, device, "Base GPT-2")

        peft_model, peft_tokenizer = setup_peft_model()
        peft_evaluator = ModelEvaluator(peft_model, peft_tokenizer, device, "PEFT-tuned")

        # Load test data (10 examples)
        print("\nLoading test dataset...")
        dataset = load_dataset("sealuzh/app_reviews", split="train[:10]")
        reviews = [item["review"] for item in dataset]
        actual_ratings = [min(5, max(0, item["star"])) for item in dataset]

        # Get predictions
        print("\nGenerating predictions...")
        base_predictions = base_evaluator.predict(reviews)
        base_evaluator.cleanup()  # Clean up base model

        peft_predictions = peft_evaluator.predict(reviews)
        peft_evaluator.cleanup()  # Clean up PEFT model

        # Calculate accuracy for valid predictions
        valid_predictions = [(true, base, peft) for true, base, peft in 
                           zip(actual_ratings, base_predictions, peft_predictions) 
                           if base is not None and peft is not None]

        if valid_predictions:
            true_ratings, base_preds, peft_preds = zip(*valid_predictions)

            base_correct = sum(1 for true, pred in zip(true_ratings, base_preds) if true == pred)
            peft_correct = sum(1 for true, pred in zip(true_ratings, peft_preds) if true == pred)

            base_accuracy = base_correct / len(true_ratings)
            peft_accuracy = peft_correct / len(true_ratings)

            # Print results
            print("\n=== Results ===")
            print(f"Number of test examples: {len(true_ratings)}")
            print(f"Base GPT-2 Accuracy: {base_accuracy:.2%}")
            print(f"PEFT-tuned Accuracy: {peft_accuracy:.2%}")
            print(f"Improvement: {(peft_accuracy - base_accuracy):.2%}")

            # Show detailed predictions
            print("\nDetailed Results:")
            print("-" * 80)

            for review, actual, base_pred, peft_pred in zip(reviews, actual_ratings, base_predictions, peft_predictions):
                if base_pred is not None and peft_pred is not None:
                    print(f"\nReview: {review[:100]}..." if len(review) > 100 else f"\nReview: {review}")
                    print(f"Actual rating:     {rating_to_stars(actual)} ({actual} stars)")
                    print(f"Base prediction:   {rating_to_stars(base_pred)} ({base_pred} stars)")
                    print(f"PEFT prediction:   {rating_to_stars(peft_pred)} ({peft_pred} stars)")
                    print("-" * 80)
        else:
            print("\nNo valid predictions were generated.")

    except Exception as e:
        print(f"\nError during model comparison: {str(e)}")
        raise
    finally:
        print("\nComparison completed.")

if __name__ == "__main__":
    main()