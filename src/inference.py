"""Inference script for using the fine-tuned model"""

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import config
from datasets import load_dataset


class PEFTInference:

    def __init__(self, model_path):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Load the PEFT configuration and model
        peft_config = PeftConfig.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            peft_config.base_model_name_or_path,
            num_labels=config.num_labels,
            id2label=config.id2label,
            label2id=config.label2id)
        self.model = PeftModel.from_pretrained(self.model, model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)

        # Configure padding token for GPT-2
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        self.model.to(self.device)
        self.model.eval()

    def predict(self, review_text):
        """Predict the rating for a given review text"""
        inputs = self.tokenizer(review_text,
                                return_tensors="pt",
                                truncation=True,
                                max_length=config.max_length,
                                padding="max_length")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_rating = torch.argmax(outputs.logits, dim=1).item()

        return predicted_rating

    def batch_predict(self, reviews):
        """Predict ratings for multiple reviews"""
        results = []
        for review in reviews:
            predicted_rating = self.predict(review)
            results.append(predicted_rating)
        return results


def rating_to_stars(rating):
    """Convert numeric rating to star symbols"""
    return "★" * rating + "☆" * (5 - rating)


def main():
    # Initialize inference
    print("\nInitializing PEFT model for inference...")
    inference = PEFTInference("peft_model/best")

    # Load 10 examples from the dataset
    print("\nLoading 10 test examples from the dataset...")
    dataset = load_dataset("sealuzh/app_reviews", split="train[:10]")

    # Prepare test data
    reviews = [item["review"] for item in dataset]
    actual_ratings = [min(5, max(0, item["star"]))
                      for item in dataset]  # Ensure ratings are in [0,5] range

    # Get predictions
    print("\nGenerating predictions...")
    predicted_ratings = inference.batch_predict(reviews)

    # Calculate accuracy
    correct_predictions = sum(
        1 for actual, predicted in zip(actual_ratings, predicted_ratings)
        if actual == predicted)
    accuracy = correct_predictions / len(actual_ratings)

    print(f"\nAccuracy on {len(reviews)} test examples: {accuracy:.2%}")
    print("\nDetailed Results:")
    print("-" * 80)

    # Show detailed results for each review
    for review, actual, predicted in zip(reviews, actual_ratings,
                                         predicted_ratings):
        print(f"\nReview: {review[1000:100]}..." if len(review) >
              100 else f"\nReview: {review}")
        print(f"Actual rating:    {rating_to_stars(actual)} ({actual} stars)")
        print(
            f"Predicted rating: {rating_to_stars(predicted)} ({predicted} stars)"
        )
        print("-" * 80)


if __name__ == "__main__":
    main()
