"""Evaluation utilities for model comparison"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
from config import config


def load_models():
    """Load both original and fine-tuned models for comparison"""
    # Load original model
    original_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)

    # Load fine-tuned PEFT model
    peft_model = AutoPeftModelForCausalLM.from_pretrained(config.output_dir)

    return original_model, peft_model, tokenizer


def generate_text(model, tokenizer, prompt, max_new_tokens=50):
    """Generate text from a given prompt"""
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(input_ids=inputs["input_ids"],
                                 attention_mask=inputs["attention_mask"],
                                 max_new_tokens=max_new_tokens,
                                 temperature=0,
                                 num_return_sequences=1,
                                 pad_token_id=tokenizer.eos_token_id)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def compare_models(test_prompts):
    """Compare original and fine-tuned models on test prompts"""
    original_model, peft_model, tokenizer = load_models()

    print("Model Comparison Results:")
    print("-" * 50)

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        print("\nOriginal model output:")
        original_output = generate_text(original_model, tokenizer, prompt)
        print(original_output)

        print("\nFine-tuned model output:")
        peft_output = generate_text(peft_model, tokenizer, prompt)
        print(peft_output)
        print("-" * 50)


def main():
    test_prompts = [
        "Hello, my name is", "The weather today is", "I love to program in",
        "The best thing about AI is"
    ]

    compare_models(test_prompts)


if __name__ == "__main__":
    main()
