"""Inference script for using the fine-tuned model"""

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from config import config

class PEFTInference:
    def __init__(self, model_path):
        self.device = torch.device(config.device)
        self.model = AutoPeftModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        self.model.to(self.device)
        self.model.eval()
    
    def generate(self, prompt, max_new_tokens=50, temperature=0.7):
        """Generate text from a prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def batch_generate(self, prompts, max_new_tokens=50, temperature=0.7):
        """Generate text for multiple prompts"""
        results = []
        for prompt in prompts:
            generated_text = self.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            results.append(generated_text)
        return results

def main():
    # Initialize inference
    inference = PEFTInference(config.output_dir)
    
    # Example prompts
    test_prompts = [
        "Hello, my name is",
        "The future of AI is",
        "Once upon a time",
        "The most interesting thing about technology is"
    ]
    
    # Single prompt generation
    print("\nSingle prompt generation:")
    output = inference.generate(test_prompts[0])
    print(f"Prompt: {test_prompts[0]}")
    print(f"Generated: {output}")
    
    # Batch generation
    print("\nBatch generation:")
    outputs = inference.batch_generate(test_prompts)
    for prompt, output in zip(test_prompts, outputs):
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {output}")

if __name__ == "__main__":
    main()
