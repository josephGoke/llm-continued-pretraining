"""
Inference script for the trained model.
Simple wrapper for generation from the continued pretrained model.
"""

import argparse
import logging
from typing import Optional, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PretrainedModelInference:
    """Wrapper for inference with continued pretrained model."""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize inference wrapper.
        
        Args:
            model_path: Path to model checkpoint
            device: Device to run on (auto-detect if None)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = torch.device(device)
        logger.info(f"Using device: {self.device}")
        
        logger.info(f"Loading model from {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto" if "cuda" in device else None,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
    
    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
        repetition_penalty: float = 1.2,
    ) -> List[str]:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            num_return_sequences: Number of sequences to generate
            repetition_penalty: Penalty for repeating tokens
        
        Returns:
            List of generated sequences
        """
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            output_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            outputs = self.tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
            )
        
        return outputs
    
    def generate_batch(
        self,
        prompts: List[str],
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
    ) -> List[List[str]]:
        """Generate text for multiple prompts."""
        results = []
        for prompt in prompts:
            result = self.generate(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            results.append(result)
        return results


def main():
    """Interactive generation interface."""
    parser = argparse.ArgumentParser(description="Inference with continued pretrained model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda or cpu)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt to generate from (if not provided, enters interactive mode)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum generation length",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter",
    )
    
    args = parser.parse_args()
    
    # Initialize model
    model = PretrainedModelInference(args.model_path, device=args.device)
    
    if args.prompt:
        # Single prompt mode
        outputs = model.generate(
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print("\n=== Generated Output ===")
        for i, output in enumerate(outputs):
            print(f"\nOutput {i+1}:\n{output}")
    else:
        # Interactive mode
        print("\n=== Interactive Generation ===")
        print("Enter a prompt (or 'quit' to exit):")
        
        while True:
            prompt = input("\nPrompt: ").strip()
            if prompt.lower() == "quit":
                break
            
            if not prompt:
                continue
            
            outputs = model.generate(
                prompt=prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            
            print("\nGenerated:")
            print(outputs[0])


if __name__ == "__main__":
    main()
