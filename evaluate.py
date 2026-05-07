"""
Evaluation script for the trained model.
Computes metrics like perplexity and loss on validation set.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from data.data_loader import JSONLDataset
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_perplexity(model, data_loader, device, max_batches: Optional[int] = None):
    """
    Compute perplexity on validation dataset.
    
    Args:
        model: Language model
        data_loader: Validation data loader
        device: Device to run on
        max_batches: Max batches to evaluate (None = all)
    
    Returns:
        Dictionary with metrics: perplexity, loss, num_tokens
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating")):
            if max_batches and batch_idx >= max_batches:
                break
            
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            # Accumulate loss
            batch_size = input_ids.shape[0]
            total_loss += loss.item() * batch_size
            total_tokens += batch_size * input_ids.shape[1]
    
    avg_loss = total_loss / max(total_tokens / 2048, 1)  # Normalize by seq length
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "num_tokens": total_tokens,
    }


def evaluate(
    model_path: str,
    validation_data_path: str,
    output_file: Optional[str] = None,
    batch_size: int = 8,
    max_seq_length: int = 2048,
    field_name: str = "text",
):
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to saved model
        validation_data_path: Path to validation JSONL file
        output_file: Optional file to save results
        batch_size: Batch size
        max_seq_length: Max sequence length
        field_name: JSON field for text
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model and tokenizer
    logger.info(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.to(device)
    model.eval()
    
    # Load validation data
    logger.info(f"Loading validation data from {validation_data_path}")
    val_dataset = JSONLDataset(
        file_path=validation_data_path,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        field_name=field_name,
    )
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Evaluate
    logger.info("Computing metrics...")
    metrics = compute_perplexity(model, val_loader, device)
    
    logger.info("\n=== Evaluation Results ===")
    logger.info(f"Loss: {metrics['loss']:.4f}")
    logger.info(f"Perplexity: {metrics['perplexity']:.2f}")
    logger.info(f"Total tokens evaluated: {metrics['num_tokens']:,}")
    
    # Save results if requested
    if output_file:
        import json
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate continued pretraining model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--validation_data",
        type=str,
        required=True,
        help="Path to validation JSONL file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save evaluation results JSON",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    
    args = parser.parse_args()
    
    evaluate(
        model_path=args.model_path,
        validation_data_path=args.validation_data,
        output_file=args.output_file,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
    )
