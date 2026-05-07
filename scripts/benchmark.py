"""
Utility to benchmark inference speed of the model.
"""

import argparse
import logging
import time
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def benchmark_inference(
    model_path: str,
    batch_sizes: List[int] = [1, 4, 8],
    sequence_lengths: List[int] = [128, 512, 2048],
    num_runs: int = 10,
):
    """
    Benchmark model inference across different batch sizes and sequence lengths.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.to(device)
    model.eval()
    
    # Benchmark
    logger.info("\n=== Inference Benchmark ===\n")
    
    results = []
    
    for seq_len in sequence_lengths:
        for batch_size in batch_sizes:
            # Create dummy input
            dummy_input = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len)).to(device)
            
            # Warm up
            with torch.no_grad():
                _ = model(dummy_input)
            
            torch.cuda.synchronize() if "cuda" in device.type else None
            
            # Benchmark
            start = time.time()
            with torch.no_grad():
                for _ in range(num_runs):
                    _ = model(dummy_input)
            torch.cuda.synchronize() if "cuda" in device.type else None
            elapsed = time.time() - start
            
            avg_time = elapsed / num_runs
            tokens_per_sec = (batch_size * seq_len) / avg_time
            
            logger.info(
                f"Batch={batch_size}, SeqLen={seq_len}: "
                f"{avg_time*1000:.2f}ms/run, {tokens_per_sec:.0f} tokens/sec"
            )
            
            results.append({
                "batch_size": batch_size,
                "seq_len": seq_len,
                "time_ms": avg_time * 1000,
                "tokens_per_sec": tokens_per_sec,
            })
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark model inference")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=[1, 4, 8],
        help="Batch sizes to benchmark",
    )
    parser.add_argument(
        "--seq_lengths",
        type=int,
        nargs="+",
        default=[128, 512, 2048],
        help="Sequence lengths to benchmark",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=10,
        help="Number of runs per configuration",
    )
    
    args = parser.parse_args()
    
    benchmark_inference(
        model_path=args.model_path,
        batch_sizes=args.batch_sizes,
        sequence_lengths=args.seq_lengths,
        num_runs=args.num_runs,
    )
