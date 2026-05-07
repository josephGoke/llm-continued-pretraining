"""
Utility script to download a model from Hugging Face Hub.
Useful for pre-downloading models before training starts.
"""

import argparse
import logging
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_model(model_name: str, save_dir: str = "./models"):
    """
    Download model and tokenizer from HF Hub.
    
    Args:
        model_name: Model ID on HF Hub (e.g., "meta-llama/Llama-2-7b")
        save_dir: Directory to save model
    """
    logger.info(f"Downloading model: {model_name}")
    
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model.save_pretrained(f"{save_dir}/{model_name.split('/')[-1]}")
    tokenizer.save_pretrained(f"{save_dir}/{model_name.split('/')[-1]}")
    
    logger.info(f"Model saved to {save_dir}/{model_name.split('/')[-1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download model from Hugging Face Hub")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-3b",
        help="Model name/ID on HF Hub",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./models",
        help="Directory to save model",
    )
    
    args = parser.parse_args()
    download_model(args.model, args.save_dir)
