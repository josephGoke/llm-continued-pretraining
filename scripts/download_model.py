"""
Utility script to download a model from Hugging Face Hub.
Useful for pre-downloading models before training starts.
Supports tokenizer and model downloading with error recovery.
"""

import argparse
import logging
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_model(
    model_name: str,
    save_dir: str = "./models",
    save_tokenizer_only: bool = False,
    trust_remote_code: bool = False,
):
    """
    Download model and tokenizer from HF Hub.
    
    Args:
        model_name: Model ID on HF Hub (e.g., "meta-llama/Llama-3.2-3b")
        save_dir: Directory to save model
        save_tokenizer_only: If True, only download tokenizer (skip model)
        trust_remote_code: Allow custom modeling code from HF Hub
    """
    save_path = Path(save_dir) / model_name.split('/')[-1]
    save_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading from HF Hub: {model_name}")
    logger.info(f"Save location: {save_path}")
    
    # Download tokenizer
    try:
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        tokenizer.save_pretrained(str(save_path))
        logger.info(f"✓ Tokenizer saved to {save_path}")
    except Exception as e:
        logger.error(f"✗ Failed to download tokenizer: {e}")
        raise
    
    # Download model (optional)
    if save_tokenizer_only:
        logger.info("Skipping model download (tokenizer only mode)")
        return
    
    try:
        logger.info("Downloading model (this may take a few minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype="auto",  # Infer dtype from model
        )
        model.save_pretrained(str(save_path))
        logger.info(f"✓ Model saved to {save_path}")
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✓ Model size: {total_params / 1e9:.2f}B parameters")
        
    except Exception as e:
        logger.error(f"✗ Failed to download model: {e}")
        logger.info("Tip: Model download failed, but tokenizer was saved successfully")
        logger.info(f"You can still use the tokenizer from: {save_path}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download model and tokenizer from Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download model + tokenizer
  python download_model.py --model meta-llama/Llama-3.2-3b
  
  # Download only tokenizer (faster, for testing)
  python download_model.py --model meta-llama/Llama-3.2-3b --tokenizer_only
  
  # Custom save directory
  python download_model.py --model meta-llama/Llama-3.2-3b --save_dir /path/to/models
  
  # Trust remote code (for custom models)
  python download_model.py --model meta-llama/Llama-3.2-3b --trust_remote_code
        """
    )
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
    parser.add_argument(
        "--tokenizer_only",
        action="store_true",
        help="Only download tokenizer (skip model)",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow custom modeling code from HF Hub",
    )
    
    args = parser.parse_args()
    
    try:
        download_model(
            model_name=args.model,
            save_dir=args.save_dir,
            save_tokenizer_only=args.tokenizer_only,
            trust_remote_code=args.trust_remote_code,
        )
        logger.info("\n✓ Download successful!")
    except Exception as e:
        logger.error(f"\n✗ Download failed: {e}")
        exit(1)