"""
Checkpoint recovery and resuming script.
Automatically finds the latest checkpoint and resumes training from there.
"""

import argparse
import logging
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_latest_checkpoint(output_dir: str) -> Path:
    """Find the latest checkpoint in output directory."""
    output_path = Path(output_dir)
    
    checkpoints = sorted(
        [d for d in output_path.glob("checkpoint-*") if d.is_dir()],
        key=lambda x: int(x.name.split("-")[-1]) if x.name.split("-")[-1].isdigit() else 0,
    )
    
    if not checkpoints:
        logger.error(f"No checkpoints found in {output_dir}")
        return None
    
    latest = checkpoints[-1]
    logger.info(f"Found latest checkpoint: {latest}")
    return latest


def update_config_for_resume(config_path: str, checkpoint_path: str):
    """Update config to resume from checkpoint."""
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config["checkpointing"]["resume_from_checkpoint"] = str(checkpoint_path)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    logger.info(f"Updated config to resume from: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Recover and resume training from checkpoint")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/checkpoint",
        help="Output directory where checkpoints are saved",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config/training_config.yaml",
        help="Path to training config",
    )
    parser.add_argument(
        "--list_only",
        action="store_true",
        help="List all checkpoints without updating config",
    )
    
    args = parser.parse_args()
    
    # Find checkpoints
    output_path = Path(args.output_dir)
    checkpoints = sorted(
        [d for d in output_path.glob("checkpoint-*") if d.is_dir()],
        key=lambda x: int(x.name.split("-")[-1]) if x.name.split("-")[-1].isdigit() else 0,
    )
    
    if not checkpoints:
        logger.error(f"No checkpoints found in {args.output_dir}")
        return
    
    logger.info(f"\n=== Available Checkpoints ===")
    for i, ckpt in enumerate(checkpoints, 1):
        step = ckpt.name.split("-")[-1]
        logger.info(f"{i}. {ckpt.name} (Step {step})")
    
    if args.list_only:
        return
    
    # Use latest checkpoint
    latest = checkpoints[-1]
    logger.info(f"\n✓ Using latest checkpoint: {latest}")
    
    # Update config
    update_config_for_resume(args.config, str(latest))
    
    logger.info("\nTo resume training, run:")
    logger.info(f"  python train.py --config {args.config}")


if __name__ == "__main__":
    main()
