"""
Main training script for continued pretraining with FSDP.
Supports distributed training across multiple GPUs/nodes.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional
from datetime import datetime

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, CPUOffload, BackwardPrefetch
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
import yaml
from tqdm import tqdm
import functools

import wandb

try:
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from data.data_loader import create_data_loaders

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load training config from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_distributed():
    """Initialize distributed training."""
    if not torch.distributed.is_available():
        logger.warning("Distributed training not available, running single GPU")
        return False
    
    if torch.cuda.is_available():
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        logger.info(f"Initialized NCCL backend, rank {dist.get_rank()}")
        return True
    return False


def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    """Find the latest checkpoint in output directory."""
    checkpoints = sorted(
        [d for d in output_dir.glob("checkpoint-*") if d.is_dir()],
        key=lambda x: int(x.name.split("-")[-1]),
    )
    return checkpoints[-1] if checkpoints else None


def setup_wandb(config: dict, is_main_process: bool):
    """Initialize Weights & Biases logging."""
    if not is_main_process or not HAS_WANDB:
        return
    
    wandb_cfg = config.get("logging", {})
    if not wandb_cfg.get("use_wandb", False):
        return
    
    wandb.init(
        project=wandb_cfg.get("wandb_project", "llm-pretraining"),
        entity=wandb_cfg.get("wandb_entity"),
        config={
            "model": config.get("model", {}),
            "training": config.get("training", {}),
        },
        name=f"pretraining-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    logger.info("✓ Weights & Biases initialized")


def get_model_layer_class(model):
    """
    Detect and return the transformer layer class for the model.
    Supports Llama, Mistral, Qwen, GPT-2, Bloom, Falcon, OPT, etc.
    Useful for advanced FSDP configurations.
    """
    model_type = model.config.model_type.lower() if hasattr(model.config, 'model_type') else ""
    
    # Map model types to their decoder/transformer layer classes
    layer_class_map = {
        "llama": "LlamaDecoderLayer",
        "mistral": "MistralDecoderLayer",
        "qwen": "QWenBlock",
        "gpt2": "GPT2Block",
        "gpt-2": "GPT2Block",
        "bloom": "BloomBlock",
        "falcon": "FalconDecoderLayer",
        "opt": "OPTDecoderLayer",
    }
    
    try:
        if model_type in layer_class_map:
            layer_name = layer_class_map[model_type]
            logger.debug(f"Detected model type: {model_type}, layer class: {layer_name}")
            return layer_name
    except Exception as e:
        logger.debug(f"Could not detect layer class: {e}")
    
    return None


def setup_fsdp_model(model, config):
    """
    Wrap model with FSDP for distributed training.
    Uses layer-wise auto-wrap policy for efficient communication and compute overlap.
    """
    if not dist.is_available() or not dist.is_initialized():
        logger.warning("Distributed not initialized, skipping FSDP wrapping")
        return model
    
    fsdp_cfg = config.get("distributed", {}).get("fsdp_config", {})
    
    # Layer-wise auto-wrap policy: wraps each transformer block independently
    # Benefit: better gradient accumulation, reduced comm overhead, compute/comm overlap
    auto_wrap_policy = functools.partial(
        lambda_auto_wrap_policy,
        excluded_modules={torch.nn.Embedding},  #(small, frequent syncs)
    )
    
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=CPUOffload(offload_params=fsdp_cfg.get("cpu_offload", False)),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=fsdp_cfg.get("limit_all_gathers", True),
        sync_module_states=True,
        device_id=torch.cuda.current_device(),
    )
    
    logger.info("✔ Model wrapped with FSDP (layer-wise auto-wrap policy enabled)")
    return model


def train(
    config_path: str = "./config/training_config.yaml",
    local_rank: int = 0,
    world_size: int = 1,
):
    """Main training loop."""
    
    # Load config
    config = load_config(config_path)
    
    # Setup distributed training
    use_distributed = world_size > 1
    if use_distributed:
        setup_distributed()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
    
    is_main_process = rank == 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if is_main_process:
        logger.info(f"Using device: {device}")
        logger.info(f"Distributed training: {use_distributed} (rank {rank}/{world_size})")
    
    # Create output directory
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    if is_main_process:
        log_file = output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
            )
        )
        logger.addHandler(file_handler)
        logger.info(f"Saving logs to {log_file}")
    
    # Setup W&B monitoring
    setup_wandb(config, is_main_process)
    
    # Load model and tokenizer
    if is_main_process:
        logger.info(f"Loading model: {config['model']['name']}")
    
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        torch_dtype=torch.bfloat16 if config["training"]["mixed_precision"] == "bf16" else torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["tokenizer_name"])
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.to(device)
    
    if config["training"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    
    # Setup FSDP if distributed
    if use_distributed:
        model = setup_fsdp_model(model, config)
    
    if is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Load data
    if is_main_process:
        logger.info("Loading training data...")
    
    train_loader, val_loader = create_data_loaders(
        train_path=config["data"]["train_path"],
        validation_path=config["data"].get("validation_path"),
        tokenizer=tokenizer,
        batch_size=config["training"]["batch_size"],
        max_seq_length=config["model"]["max_seq_length"],
        field_name=config["data"]["field_name"],
        num_workers=config["data"].get("num_workers", 0),
        streaming=config["data"].get("streaming", False),
    )
    
    # Setup gradient scaler for mixed precision
    use_amp = config["training"].get("mixed_precision") == "bf16"
    grad_scaler = GradScaler(enabled=use_amp) if use_amp else None
    if is_main_process and use_amp:
        logger.info("✓ Gradient scaling enabled for BF16 training")
    
    # Setup optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config["training"]["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=config["training"]["learning_rate"],
    )
    
    # Setup scheduler
    num_training_steps = len(train_loader) * config["training"]["num_epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["training"]["warmup_steps"],
        num_training_steps=num_training_steps,
    )
    
    if is_main_process:
        logger.info(f"Total training steps: {num_training_steps}")
        logger.info(f"Warmup steps: {config['training']['warmup_steps']}")
    
    # Training loop
    model.train()
    global_step = 0
    best_val_loss = float("inf")
    
    # Error recovery: check for existing checkpoints
    resume_checkpoint = find_latest_checkpoint(output_dir)
    if resume_checkpoint and config["checkpointing"].get("resume_from_checkpoint") is None:
        if is_main_process:
            logger.info(f"Found checkpoint {resume_checkpoint}, will resume from there")
        config["checkpointing"]["resume_from_checkpoint"] = str(resume_checkpoint)
    
    try:
    
        for epoch in range(config["training"]["num_epochs"]):
            epoch_loss = 0.0
            
            if is_main_process:
                logger.info(f"Epoch {epoch + 1}/{config['training']['num_epochs']}")
            
            train_loader_iter = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}",
                disable=not is_main_process,
            )
            
            for step, batch in enumerate(train_loader_iter):
                # Move batch to device
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                
                # Forward pass with autocast for mixed precision
                with autocast(enabled=use_amp, dtype=torch.bfloat16):
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss
                
                # Scale loss for gradient accumulation
                scaled_loss = loss / config["training"]["gradient_accumulation_steps"]
                
                # Backward with gradient scaling
                if grad_scaler:
                    grad_scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                
                epoch_loss += loss.item()
                
                # Optimizer step
                if (step + 1) % config["training"]["gradient_accumulation_steps"] == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Optimizer step with gradient scaling
                    if grad_scaler:
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                    else:
                        optimizer.step()
                    
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                    
                    if global_step % config["evaluation"]["eval_interval"] == 0 and is_main_process:
                        train_loss_avg = epoch_loss / (step + 1)
                        logger.info(f"Step {global_step}: loss={loss.item():.4f}, avg_loss={train_loss_avg:.4f}")
                        
                        # Log to W&B
                        if HAS_WANDB and wandb.run:
                            wandb.log({
                                "train/loss": loss.item(),
                                "train/avg_loss": train_loss_avg,
                                "train/learning_rate": scheduler.get_last_lr()[0],
                                "train/step": global_step,
                            })
                
                # Validation
                if global_step % config["evaluation"]["eval_interval"] == 0 and val_loader:
                    model.eval()
                    val_loss = 0.0
                    val_steps = 0
                    
                    with torch.no_grad():
                        for val_batch in val_loader:
                            val_input_ids = val_batch["input_ids"].to(device)
                            val_labels = val_batch["labels"].to(device)
                            
                            val_outputs = model(input_ids=val_input_ids, labels=val_labels)
                            val_loss += val_outputs.loss.item()
                            val_steps += 1
                    
                    val_loss_avg = val_loss / val_steps if val_steps > 0 else 0
                    
                    if is_main_process:
                        logger.info(f"Step {global_step}: val_loss={val_loss_avg:.4f}")
                        
                        # Log to W&B
                        if HAS_WANDB and wandb.run:
                            wandb.log({
                                "validation/loss": val_loss_avg,
                                "validation/step": global_step,
                            })
                        
                        # Save checkpoint if best model
                        if val_loss_avg < best_val_loss:
                            best_val_loss = val_loss_avg
                            checkpoint_dir = output_dir / f"best_model"
                            checkpoint_dir.mkdir(exist_ok=True)
                            
                            if use_distributed:
                                model.module.save_pretrained(checkpoint_dir)
                            else:
                                model.save_pretrained(checkpoint_dir)
                            
                            tokenizer.save_pretrained(checkpoint_dir)
                            logger.info(f"Saved best model to {checkpoint_dir}")
                    
                    model.train()
                
                # Periodic checkpoint
                if global_step % config["checkpointing"]["save_interval"] == 0 and is_main_process:
                    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                    checkpoint_dir.mkdir(exist_ok=True)
                    
                    if use_distributed:
                        model.module.save_pretrained(checkpoint_dir)
                    else:
                        model.save_pretrained(checkpoint_dir)
                    
                    tokenizer.save_pretrained(checkpoint_dir)
                    logger.info(f"Saved checkpoint to {checkpoint_dir}")
        
    except KeyboardInterrupt:
        if is_main_process:
            logger.warning("⚠ Training interrupted by user (Ctrl+C)")
            logger.info("Saving checkpoint before exit...")
            checkpoint_dir = output_dir / f"checkpoint-interrupted-{global_step}"
            checkpoint_dir.mkdir(exist_ok=True)
            if use_distributed:
                model.module.save_pretrained(checkpoint_dir)
            else:
                model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            logger.info(f"Saved interrupt checkpoint to {checkpoint_dir}")
        raise
    
    except Exception as e:
        if is_main_process:
            logger.error(f"✗ Training failed with error: {e}", exc_info=True)
            # Save emergency checkpoint
            checkpoint_dir = output_dir / f"checkpoint-error-{global_step}"
            checkpoint_dir.mkdir(exist_ok=True)
            try:
                if use_distributed:
                    model.module.save_pretrained(checkpoint_dir)
                else:
                    model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                logger.info(f"Saved error recovery checkpoint to {checkpoint_dir}")
            except Exception as save_err:
                logger.error(f"Could not save recovery checkpoint: {save_err}")
        raise
    
    # Save final model
    if is_main_process:
        final_dir = output_dir / "final_model"
        final_dir.mkdir(exist_ok=True)
        
        if use_distributed:
            model.module.save_pretrained(final_dir)
        else:
            model.save_pretrained(final_dir)
        
        tokenizer.save_pretrained(final_dir)
        logger.info(f"Saved final model to {final_dir}")
        
        # Push to Hub if configured
        if config.get("model_upload", {}).get("push_to_hub", False):
            hub_model_id = config["model_upload"]["hub_model_id"]
            logger.info(f"Pushing model to Hub: {hub_model_id}")
            
            if use_distributed:
                model.module.push_to_hub(
                    hub_model_id,
                    private=config["model_upload"].get("hub_private", True),
                    commit_message=config["model_upload"].get("commit_message", "Continued pretraining"),
                )
            else:
                model.push_to_hub(
                    hub_model_id,
                    private=config["model_upload"].get("hub_private", True),
                    commit_message=config["model_upload"].get("commit_message", "Continued pretraining"),
                )
    
    if use_distributed:
        dist.destroy_process_group()
    
    if is_main_process:
        logger.info("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LLM with continued pretraining")
    parser.add_argument(
        "--config",
        type=str,
        default="./config/training_config.yaml",
        help="Path to training config YAML file",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="Local rank for distributed training",
    )
    
    args = parser.parse_args()
    
    # Get world size from environment if distributed
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    train(
        config_path=args.config,
        local_rank=args.local_rank,
        world_size=world_size,
    )
