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
import random
import numpy as np
import torch
import torch.distributed as dist

from utils import (
    set_seed,
    load_config,
    setup_distributed,
    setup_wandb,
    transformer_layer_wrap_policy,
    activation_wrap_policy,
    precision_policy,
    find_latest_checkpoint,
    setup_fsdp_model,
    save_checkpoint
)

from data import create_data_loaders
from utils import model_config_validator, EarlyStopping

from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
import functools


# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)




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
        try:
            model_config_validator(config)
        except Exception as e:
            logger.error(f"Config validation failed: {e}")
            raise
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
    set_seed(config["training"].get("seed", 42))
    
    # Load model and tokenizer
    if is_main_process:
        logger.info(f"Loading model: {config['model']['name']}")
    
    # If resuming, load weights from checkpoint instead of base model
    resume_path = find_latest_checkpoint(output_dir) if find_latest_checkpoint(output_dir) else config["checkpointing"].get("resume_from_checkpoint")
    model_name_or_path = str(resume_path) if resume_path and Path(resume_path).exists() else config["model"]["name"]
    
    if is_main_process and hasattr(config["model"], "name") and model_name_or_path != config["model"]["name"]:
        logger.info(f"Resuming model weights from {model_name_or_path}")
    
    model, tokenizer = setup_model()


    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.to(device)
    
    if config["training"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    
    # if is_main_process:
    #     pre_flight_check(config, model, tokenizer)

    
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
        validation_path=config["data"].get("validation_path"), world_size=world_size, rank=rank,
        tokenizer=tokenizer,
        batch_size=config["training"]["batch_size"],
        max_seq_length=config["model"]["max_seq_length"],
        field_name=config["data"]["field_name"],
        num_workers=config["data"].get("num_workers", 0),
        streaming=config["data"].get("streaming", False),
    )
    
    # Setup gradient scaler for mixed precision
    use_amp = config["training"].get("mixed_precision") == "bf16"
    grad_scaler = GradScaler("cuda", enabled=use_amp) if use_amp else None
    if is_main_process and use_amp:
        logger.info("[OK] Gradient scaling enabled for BF16 training")
    
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
    
    early_stopper = EarlyStopping(patience=config.get("evaluation", {}).get("patience", 3))
    
    # Error recovery: check for existing checkpoints
    resume_checkpoint = find_latest_checkpoint(output_dir)
    if resume_checkpoint and config["checkpointing"].get("resume_from_checkpoint") is None:
        if is_main_process:
            logger.info(f"Found checkpoint {resume_checkpoint}, will resume from there")
        config["checkpointing"]["resume_from_checkpoint"] = str(resume_checkpoint)
    
    state_file = None
    resume_epoch = 0
    if config["checkpointing"].get("resume_from_checkpoint"):
        checkpoint_path = Path(config["checkpointing"]["resume_from_checkpoint"])
        if checkpoint_path.exists():
            state_file = checkpoint_path / "training_state.pt"
            if state_file.exists():
                if is_main_process:
                    logger.info(f"Loading training state from {state_file}")
                
                loc = "cuda:{}".format(local_rank) if use_distributed else device
                try:
                    checkpoint_state = torch.load(state_file, map_location=loc)
                    optimizer.load_state_dict(checkpoint_state["optimizer"])
                    scheduler.load_state_dict(checkpoint_state["scheduler"])
                    global_step = checkpoint_state.get("global_step", 0)
                    best_val_loss = checkpoint_state.get("best_val_loss", float("inf"))
                    resume_epoch = checkpoint_state.get("epoch", 0)
                    
                    if is_main_process:
                        logger.info(f"Resumed from step {global_step}, epoch {resume_epoch}")
                except Exception as e:
                    if is_main_process:
                        logger.warning(f"Could not load training state: {e}")
            
    try:
    
        for epoch in range(config["training"]["num_epochs"]):
            if state_file and epoch < resume_epoch:
                continue

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
                with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
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
                    
                    # Synchronize validation loss across all ranks
                    if use_distributed:
                        val_loss_tensor = torch.tensor(val_loss_avg, device=device)
                        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
                        val_loss_avg = val_loss_tensor.item()


                    if is_main_process:
                        logger.info(f"Step {global_step}: val_loss={val_loss_avg:.4f}")
                        
                        # Log to W&B
                        if HAS_WANDB and wandb.run:
                            wandb.log({
                                "validation/loss": val_loss_avg,
                                "validation/step": global_step,
                            })
                        
                        # Save checkpoint if best model

                        if early_stopper(val_loss_avg):
                            logger.info("Early stopping triggered!")
                            break
                        if val_loss_avg < best_val_loss:
                            best_val_loss = val_loss_avg
                            checkpoint_dir = output_dir / f"best_model"
                            checkpoint_dir.mkdir(exist_ok=True)
                            
                            save_checkpoint(model, tokenizer, optimizer, scheduler, global_step, epoch, best_val_loss, checkpoint_dir, use_distributed, is_main_process)
                    
                    model.train()
                
                # Evaluate early stopping globally using synchronized val_loss_avg
                if early_stopper(val_loss_avg):
                    if is_main_process:
                        logger.info("Early stopping triggered!")
                    break
                if global_step % config["checkpointing"]["save_interval"] == 0 and is_main_process:
                    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                    checkpoint_dir.mkdir(exist_ok=True)
                    
                    save_checkpoint(model, tokenizer, optimizer, scheduler, global_step, epoch, best_val_loss, checkpoint_dir, use_distributed, is_main_process)
        
    except KeyboardInterrupt:
        if is_main_process:
            logger.warning("⚠ Training interrupted by user (Ctrl+C)")
            logger.info("Saving checkpoint before exit...")
            checkpoint_dir = output_dir / f"checkpoint-interrupted-{global_step}"
            checkpoint_dir.mkdir(exist_ok=True)
            save_checkpoint(model, tokenizer, optimizer, scheduler, global_step, epoch, best_val_loss, checkpoint_dir, use_distributed, is_main_process)
        raise
    
    except Exception as e:
        if is_main_process:
            logger.error(f"✗ Training failed with error: {e}", exc_info=True)
            # Save emergency checkpoint
            checkpoint_dir = output_dir / f"checkpoint-error-{global_step}"
            checkpoint_dir.mkdir(exist_ok=True)
            try:
                save_checkpoint(model, tokenizer, optimizer, scheduler, global_step, epoch, best_val_loss, checkpoint_dir, use_distributed, is_main_process)
            except Exception as save_err:
                logger.error(f"Could not save recovery checkpoint: {save_err}")
        raise
    
    # Save final model
    if is_main_process:
        final_dir = output_dir / "final_model"
        final_dir.mkdir(exist_ok=True)
        
        save_checkpoint(model, tokenizer, optimizer, scheduler, global_step, epoch, best_val_loss, final_dir, use_distributed, is_main_process)
        
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
