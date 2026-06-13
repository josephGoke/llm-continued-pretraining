"""
Data utilities for streaming and local dataset loading.
Optimized for FSDP distributed training with pre-packed tokens.
"""

import logging
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset
from transformers import default_data_collator

logger = logging.getLogger(__name__)

def prepacked_collate_fn(features):
    """
    Converts the streamed lists of integers from Hugging Face 
    into PyTorch LongTensors required for Causal LM training.
    """
    batch = {}
    batch["input_ids"] = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
    batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.long)
    return batch

Done. Now write Production training pipeline (train.py).....with wandb for monitoring, Logging/printouts......

def data_loaders(
    train_path: str,
    validation_path: str,
    world_size: int,
    rank: int,
    tokenizer,
    batch_size: int,
    max_seq_length: int,
    field_name: str = "text",
    num_workers: int = 2,
    streaming: bool = True,
):
    """
    Creates DataLoaders for both streaming Hub datasets and local files.
    """
    is_main_process = (rank == 0)
    
    if is_main_process:
        logger.info(f"Initializing DataLoaders (Streaming Mode: {streaming})")
        logger.info(f"Targeting dataset path/repo: {train_path}")

    # =================================================================
    # STREAMING FROM HUGGING FACE HUB (The new method)
    # =================================================================
    if streaming:
        if is_main_process:
            logger.info("Connecting to Hugging Face Hub stream...")
            
        train_dataset = load_dataset(train_path, split="train", streaming=True)
        
        # CRITICAL DISTRIBUTED FIX: Shard the stream!
        # This replaces the need for a DistributedSampler.
        if world_size > 1:
            train_dataset = train_dataset.shard(num_shards=world_size, index=rank)
            if is_main_process:
                logger.info(f"Successfully sharded stream across {world_size} GPUs.")

        # Create a small validation stream by taking the first few thousand examples 
        # from the validation split (if it exists) or taking from train.
        try:
            val_dataset = load_dataset(train_path, split="validation", streaming=True)
            if world_size > 1:
                val_dataset = val_dataset.shard(num_shards=world_size, index=rank)
        except Exception:
            if is_main_process:
                logger.warning("No 'validation' split found. Skipping validation dataset.")
            val_dataset = None

        # Create DataLoaders (Notice: shuffle=False and sampler=None for iterables)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=prepacked_collate_fn,
            drop_last=True
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=prepacked_collate_fn,
                drop_last=True
            )

    # =================================================================
    # LOCAL STATIC FILES (JSONL, Parquet)
    # =================================================================
    else:
        if is_main_process:
            logger.info("Loading local dataset into memory...")
            
        dataset = load_dataset("json", data_files={"train": train_path})
        train_dataset = dataset["train"]
        
        # If using local data, we need the standard DistributedSampler
        train_sampler = DistributedSampler(
            train_dataset, 
            num_replicas=world_size, 
            rank=rank, 
            shuffle=True
        ) if world_size > 1 else None

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=default_data_collator,
            shuffle=(train_sampler is None),
            drop_last=True
        )
        
        val_loader = None
        if validation_path and validation_path != "None":
            val_dataset = load_dataset("json", data_files={"validation": validation_path})["validation"]
            val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                sampler=val_sampler,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=default_data_collator,
                drop_last=True
            )
            
    if is_main_process:
        logger.info("[OK] DataLoaders successfully created and primed.")
        
    return train_loader, val_loader