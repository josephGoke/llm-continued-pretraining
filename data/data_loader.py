"""
Data loading utilities for JSONL raw text pretraining data.
Supports both local files and streaming from HuggingFace datasets.
"""

import json
import logging
from typing import Optional, Dict, List
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, PreTrainedTokenizer

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

logger = logging.getLogger(__name__)


class JSONLDataset(Dataset):
    """
    PyTorch Dataset for reading JSONL files for continued pretraining.
    Expects each line to be valid JSON with a text field.
    """
    
    def __init__(
        self,
        file_path: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 2048,
        field_name: str = "text",
        stride: int = None,
    ):
        """
        Args:
            file_path: Path to JSONL file
            tokenizer: HF tokenizer instance
            max_seq_length: Maximum sequence length for tokenization
            field_name: Key in JSON for text content
            stride: If set, use overlapping windows with this stride (for document packing)
        """
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.field_name = field_name
        self.stride = stride
        self.examples = []
        
        self._load_data()
    
    def _load_data(self):
        """Load and tokenize data from JSONL file."""
        logger.info(f"Loading data from {self.file_path}")
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        
        texts = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    if self.field_name not in data:
                        logger.warning(
                            f"Line {line_num}: Missing field '{self.field_name}', skipping"
                        )
                        continue
                    
                    text = data[self.field_name]
                    if not isinstance(text, str):
                        logger.warning(
                            f"Line {line_num}: Field '{self.field_name}' is not string, skipping"
                        )
                        continue
                    
                    if len(text.strip()) == 0:
                        logger.warning(f"Line {line_num}: Empty text, skipping")
                        continue
                    
                    texts.append(text)
                
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num}: Invalid JSON ({e}), skipping")
                    continue
        
        if not texts:
            raise ValueError(f"No valid text data found in {self.file_path}")
        
        logger.info(f"Loaded {len(texts)} documents from {self.file_path}")
        
        # Tokenize all texts
        concatenated_text = " ".join(texts)
        del texts  # Free memory
        
        logger.info(f"Tokenizing concatenated text (length: {len(concatenated_text)})")
        tokenized = self.tokenizer(
            concatenated_text,
            return_tensors=None,
            truncation=False,
            max_length=None,
        )
        
        input_ids = tokenized["input_ids"]
        logger.info(f"Total tokens: {len(input_ids)}")
        
        # Create examples with sliding window if stride not specified
        if self.stride is None:
            self.stride = self.max_seq_length
        
        for i in range(0, len(input_ids) - self.max_seq_length + 1, self.stride):
            self.examples.append({
                "input_ids": input_ids[i : i + self.max_seq_length],
            })
        
        logger.info(f"Created {len(self.examples)} examples from {len(input_ids)} tokens")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        input_ids = torch.tensor(example["input_ids"], dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),  # For language modeling, labels = input_ids
        }


def create_data_loaders(
    train_path: str,
    validation_path: Optional[str],
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    max_seq_length: int = 2048,
    field_name: str = "text",
    num_workers: int = 0,
    pin_memory: bool = True,
    streaming: bool = False,
) -> tuple:
    """
    Create train and validation data loaders.
    Supports both local files and streaming datasets.
    
    Args:
        train_path: Path to training JSONL file or dataset identifier
        validation_path: Path to validation JSONL file (optional)
        tokenizer: HF tokenizer instance
        batch_size: Batch size for loaders
        max_seq_length: Max sequence length
        field_name: JSON field containing text
        num_workers: Number of workers for DataLoader
        pin_memory: Whether to pin memory
        streaming: If True, stream from HuggingFace datasets instead of loading all data
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    logger.info("Creating datasets...")
    
    # Check if using streaming (dataset identifier) or local files
    if streaming and HAS_DATASETS:
        logger.info(f"Using streaming dataset: {train_path}")
        train_dataset = load_dataset(train_path, split="train", streaming=True)
        
        # Tokenize on-the-fly for streaming
        def tokenize_function(examples):
            return tokenizer(
                examples[field_name],
                truncation=True,
                max_length=max_seq_length,
                padding="max_length",
            )
        
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[field_name],
        )
        
        val_dataset = None
        if validation_path:
            val_dataset = load_dataset(validation_path, split="validation", streaming=True)
            val_dataset = val_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=[field_name],
            )
    else:
        # Local file loading
        train_dataset = JSONLDataset(
            file_path=train_path,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            field_name=field_name,
        )
        
        val_dataset = None
        if validation_path:
            val_dataset = JSONLDataset(
                file_path=validation_path,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                field_name=field_name,
            )
    
    logger.info("Creating data loaders...")
    
    # For streaming datasets, we can't use custom samplers
    if streaming and HAS_DATASETS and hasattr(train_dataset, 'take'):
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=0,  # Streaming doesn't support num_workers
            pin_memory=pin_memory,
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                num_workers=0,
                pin_memory=pin_memory,
            )
    else:
        # Local file loading with samplers
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=RandomSampler(train_dataset),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,  # Important for distributed training
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                sampler=SequentialSampler(val_dataset),
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False,
            )
    
    logger.info(
        f"Train dataset size: {len(train_dataset)}, "
        f"Val dataset size: {len(val_dataset) if val_dataset else 0}"
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")
    train_loader, val_loader = create_data_loaders(
        train_path="./data/train.jsonl",
        validation_path="./data/val.jsonl",
        tokenizer=tokenizer,
        batch_size=8,
        max_seq_length=2048,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader) if val_loader else 0}")
    
    # Test loading a batch
    for batch in train_loader:
        print(f"Batch shape: {batch['input_ids'].shape}")
        break
