import random
from datasets import config
import numpy as np
import torch
import yaml
import os
import pathlib
import logging
import functools



from pathlib import Path
from typing import Optional
from datetime import datetime
from torch import distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper
)
from torch.distributed.fsdp  import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    BackwardPrefetch,
    ShardingStrategy,
    MixedPrecision,
    FullStateDictConfig,
    StateDictConfig,
    StateDictType
)

#configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

def set_seed(seed: int):
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)
    # logger.info(f"Random seed set to {seed}")
    return 

# Load config: training_config.yaml
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


# checkpoint util
def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    """Find the latest checkpoint in output directory."""
    checkpoints = sorted(
        [d for d in output_dir.glob("checkpoint-*") if d.is_dir()],
        key=lambda x: int(x.name.split("-")[-1]),
    )
    return checkpoints[-1] if checkpoints else None



# Weights & Bias Setup
def setup_wandb(config: dict, is_main_process: bool):
    """Initialize Weights & Biases logging."""
    if not is_main_process or not HAS_WANDB:
        return
    
    wandb_cfg = config.get("logging", {})
    if not wandb_cfg.get("use_wandb", False):
        return
    
    try:
        wandb.init(
            project=wandb_cfg.get("wandb_project", "llm-pretraining"),
            entity=wandb_cfg.get("wandb_entity"),
            config={
                "model": config.get("model", {}),
                "training": config.get("training", {}),
            },
            name=f"pretraining-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        logger.info("[OK] Weights & Biases initialized")
    except Exception as e:
        logger.warning(f"[WARN] Failed to initialize W&B: {e}")
        logger.warning("[WARN] Continuing training without W&B monitoring")


# Util for getting transformer layer class wrapper
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
        "lfm2": "Lfm2DecoderLayer",
        "lfm2_moe": "Lfm2MoeDecoderLayer"
    }
    
    try:
        if model_type in layer_class_map:
            layer_name = layer_class_map[model_type]
            logger.debug(f"Detected model type: {model_type}, layer class: {layer_name}")
            return layer_name
    except Exception as e:
        logger.debug(f"Could not detect layer class: {e}")
    

# Transformer Auto-Wrap Policy for FSDP
def transformer_layer_wrap_policy(blocks):
    layer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=[blocks]
    )
    return layer_wrap_policy


# Activation Checkpointing Wrapper for FSDP
def set_activation_checkpointing(layer_cls):
    activation_wrap_policy = functools.partial(
        checkpoint_wrapper,
        offload_to_cpu=False,
        checkpoint_impl=CheckpointImpl.No_REENTRANT,
    )
    check_fn = lambda module: isinstance(module, layer_cls)
    apply_activation_checkpointing(checkpoint_wrapper_fn=activation_wrap_policy, check_fn=check_fn)


# Mixed Precision policy for FSDP: using bfloat16 for parameters, gradients, and buffers
precision_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)



def setup_model(config: dict, device: torch.device, is_main_process: bool = True):
    """
    Load and configure model and tokenizer for training.
    
    Args:
        config: Training configuration dictionary
        device: PyTorch device (cuda or cpu)
        is_main_process: Whether this is the main process (for logging)
        
    Returns:
        tuple: (model, tokenizer)
    """
    # Extract model configuration
    model_cfg = config.get("model", {})
    model_name_or_path = model_cfg.get("name")
    tokenizer_name = model_cfg.get("tokenizer_name", model_name_or_path)
    
    if not model_name_or_path:
        raise ValueError("Model name not specified in config['model']['name']")
    
    if is_main_process:
        logger.info(f"Loading model from: {model_name_or_path}")
    
    try:
        # Determine data type
        dtype = torch.bfloat16 if config["training"].get("mixed_precision") == "bf16" else torch.float32
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=True,  # For custom models like LFM2.5-8B-A1B-Base
            attn_implementation="flash_attention_2" if config["training"].get("use_flash_attention", True) else "eager",
        )
        
        if is_main_process:
            logger.info(f"[OK] Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        
        # Set pad token if not defined
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            if is_main_process:
                logger.info("Pad token set to EOS token")
        
        if is_main_process:
            logger.info(f"[OK] Tokenizer loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise
    
    # Move model to device (will be wrapped with FSDP later if distributed)
    model.to(device)
    
    # Enable gradient checkpointing if configured (saves memory at cost of compute)
    if config["training"].get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        if is_main_process:
            logger.info("[OK] Gradient checkpointing enabled")
    
    # Log model info
    if is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return model, tokenizer


def setup_fsdp_model(model, config):
    """
    Wrap model with FSDP for distributed training.
    Uses layer-wise auto-wrap policy for efficient communication and compute overlap.
    """
    if not dist.is_available() or not dist.is_initialized():
        logger.warning("Distributed not initialized, skipping FSDP wrapping")
        return model
    
    #loads distributed configs
    fsdp_cfg = config.get("distributed", {}).get("fsdp_config", {})
    
    # Sharded Model
    model = FSDP(
        model,
        auto_wrap_policy=transformer_auto_wrap_policy,
        sharding_strategy=fsdp_cfg.get("sharding_strategy", ShardingStrategy.FULL_SHARD),
        cpu_offload=CPUOffload(offload_params=fsdp_cfg.get("cpu_offload", False)),
        backward_prefetch=fsdp_cfg.get("backward_prefetch", BackwardPrefetch.BACKWARD_PRE),
        limit_all_gathers=fsdp_cfg.get("limit_all_gathers", True),
        sync_module_states=True,
        device_id=torch.cuda.current_device(),
    )
    
    set_activation_checkpointing(get_model_layer_class(model))

    logger.info("[OK] Model wrapped with FSDP (layer-wise auto-wrap policy enabled)")
    return model




# 

def save_checkpoint(model, tokenizer, optimizer, scheduler, global_step, epoch, best_val_loss, checkpoint_dir, use_distributed, is_main_process):
    """
    Save model checkpoint with proper FSDP support and error handling.
    
    Args:
        model: Model to save (may be wrapped with FSDP)
        tokenizer: Tokenizer to save
        optimizer: Optimizer state to save
        scheduler: Learning rate scheduler state to save
        global_step: Current global training step
        epoch: Current epoch number
        best_val_loss: Best validation loss so far
        checkpoint_dir: Directory to save checkpoint to
        use_distributed: Whether distributed training is enabled
        is_main_process: Whether this is the main process
    """
    # Create checkpoint directory
    try:
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
    except Exception as e:
        logger.error(f"Failed to create checkpoint directory {checkpoint_dir}: {e}")
        raise
    
    try:
        # Save model weights
        if use_distributed:
            logger.debug("Gathering sharded model state from all ranks...")
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                state_dict = model.state_dict()
            
            if is_main_process:
                # Properly unwrap FSDP wrapper(s) to get the base model
                unwrapped_model = model
                while hasattr(unwrapped_model, 'module'):
                    unwrapped_model = unwrapped_model.module
                
                unwrapped_model.save_pretrained(checkpoint_dir, state_dict=state_dict)
                logger.info(f"[OK] Model weights saved to {checkpoint_dir}")
        else:
            if is_main_process:
                model.save_pretrained(checkpoint_dir)
                logger.info(f"[OK] Model weights saved to {checkpoint_dir}")
        
        # Save tokenizer and training state (only on main process)
        if is_main_process:
            tokenizer.save_pretrained(checkpoint_dir)
            logger.info(f"[OK] Tokenizer saved to {checkpoint_dir}")
            
            # Save training state
            training_state = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "global_step": global_step,
                "epoch": epoch,
                "best_val_loss": best_val_loss
            }
            torch.save(training_state, checkpoint_dir / "training_state.pt")
            logger.info(f"[OK] Training state saved (step={global_step}, epoch={epoch}, val_loss={best_val_loss:.4f})")
        
        # Synchronize all ranks after checkpoint save
        if use_distributed:
            dist.barrier()
            
    except Exception as e:
        logger.error(f"[ERROR] Failed to save checkpoint: {e}", exc_info=True)
        raise



class EarlyStopping:
    """
    Stops training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
        return self.early_stop


def model_config_validator(config: dict):
    """
    Validates that the essential keys exist in the training_config.yaml.
    Prevents the script from crashing mid-execution due to missing parameters.
    """
    required_keys = {
        "model": ["name"],
        "data": ["train_path"],
        "training": ["output_dir", "batch_size", "num_epochs", "learning_rate"]
    }
    
    missing = []
    for section, keys in required_keys.items():
        if section not in config:
            missing.append(f"Missing section: [{section}]")
            continue
        for key in keys:
            if key not in config[section]:
                missing.append(f"Missing key: [{section}][{key}]")
                
    if missing:
        raise ValueError("Config Validation Failed:\n" + "\n".join(missing))