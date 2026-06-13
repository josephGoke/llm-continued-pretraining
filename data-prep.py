"""
Standalone script to stream, mix, deduplicate, pack, and upload 
a custom pretraining dataset to the Hugging Face Hub.
Run this on a CPU-heavy instance before starting your GPU training.
"""

import os
import hashlib
from datasets import load_dataset, interleave_datasets, IterableDataset
from transformers import AutoTokenizer

# Configuration
TOKENIZER_NAME = "LiquidAI/LFM2.5-8B-A1B-Base"
MAX_SEQ_LENGTH = 4096
HUB_REPO_ID = "Crtop/packed-quant-math-mixture" # Replace with your HF username
HF_TOKEN = os.environ.get("HF_USE_AUTH") 

# Initialize Tokenizer
print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================================================================
# 1. STREAMING & STANDARDIZATION
# ============================================================================
# def standardize(example, text_col):
#     """Ensures every dataset outputs a single 'text' column."""
#     return {"text": example[text_col]}

print("Connecting to upstream data streams...")
se_stats = load_dataset("HuggingFaceH4/stack-exchange-preferences", data_dir="data/stats.stackexchange.com", split="train", streaming=True)
se_quant = load_dataset("HuggingFaceH4/stack-exchange-preferences", data_dir="data/quant.stackexchange.com", split="train", streaming=True)
arxiv = load_dataset("togethercomputer/RedPajama-Data-1T", "arxiv", split="train", streaming=True, trust_remote_code=True)

# stack_python = load_dataset("bigcode/the-stack-v2-train-smol", "python", split="train", streaming=True, token=HF_TOKEN)
# Load the default configuration for the stack-v2-smol
stack_python = load_dataset("bigcode/the-stack-v2-train-smol", split="train", streaming=True, token=HF_TOKEN)
# Filter stream to ONLY keep Python files
stack_python = stack_python.filter(lambda x: x.get("language", "").lower() == "python")

fineweb = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

# Map to standard column
# se_stats = se_stats.map(lambda x: standardize(x, "question"), remove_columns=se_stats.features.keys())
# se_quant = se_quant.map(lambda x: standardize(x, "question"), remove_columns=se_quant.features.keys())
# arxiv = arxiv.map(lambda x: standardize(x, "text"), remove_columns=arxiv.features.keys())
# stack_python = stack_python.map(lambda x: standardize(x, "content"), remove_columns=stack_python.features.keys())
# fineweb = fineweb.map(lambda x: standardize(x, "text"), remove_columns=fineweb.features.keys())
# Helper to peek at the first row of a streaming dataset to get its columns
# def get_columns(streaming_dataset):
#     return list(next(iter(streaming_dataset)).keys())

# Map to standard column and drop all original columns
# se_stats = se_stats.map(lambda x: standardize(x, "question"), remove_columns=get_columns(se_stats))
# se_quant = se_quant.map(lambda x: standardize(x, "question"), remove_columns=get_columns(se_quant))
# arxiv = arxiv.map(lambda x: standardize(x, "text"), remove_columns=get_columns(arxiv))
# stack_python = stack_python.map(lambda x: standardize(x, "content"), remove_columns=get_columns(stack_python))
# fineweb = fineweb.map(lambda x: standardize(x, "text"), remove_columns=get_columns(fineweb))

print("Standardizing datasets natively...")

# 1. Select only the target text column (drops all others automatically)
# 2. Rename it to "text" so they all match perfectly for interleaving
se_stats = se_stats.select_columns(["question"]).rename_column("question", "text")
se_quant = se_quant.select_columns(["question"]).rename_column("question", "text")
arxiv = arxiv.select_columns(["text"]) # Already named 'text'
stack_python = stack_python.select_columns(["content"]).rename_column("content", "text")
fineweb = fineweb.select_columns(["text"]) # Already named 'text'

# ============================================================================
# 2. DEDUPLICATION (MD5 Hashing)
# ============================================================================
# NOTE: This set will grow in memory. For 100M documents, this takes ~6-8GB RAM.
seen_hashes = set()

def is_not_duplicate(example):
    """Creates a fast MD5 hash of the text and checks for exact duplicates."""
    # Encode text and create a fast hex hash
    text_hash = hashlib.md5(example["text"].encode('utf-8')).hexdigest()
    
    if text_hash in seen_hashes:
        return False # It's a duplicate, filter it out
    
    seen_hashes.add(text_hash)
    return True # Keep it

# ============================================================================
# 3. INTERLEAVE MIXTURE
# ============================================================================
print("Interleaving datasets...")
# Target Mixture: 10% Stats, 10% Quant, 30% ArXiv, 25% Python, 25% FineWeb
datasets = [se_stats, se_quant, arxiv, stack_python, fineweb]
probabilities = [0.10, 0.10, 0.30, 0.25, 0.25]

mixed_dataset = interleave_datasets(datasets, probabilities=probabilities, seed=42)

# Apply the deduplication filter across the entire interleaved stream
deduplicated_dataset = mixed_dataset.filter(is_not_duplicate)

# ============================================================================
# 4. TOKENIZATION & SEQUENCE PACKING
# ============================================================================
def tokenize_and_pack(examples):
    """
    Batched mapping function to tokenize and concatenate text until it 
    perfectly fills the MAX_SEQ_LENGTH. Drops trailing remainders.
    """
    # Tokenize the batch of texts
    tokenized_inputs = tokenizer(
        examples["text"], 
        truncation=False, # Do not truncate yet, we want to pack!
        add_special_tokens=True # Adds EOS tokens between documents
    )
    
    # Concatenate all tokens into one massive list
    concatenated_ids = sum(tokenized_inputs["input_ids"], [])
    
    # Calculate how many full blocks of MAX_SEQ_LENGTH we can make
    total_length = len(concatenated_ids)
    if total_length >= MAX_SEQ_LENGTH:
        total_length = (total_length // MAX_SEQ_LENGTH) * MAX_SEQ_LENGTH
        
    # Split into perfectly sized chunks
    result_ids = [
        concatenated_ids[i : i + MAX_SEQ_LENGTH] 
        for i in range(0, total_length, MAX_SEQ_LENGTH)
    ]
    
    # Return as standard 'input_ids' and 'labels' for Causal LM training
    return {
        "input_ids": result_ids,
        "labels": result_ids.copy() # For autoregressive models, labels = input_ids
    }

print("Configuring Tokenizer and Packing routines...")
# Apply packing in batches for streaming efficiency
packed_dataset = deduplicated_dataset.map(
    tokenize_and_pack,
    batched=True,
    batch_size=1000,
    remove_columns=["text"] # Drop the raw text, we only need binary tokens now!
)

# ============================================================================
# 5. UPLOAD TO HUB
# ============================================================================
print(f"Beginning stream, pack, and upload process to {HUB_REPO_ID}...")
print("This may take several hours depending on network speed.")

# Push the iterable dataset directly to the hub
packed_dataset.push_to_hub(
    HUB_REPO_ID,
    private=True,
    token=HF_TOKEN,
    max_shard_size="1GB" # Creates clean 1GB Parquet files optimized for streaming
)

print("Data prep complete! Ready for FSDP training.")