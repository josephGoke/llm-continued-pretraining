# Continued Pretraining For LLM Pipeline

A production-ready Python pipeline for continued pretraining of Large Language Models using PyTorch FSDP for distributed training. Built for domain-specific pretraining on raw JSONL text data.

## Features

- **FSDP Distributed Training**: Fully Sharded Data Parallel support for multi-GPU/multi-node training
- **Raw Text Pretraining**: JSONL-based data loading with streaming support
- **Checkpointing & Recovery**: Save/load checkpoints with best model tracking
- **Validation Monitoring**: Regular validation with perplexity and loss metrics
- **Model Hub Integration**: Optional push to Hugging Face Model Hub
- **Mixed Precision Training**: BF16 support for memory efficiency
- **Flexible Configuration**: YAML-based config for easy experiment management
- **Evaluation & Inference**: Built-in evaluation and inference scripts

## Project Structure

```
llm-continued-pretraining/
├── config/
│   └── training_config.yaml          # Training hyperparameters
├── data/
│   ├── data_loader.py                # JSONL data loading utilities
│   ├── train.jsonl                   # Training data (user-provided)
│   └── val.jsonl                     # Validation data (user-provided)
├── scripts/
│   ├── download_model.py             # Download models from HF Hub
│   ├── convert_data.py               # Convert data formats to JSONL
│   └── benchmark.py                  # Benchmark inference speed
├── outputs/
│   ├── checkpoint/                   # Training checkpoints
│   ├── logs/                         # Training logs
│   └── best_model/                   # Best model checkpoint
├── train.py                          # Main training script (ENTRY POINT)
├── evaluate.py                       # Evaluation on validation set
├── inference.py                      # Generation/inference wrapper
├── requirements.txt                  # Python dependencies
├── .env.example                      # Environment variable template
└── README.md                         # This file

```

## Installation

### Prerequisites
- Python 3.10+
- CUDA 12.0+ (for GPU training)
- 8GB+ GPU memory (recommended: 40GB+ for 7B model training)

### Setup

1. **Clone/create the project directory:**
   ```bash
   cd llm-continued-pretraining
   ```

2. **Create a Python virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your settings (model name, paths, etc.)
   ```

## Data Preparation

### JSONL Format

The pipeline expects JSONL files where each line is valid JSON with a `text` field:

```jsonl
{"text": "First document text here..."}
{"text": "Second document text here..."}
{"text": "Third document text here..."}
```

### Data Conversion

Use the conversion utility to prepare your data:

**From plain text:**
```bash
python scripts/convert_data.py \
  --input data.txt \
  --output data.jsonl \
  --format txt
```

**From CSV:**
```bash
python scripts/convert_data.py \
  --input data.csv \
  --output data.jsonl \
  --format csv \
  --text_column "content_column"
```

**Split into train/val:**
```bash
python scripts/convert_data.py \
  --input data.jsonl \
  --output data.jsonl \
  --split 0.9  # 90% train, 10% val
```

Place the resulting `train.jsonl` and `val.jsonl` in the `data/` directory.

## Configuration

Edit `config/training_config.yaml` to customize training:

```yaml
model:
  name: meta-llama/Llama-2-7b
  max_seq_length: 2048

training:
  num_epochs: 3
  learning_rate: 5e-5
  batch_size: 8
  gradient_accumulation_steps: 4
  mixed_precision: bf16

distributed:
  use_fsdp: true
  fsdp_config:
    sharding_strategy: FULL_SHARD
    cpu_offload: false

model_upload:
  push_to_hub: false  # Set to true to upload to HF Hub
  hub_model_id: "org/model-name"
```

## Training

### Single GPU Training

```bash
python train.py --config config/training_config.yaml
```

### Multi-GPU Training (FSDP)

**Using torchrun:**
```bash
torchrun --nproc_per_node=4 train.py --config config/training_config.yaml
```

**Using accelerate:**
```bash
accelerate config  # Follow prompts to configure distributed setup
accelerate launch train.py --config config/training_config.yaml
```

### Resume from Checkpoint

Edit `config/training_config.yaml` and set:
```yaml
checkpointing:
  resume_from_checkpoint: ./outputs/checkpoint/checkpoint-5000
```

Then run:
```bash
python train.py --config config/training_config.yaml
```

## Evaluation

Evaluate the trained model on validation data:

```bash
python evaluate.py \
  --model_path ./outputs/checkpoint/best_model \
  --validation_data ./data/val.jsonl \
  --output_file ./outputs/eval_results.json
```

## Inference

### Interactive Mode

```bash
python inference.py --model_path ./outputs/checkpoint/best_model
```

Then enter prompts interactively.

### Command Line Prompt

```bash
python inference.py \
  --model_path ./outputs/checkpoint/best_model \
  --prompt "Once upon a time" \
  --max_length 512
```

### Python API

```python
from inference import PretrainedModelInference

model = PretrainedModelInference("./outputs/checkpoint/best_model")
outputs = model.generate(
    prompt="The future of AI",
    max_length=256,
    temperature=0.7,
)
print(outputs[0])
```

## Benchmarking

Benchmark inference speed:

```bash
python scripts/benchmark.py \
  --model_path ./outputs/checkpoint/best_model \
  --batch_sizes 1 4 8 \
  --seq_lengths 128 512 2048
```

## Model Upload to Hugging Face Hub

To automatically push your model to the Hub:

1. Set in `config/training_config.yaml`:
   ```yaml
   model_upload:
     push_to_hub: true
     hub_model_id: "your-org/your-model"
     hub_private: true
   ```

2. Ensure you're logged in to HF:
   ```bash
   huggingface-cli login
   ```

3. Run training as normal—the model will be pushed at the end

## Troubleshooting

### CUDA Out of Memory

- Reduce `batch_size` in config
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Enable `cpu_offload` in FSDP config
- Use smaller model (`meta-llama/Llama-2-7b` instead of larger variants)

### Data Loading Issues

- Ensure JSONL is valid: `python -c "import json; [json.loads(line) for line in open('data/train.jsonl')]"`
- Check that the `field_name` in config matches the key in your JSON
- Verify text field is not empty

### Distributed Training Hangs

- Check NCCL debugging: `export NCCL_DEBUG=INFO`
- Ensure all GPUs are visible: `nvidia-smi`
- Verify network connectivity for multi-node training

## Performance Tips

1. **Use BF16 mixed precision** (enabled by default) for 2x speedup with minimal accuracy loss
2. **Enable gradient checkpointing** to reduce memory usage at cost of compute
3. **Use FSDP with sharding** for large models that don't fit on a single GPU
4. **Increase batch size** as much as possible while staying in VRAM
5. **Monitor validation loss** to detect overfitting early

## License

MIT License - feel free to use for research and production

## References

- [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Continued Pretraining Papers](https://arxiv.org/search/?query=continued+pretraining)

## Citation

If you use this pipeline in your work, please cite:

```bibtex
@software{llm_continued_pretraining,
  title={LLM Continued Pretraining Pipeline},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/llm-continued-pretraining}
}
```

## Contributing

Contributions welcome! Please submit PRs for bug fixes, features, or documentation improvements.
