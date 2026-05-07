# Quick Start Guide

Get your LLM continued pretraining pipeline running in 5 minutes.

## Step 1: Install Dependencies (2 min)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Step 2: Prepare Your Data (1 min)

Place your JSONL files in `data/`:
- `data/train.jsonl` (training data)
- `data/val.jsonl` (validation data)

**If you only have raw text, convert it:**
```bash
python scripts/convert_data.py --input raw_text.txt --output data/train.jsonl --format txt --split 0.9
```

Or use the provided sample data:
```bash
cp data/sample_data.jsonl data/train.jsonl
cp data/sample_data.jsonl data/val.jsonl
```

## Step 3: Configure Training (1 min)

Edit `config/training_config.yaml` to set:
- Model name (default: `meta-llama/Llama-2-7b`)
- Batch size, learning rate
- Number of epochs
- Other hyperparameters

## Step 4: Start Training! (1 min)

### Single GPU:
```bash
python train.py
```

### Multi-GPU (FSDP):
```bash
torchrun --nproc_per_node=4 train.py
```

Watch the training logs in `outputs/logs/`

## Step 5: Evaluate & Inference

**Evaluate on validation set:**
```bash
python evaluate.py \
  --model_path ./outputs/checkpoint/best_model \
  --validation_data ./data/val.jsonl
```

**Generate text:**
```bash
python inference.py \
  --model_path ./outputs/checkpoint/best_model \
  --prompt "Once upon a time"
```

---

## Common Commands

| Task | Command |
|------|---------|
| Download a model | `python scripts/download_model.py --model meta-llama/Llama-2-13b` |
| Convert CSV to JSONL | `python scripts/convert_data.py --input data.csv --output data.jsonl --format csv` |
| Benchmark inference | `python scripts/benchmark.py --model_path ./outputs/checkpoint/best_model` |
| Resume training | Edit config, set `resume_from_checkpoint`, then run `python train.py` |
| Upload to Hub | Set `push_to_hub: true` in config, run training |
| Training | 

# Single GPU
python train.py

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 train.py
---

## Tips

- **First run?** Use sample data to test the pipeline: `cp data/sample_data.jsonl data/train.jsonl`
- **GPU OOM?** Reduce `batch_size` or increase `gradient_accumulation_steps` in config
- **Multi-node training?** Set environment variables and use `torchrun` with proper MASTER_ADDR/PORT
- **Monitor training?** Check `outputs/logs/training_*.log` and validation metrics in console

---

For detailed documentation, see [README.md](README.md)
