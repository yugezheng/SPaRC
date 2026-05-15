# Path Retrieval Pipeline

A DualEncoder path retrieval pipeline for KGQA using InfoNCE with in-batch and explicit negatives. This guide covers environment setup, training, evaluation, and entity ID mapping.

---

## Project Structure

```
.
├── select_path_last_train.py   # Main training script
├── select_path.py             # Shared components (imported by eval script)
├── select_path_eval.py        # Evaluation script with top-k prediction saving
├── mapping.py                 # Map Freebase entity IDs to readable names
├── data/
│   └── WebQSP(CWQ)/
│       ├── train.json         # Training set (JSONL)
│       └── test.json          # Test set (JSONL)
├── finalentities.json         # Freebase ID → name lookup table
└── output/                    # Default output directory
```

---

## 1. Environment Setup

```bash
conda create -n stage1 python=3.10 -y
conda activate stage1
pip install -r requirements.txt
```
---

## 2. Training

Training is handled by `select_path_last_train.py`. The model is a **DualEncoder** trained with InfoNCE loss using both in-batch and explicit negatives.

**Example command:**

```bash
python select_path_last_train.py \
    --train_file ./data/WebQSP/train.json \
    --test_file  ./data/WebQSP/test.json \
    --base_model bert-base-uncased \
    --output_dir ./retrieval_result \
    --batch_size 32 \
    --epochs 5 \
    --lr 2e-5
```

**Full argument reference:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--train_file` | required | Path to training JSONL |
| `--test_file` | required | Path to test JSONL |
| `--base_model` | `bert-base-uncased` | HuggingFace model name or local path |
| `--output_dir` | `./retrieval_result` | Directory for saved models and results |
| `--batch_size` | `32` | Training batch size |
| `--eval_batch_size` | `64` | Evaluation batch size |
| `--epochs` | `5` | Number of training epochs |
| `--lr` | `2e-5` | Learning rate |
| `--temperature` | `0.05` | InfoNCE temperature |
| `--max_len` | `128` | Max token sequence length |
| `--projection_dim` | `256` | DualEncoder projection dimension |
| `--top_ks` | `1 3 5` | k values for evaluation metrics |
| `--neg_num` | `1` | Explicit negatives per sample |
| `--val_ratio` | `0.0` | Fraction of training data held out for validation |
| `--seed` | `42` | Random seed |

**Model saving:** The best checkpoint (by `Hit@1`) is saved automatically to:

```
./retrieval_result/<model_name>/model_<model>_bs<bs>_lr<lr>_best_model.pt
```

---

## 3. Evaluation

`select_path_eval.py` runs inference on the test set and saves per-sample top-k predictions alongside aggregate metrics (Hit@k, Precision@k, Recall@k, F1@k).

**Before running**, edit the configuration block at the bottom of `select_path_eval.py`:

```python
# ⚠️ Modify these paths before running
test_file      = r"./data/WebQSP/test.json"   # test set path
base_model     = "bert-base-uncased"           # must match training
model_path     = "model.pt"                    # path to saved .pt checkpoint
output_dir     = f"./output"
projection_dim = 256                           # must match training
top_ks         = [1, 3, 5, 10]
batch_size     = 16
```

> ⚠️ `select_path.py` is imported as a module. Both files **must be in the same directory**.

```bash
python select_path_eval.py
```

Outputs saved to `output_dir`:

- `*_predictions_<timestamp>.jsonl` — per-sample top-k predicted paths
- `*_eval_results_<timestamp>.json` — aggregate metric scores

---

## 4. Entity ID Mapping

`mapping.py` replaces Freebase IDs (`m.xxx` / `g.xxx`) in the evaluation output with human-readable entity names.

**Before running**, edit the path variables at the top of `mapping.py`:

```python
# ⚠️ Modify these paths before running
FILE_A       = "./eval.jsonl"              # input: predictions JSONL from Step 3
FILE_B       = "./finalentities.json"      # Freebase ID → name lookup table
FILE_C       = "output/mapping.json"       # output: mapped result file
FILE_MISSING = './missing_entities1.txt'   # IDs not found in the lookup table
```

```bash
python mapping.py
```

Any entity IDs absent from `finalentities.json` are left as-is and logged to `missing_entities1.txt`.

---

## Quick Reference

```bash
# 1. Create environment
conda create -n stage1 python=3.10 -y && conda activate stage1
pip install -r requirements.txt

# 2. Train
python select_path_last_train.py \
    --train_file ./data/WebQSP/train.json \
    --test_file  ./data/WebQSP/test.json \
    --base_model bert-base-uncased \
    --output_dir ./retrieval_result \
    --batch_size 32 --epochs 5 --lr 2e-5

# 3. Evaluate  (edit paths in select_path_eval.py first)
python select_path_eval.py

# 4. Map entity IDs  (edit paths in mapping.py first)
python mapping.py
```
